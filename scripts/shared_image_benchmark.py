#!/usr/bin/env python3
"""Measure request-driven image switching while text streams from a test listener."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request

PROMPT = "A red ceramic mug beside a blue book on a light wooden table, studio photograph."
TEXT = "Write a long, detailed manual for setting up and operating a pottery studio."


def request(args, path, payload=None, *, timeout=None):
    headers = {}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(args.url + path, data=data, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout or args.timeout)
    except urllib.error.HTTPError as exc:
        with exc:
            detail = exc.read(4096).decode(errors="replace")
        raise RuntimeError(f"{path}: HTTP {exc.code}: {detail}") from exc


def decoded(args, path, payload=None):
    with request(args, path, payload) as response:
        return json.load(response)


def stream_text(args, events, first, stop, errors, *, receipt_lock=None):
    if receipt_lock is None:
        receipt_lock = threading.Lock()
    payload = {
        "model": args.text_model, "stream": True, "max_tokens": 512,
        "temperature": 0, "seed": args.seed,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": TEXT}],
    }
    try:
        with (args.output / "text_streams.jsonl").open("w") as log:
            while not stop.is_set():
                begin = time.monotonic()
                chunk_times = []
                completion_tokens = None
                with request(args, "/v1/chat/completions", payload, timeout=args.timeout) as response:
                    completed = False
                    for line in response:
                        if stop.is_set():
                            return
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            completed = True
                            break
                        event = json.loads(data)
                        if "error" in event:
                            raise RuntimeError(str(event["error"]))
                        usage = event.get("usage")
                        if usage is not None:
                            completion_tokens = usage.get("completion_tokens")
                            if completion_tokens is not None and (
                                    type(completion_tokens) is not int or completion_tokens < 0):
                                raise RuntimeError("invalid completion_tokens in text stream usage")
                        deltas = [choice.get("delta", {}) for choice in event.get("choices", [])]
                        if any(delta.get("content") or delta.get("reasoning_content") for delta in deltas):
                            now = time.monotonic()
                            events.append(now)
                            chunk_times.append(now)
                            first.set()
                    if not completed:
                        raise RuntimeError("text stream closed before [DONE]")
                elapsed = time.monotonic() - begin
                record = {
                    "started_monotonic_s": begin,
                    "elapsed_s": elapsed,
                    "first_output_s": chunk_times[0] - begin if chunk_times else None,
                    "output_chunks": len(chunk_times),
                    "gaps_between_chunks_s": [b - a for a, b in zip(chunk_times, chunk_times[1:])],
                    "completion_tokens": completion_tokens,
                    "completion_tokens_per_s": (
                        completion_tokens / elapsed if completion_tokens is not None else None
                    ),
                }
                with receipt_lock:
                    if stop.is_set():
                        return
                    log.write(json.dumps(record) + "\n")
                    log.flush()
    except Exception as exc:
        if not stop.is_set():
            errors.append(str(exc))
            first.set()


def run_image(args, model, label, events, expected_state, path, fields):
    before = None
    if expected_state is not None:
        before = decoded(args, "/v1/images/status")
        states = {item["id"]: item["state"] for item in before["models"]}
        if not before["cache_status_available"] or before["queued"] or before["activating_model"]:
            raise RuntimeError(f"{label}: the image listener must be idle before each sample")
        if states.get(model) != expected_state:
            raise RuntimeError(
                f"{label}: expected {model} to be {expected_state}, got {states.get(model)}")
    payload = {
        "model": model, "size": f"{args.size}x{args.size}", "seed": args.seed,
        "num_inference_steps": args.steps, "output_format": "png", **fields,
    }
    begin = time.monotonic()
    with request(args, path, payload) as response:
        server_timing = response.headers.get("Server-Timing")
        reply = json.load(response)
    end = time.monotonic()
    image, = reply["data"]
    png = base64.b64decode(image["b64_json"], validate=True)
    if (png[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
            or struct.unpack(">II", png[16:24]) != (args.size, args.size)):
        raise RuntimeError(f"{label}: expected a {args.size}x{args.size} PNG")
    if image["seed"] != args.seed:
        raise RuntimeError(f"{label}: response seed {image['seed']} differs from {args.seed}")
    status = None
    if before is not None:
        status = decoded(args, "/v1/images/status")
        if status["active_model"] != model:
            raise RuntimeError(f"{label}: expected active model {model}, got {status['active_model']}")
    points = [begin, *(event for event in list(events) if begin <= event <= end), end]
    record = {
        "label": label, "model": model, "seed": image["seed"], "endpoint": path,
        "request_s": end - begin, "server_timing": server_timing,
        "request_started_s": begin, "response_read_s": end,
        "text_output_chunks": len(points) - 2,
        "max_text_silence_s": max(b - a for a, b in zip(points, points[1:])),
        "image_sha256": hashlib.sha256(png).hexdigest(), "status_before": before,
        "status_after": status,
    }
    (args.output / f"{label}.png").write_bytes(png)
    print(json.dumps(record), flush=True)
    return record


def verify_warm_queue_state(args, status, label):
    states = {item["id"]: item["state"] for item in status["models"]}
    models = {args.qwen_model, args.krea_model}
    active = status["active_model"]
    expected = {model: "active" if model == active else "host_cached" for model in models}
    actual = {model: states.get(model) for model in models}
    if (not status["cache_status_available"] or status["queued"] or status["activating_model"]
            or active not in models or actual != expected):
        raise RuntimeError(f"{label}: expected one active model and one host snapshot, got {status}")


def run_queue_bursts(args, events, errors, qwen_path, qwen_fields, krea_fields, baseline):
    # Launch a mixed burst against the model state left by the previous
    # round. Keep response labels independent of completion order.
    jobs = [
        ("qwen-first", args.qwen_model, qwen_path, qwen_fields),
        ("krea-base", args.krea_model, "/v1/images/generations", {"prompt": PROMPT}),
        ("qwen-later", args.qwen_model, qwen_path, qwen_fields),
        ("krea-adapted", args.krea_model, "/v1/images/generations", krea_fields),
    ]
    started = time.monotonic()
    deadline = started + args.queue_seconds
    burst = 0
    with ((args.output / "queue.jsonl").open("w") as log,
          (args.output / "queue_boundaries.jsonl").open("w") as boundaries):
        with ThreadPoolExecutor(max_workers=len(jobs)) as workers:
            while burst < args.queue_bursts or time.monotonic() < deadline:
                burst += 1
                start = decoded(args, "/v1/images/status")
                if args.require_warm_queue:
                    boundaries.write(json.dumps({"burst": burst, "monotonic_s": time.monotonic(), "start": start}) + "\n")
                    boundaries.flush()
                    verify_warm_queue_state(args, start, f"start of burst {burst}")
                futures = {}
                for i, (label, model, path, fields) in enumerate(jobs):
                    if i:
                        time.sleep(args.queue_interval)
                    future = workers.submit(run_image, args, model, f"queue-{burst:03d}-{label}",
                                            events, None, path, fields)
                    futures[future] = label
                records = []
                failures = []
                for future in as_completed(futures):
                    label = futures[future]
                    try:
                        record = future.result()
                        if record["image_sha256"] != baseline[label]:
                            raise RuntimeError(f"{record['label']}: seeded output differs from baseline")
                        header = record["server_timing"] or ""
                        durations = {}
                        selection = None
                        for entry in header.split(","):
                            metric, *params = entry.strip().split(";")
                            if metric not in {"image_queue", "image_selection", "image_activation"}:
                                continue
                            for param in params:
                                name, _, value = param.partition("=")
                                if name == "dur" and metric != "image_activation":
                                    durations[metric] = float(value)
                                if metric == "image_activation" and name == "desc":
                                    selection = value.strip('"')
                        if args.require_warm_queue and selection not in {"active", "warm"}:
                            raise RuntimeError(
                                f"{record['label']}: expected warm restore or active reuse, got {selection!r}")
                        record["selection"] = selection
                        if len(durations) != 2 or any(
                                not math.isfinite(value) or value < 0 for value in durations.values()):
                            raise RuntimeError(f"{record['label']}: invalid Server-Timing header")
                    except Exception as error:
                        failures.append(f"{label}: {error}")
                        continue
                    if record["text_output_chunks"] == 0:
                        failures.append(f"{label}: no text output during image request")
                        continue
                    record["queue_ms"] = durations["image_queue"]
                    record["selection_ms"] = durations["image_selection"]
                    record["burst"] = burst
                    record["burst_start_status"] = start
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    records.append(record)
                if failures:
                    raise RuntimeError(f"burst {burst} failed: {'; '.join(failures)}")
                if args.require_warm_queue:
                    end = decoded(args, "/v1/images/status")
                    boundaries.write(json.dumps({"burst": burst, "monotonic_s": time.monotonic(), "end": end}) + "\n")
                    boundaries.flush()
                    verify_warm_queue_state(args, end, f"end of burst {burst}")
                    if args.queue_sample_seconds:
                        # Let the driver retire the last response before sampling.
                        drain_deadline = time.monotonic() + min(30, args.timeout)
                        while end["adapter_request_holds"] or end["adapter_snapshot_bytes"]:
                            if time.monotonic() >= drain_deadline:
                                raise RuntimeError(f"burst {burst}: adapter holds did not retire")
                            time.sleep(0.1)
                            end = decoded(args, "/v1/images/status")
                        verify_warm_queue_state(args, end, f"memory sample for burst {burst}")
                        opened = time.monotonic()
                        time.sleep(args.queue_sample_seconds)
                        confirmed = decoded(args, "/v1/images/status")
                        if confirmed != end:
                            raise RuntimeError(f"burst {burst}: image state changed during memory sampling")
                        boundaries.write(json.dumps({
                            "burst": burst, "sample_start_s": opened,
                            "sample_end_s": time.monotonic(), "sample_state": end,
                        }) + "\n")
                        boundaries.flush()
                if not any(record["queue_ms"] >= 1 for record in records):
                    raise RuntimeError(f"burst {burst} did not exercise the queue; reduce --queue-interval")
                if errors:
                    raise RuntimeError(f"text stream failed: {errors}")
    return {"queue_bursts_completed": burst, "queue_elapsed_s": time.monotonic() - started}


def verify_unload(args):
    """Drain both registered generators and verify that text still serves."""
    path = "/v1/images/status"
    before = decoded(args, path)
    before_s = time.monotonic()
    verify_warm_queue_state(args, before, "before unloading")
    models = {args.qwen_model, args.krea_model}
    inactive, = models - {before["active_model"]}
    for model in (inactive, before["active_model"]):
        reply = decoded(args, "/v1/images/unload_model", {"model": model})
        if reply.get("model") != model or reply.get("status") != "unloaded":
            raise RuntimeError(f"unexpected unload response for {model}: {reply}")
    after = decoded(args, path)
    after_s = time.monotonic()
    states = {item["id"]: item["state"] for item in after["models"] if item["id"] in models}
    if (not after["cache_status_available"] or after["queued"] or after["active_model"]
            or after["activating_model"] or after["inactive_cache_bytes"] != 0
            or states != dict.fromkeys(models, "registered")):
        raise RuntimeError(f"expected both models unloaded and an empty cache: {after}")
    if after.get("adapter_request_holds") != 0 or after.get("adapter_snapshot_bytes") != 0:
        raise RuntimeError(f"expected all adapter holds and source bytes released: {after}")
    reply = decoded(args, "/v1/chat/completions", {
        "model": args.text_model, "temperature": 0, "seed": args.seed, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Name one tool used in pottery."}],
    })
    choices = reply.get("choices", [])
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message", {})
    output = tuple(message.get(key) for key in ("content", "reasoning_content"))
    tokens = reply.get("usage", {}).get("completion_tokens")
    if (reply.get("model") != args.text_model or choice.get("finish_reason") not in {"stop", "length"}
            or not any(isinstance(value, str) and value.strip() for value in output)
            or type(tokens) is not int or tokens <= 0):
        raise RuntimeError(f"text completion failed after unloading generators: {reply}")
    return {"before": before, "before_monotonic_s": before_s, "after": after,
            "after_monotonic_s": after_s, "text_completed_monotonic_s": time.monotonic(),
            "text_completion_tokens": tokens}


def distribution(values):
    """Nearest-rank percentiles over observed samples."""
    samples = sorted(value for value in values if value is not None)
    summary = {"count": len(samples)}
    for label, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100)):
        summary[label] = samples[(percentile * len(samples) + 99) // 100 - 1] if samples else None
    return summary


def summarize_timings(output, queue_enabled):
    with (output / "text_streams.jsonl").open() as log:
        streams = [json.loads(line) for line in log]
    with (output / "requests.jsonl").open() as log:
        sequential = [json.loads(line) for line in log]
    queued = []
    if queue_enabled:
        with (output / "queue.jsonl").open() as log:
            queued = [json.loads(line) for line in log]
    return {
        "text": {
            "completed_streams": len(streams),
            "first_output_s": distribution(row["first_output_s"] for row in streams),
            "gap_between_chunks_s": distribution(
                gap for row in streams for gap in row["gaps_between_chunks_s"]),
            "completion_tokens_per_s": distribution(row["completion_tokens_per_s"] for row in streams),
        },
        "images": {
            "sequential_request_s": distribution(row["request_s"] for row in sequential),
            "queued_request_s": distribution(row["request_s"] for row in queued),
            "queue_ms": distribution(row["queue_ms"] for row in queued),
            "selection_ms": distribution(row["selection_ms"] for row in queued),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18094")
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--krea-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-return", required=True, choices=["cold", "warm"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=720)
    parser.add_argument("--qwen-reference", type=Path, help="PNG reference for both Qwen edits")
    parser.add_argument("--qwen-edit-prompt", default="Change the mug to green. Preserve the scene.")
    parser.add_argument("--krea-lora", action="append", default=[], metavar="NAME=STRENGTH",
                        help="Krea adapter selection, repeatable")
    parser.add_argument("--revisit-krea", action="store_true",
                        help="repeat the same Krea request after the Qwen round trip")
    parser.add_argument("--queue-bursts", type=int, default=0,
                        help="number of four-request bursts after the sequential benchmark")
    parser.add_argument("--queue-seconds", type=float, default=0,
                        help="minimum time in mixed bursts, excluding sequential warmup")
    parser.add_argument("--require-warm-queue", action="store_true",
                        help="require active and host-cached model states at every burst boundary")
    parser.add_argument("--verify-unload", action="store_true",
                        help="unload both generators after the warm queue test and check text")
    parser.add_argument("--queue-sample-seconds", type=float, default=0,
                        help="seconds to hold the image queue idle after each verified warm burst")
    parser.add_argument("--queue-interval", type=float, default=0.1,
                        help="seconds between submissions within a burst")
    args = parser.parse_args()
    if args.queue_bursts < 0 or not math.isfinite(args.queue_interval) or args.queue_interval <= 0:
        parser.error("queue bursts must be nonnegative and the queue interval must be positive")
    if not math.isfinite(args.queue_seconds) or args.queue_seconds < 0:
        parser.error("--queue-seconds must be finite and nonnegative")
    queue_enabled = bool(args.queue_bursts or args.queue_seconds)
    if args.require_warm_queue and (not queue_enabled or args.expect_return != "warm"):
        parser.error("--require-warm-queue requires queue bursts or duration, and --expect-return warm")
    if not math.isfinite(args.queue_sample_seconds) or not 0 <= args.queue_sample_seconds <= 30:
        parser.error("--queue-sample-seconds must be between 0 and 30")
    if args.queue_sample_seconds and not args.require_warm_queue:
        parser.error("--queue-sample-seconds requires --require-warm-queue")
    if args.verify_unload and not args.require_warm_queue:
        parser.error("--verify-unload requires --require-warm-queue")
    qwen_fields = {"prompt": PROMPT}
    qwen_path = "/v1/images/generations"
    reference_hash = None
    if args.qwen_reference:
        reference = args.qwen_reference.read_bytes()
        if not reference.startswith(b"\x89PNG\r\n\x1a\n"):
            parser.error("--qwen-reference must be a PNG")
        reference_hash = hashlib.sha256(reference).hexdigest()
        qwen_path = "/v1/images/edits"
        qwen_fields = {
            "prompt": args.qwen_edit_prompt,
            "images": [{"image_url": "data:image/png;base64," + base64.b64encode(reference).decode("ascii")}],
        }
    krea_loras = {}
    for selection in args.krea_lora:
        name, separator, strength = selection.rpartition("=")
        try:
            weight = float(strength)
        except ValueError:
            parser.error("--krea-lora requires NAME=STRENGTH")
        if not separator or not name or name in krea_loras or not math.isfinite(weight) or abs(weight) > 16:
            parser.error("--krea-lora requires a unique name and a finite strength in [-16, 16]")
        krea_loras[name] = weight
    if queue_enabled and not any(krea_loras.values()):
        parser.error("mixed queue tests require at least one nonzero --krea-lora")
    krea_fields = {"prompt": PROMPT}
    if krea_loras:
        krea_fields["loras"] = krea_loras
    args.url = args.url.rstrip("/")
    args.api_key = os.environ.get("RADIANCE_API_KEY")
    ids = {args.text_model, args.qwen_model, args.krea_model}
    if len(ids) != 3 or min(args.size, args.steps, args.timeout) <= 0:
        parser.error("models must be distinct and size, steps, and timeout must be positive")
    catalog = {item["id"] for item in decoded(args, "/v1/models")["data"]}
    if not ids <= catalog:
        parser.error("models missing from the listener: " + ", ".join(sorted(ids - catalog)))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(json.dumps({
        "text_model": args.text_model, "qwen_model": args.qwen_model, "krea_model": args.krea_model,
        "expected_return": args.expect_return, "size": args.size, "steps": args.steps,
        "seed": args.seed, "text_max_tokens": 512, "qwen_endpoint": qwen_path,
        "qwen_prompt": qwen_fields["prompt"], "qwen_reference_sha256": reference_hash,
        "krea_prompt": PROMPT, "krea_loras": krea_loras,
        "revisit_krea": args.revisit_krea, "queue_bursts": args.queue_bursts, "queue_interval_s": args.queue_interval,
        "require_warm_queue": args.require_warm_queue, "queue_minimum_s": args.queue_seconds,
        "verify_unload": args.verify_unload,
        "queue_sample_seconds": args.queue_sample_seconds,
    }, indent=2) + "\n")
    events, errors = [], []
    first, stop = threading.Event(), threading.Event()
    receipt_lock = threading.Lock()
    worker = threading.Thread(target=stream_text,
                              args=(args, events, first, stop, errors),
                              kwargs={"receipt_lock": receipt_lock}, daemon=True)
    worker.start()
    try:
        if not first.wait(min(60, args.timeout)) or not events:
            raise RuntimeError(f"no content from the text stream: {errors}")
        return_state = "host_cached" if args.expect_return == "warm" else "registered"
        sequential_jobs = [(args.qwen_model, "qwen-before", "active", qwen_path, qwen_fields)]
        krea_path = "/v1/images/generations"
        krea_state = "registered"
        if queue_enabled:
            sequential_jobs.append((args.krea_model, "krea-base", krea_state, krea_path,
                                    {"prompt": PROMPT}))
            krea_state = "active"
        sequential_jobs.extend([
            (args.krea_model, "krea", krea_state, krea_path, krea_fields),
            (args.qwen_model, "qwen-return", return_state, qwen_path, qwen_fields),
        ])
        if args.revisit_krea:
            sequential_jobs.append((args.krea_model, "krea-return", return_state, krea_path,
                                    krea_fields))
        hashes = {}
        with (args.output / "requests.jsonl").open("w") as log:
            for model, label, state, path, fields in sequential_jobs:
                record = run_image(args, model, label, events, state, path, fields)
                log.write(json.dumps(record) + "\n")
                log.flush()
                hashes[label] = record["image_sha256"]
                if errors:
                    raise RuntimeError(f"text stream failed: {errors}")
        if hashes["qwen-before"] != hashes["qwen-return"]:
            raise RuntimeError("seeded Qwen image changed after the round trip")
        if args.revisit_krea and hashes["krea"] != hashes["krea-return"]:
            raise RuntimeError("seeded Krea image changed after the round trip")
        queue_summary = {}
        if queue_enabled:
            baseline = {"qwen-first": hashes["qwen-before"],
                        "krea-base": hashes["krea-base"],
                        "qwen-later": hashes["qwen-before"],
                        "krea-adapted": hashes["krea"]}
            queue_summary = run_queue_bursts(
                args, events, errors, qwen_path, qwen_fields, krea_fields, baseline)
        image_done = time.monotonic()
        deadline = image_done + min(60, args.timeout)
        while not errors and events[-1] <= image_done and time.monotonic() < deadline:
            time.sleep(0.05)
        if errors:
            raise RuntimeError(f"text stream failed: {errors}")
        if events[-1] <= image_done:
            raise RuntimeError("text did not resume after the final image")
        next_chunk = next(event for event in list(events) if event > image_done)
        summary = {"text_resume_s": next_chunk - image_done, "qwen_round_trip_match": True, **queue_summary}
        if args.revisit_krea:
            summary["krea_round_trip_match"] = True
        with receipt_lock:
            stop.set()
        if args.verify_unload:
            worker.join(timeout=min(60, args.timeout))
            if worker.is_alive():
                raise RuntimeError("text worker did not stop before verifying unload")
            if errors:
                raise RuntimeError(f"text stream failed: {errors}")
            summary["unload"] = verify_unload(args)
        summary["timings"] = summarize_timings(args.output, queue_enabled)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        stop.set()
        worker.join(timeout=1)


if __name__ == "__main__":
    main()
