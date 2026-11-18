#!/usr/bin/env python3
"""Generate a Radiance MiniMax H3 clip from text, an image, or a new Krea 2 still."""

import argparse
import base64
import binascii
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4


def request(base_url, route, key, body=None, content_type=None, timeout=60):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if content_type:
        headers["Content-Type"] = content_type
    try:
        return urlopen(Request(f"{base_url.rstrip('/')}/{route}", body, headers), timeout=timeout)
    except HTTPError as exc:
        with exc:
            details = exc.read(4096).decode(errors="replace")
        raise RuntimeError(f"{route} returned HTTP {exc.code}: {details}") from exc


def json_request(base_url, route, key, payload=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    with request(base_url, route, key, body, "application/json" if body else None, timeout) as response:
        return json.load(response)


def make_image(args, key):
    if args.source:
        content_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp"}.get(args.source.suffix.lower())
        if not content_type:
            raise ValueError("the input image must be PNG, JPEG, or WebP")
        return args.source.read_bytes(), content_type
    payload = {"model": args.image_model, "prompt": args.image_prompt, "size": args.image_size,
               "seed": args.seed, "num_inference_steps": args.image_steps}
    if args.image_loras:
        payload["loras"] = args.image_loras
    if args.image_guidance is not None:
        payload["guidance_scale"] = args.image_guidance
    start = time.monotonic()
    result = json_request(args.url, "images/generations", key, payload, args.timeout)
    image = base64.b64decode(result["data"][0]["b64_json"], validate=True)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Krea did not return a PNG")
    if args.save_image:
        args.save_image.write_bytes(image)
    print(f"Krea image returned in {time.monotonic() - start:.1f} s", file=sys.stderr)
    return image, "image/png"


def start_video(args, key, image, image_type):
    fields = {"model": args.video_model, "prompt": args.prompt, "size": args.size,
              "seconds": args.seconds, "seed": args.seed}
    if args.video_steps is not None:
        fields["num_inference_steps"] = args.video_steps
    if args.video_loras:
        fields["loras"] = args.video_loras
    if args.text_only:
        fields["task"] = "t2va"
        return json_request(args.url, "videos", key, fields, args.timeout)

    boundary = uuid4().hex
    if args.mode == "first-frame":
        fields.update(task="i2va", first_frame_mode="center")
        image_field = "first_frame"
    else:
        fields["task"] = "ref2va"
        image_field = "ref_image_1"
    parts = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n'
         f'{json.dumps(value) if field == "loras" else value}\r\n').encode()
        for field, value in fields.items()
    ]
    filename = {"image/png": "source.png", "image/jpeg": "source.jpg", "image/webp": "source.webp"}[image_type]
    parts.extend([
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{image_field}"; filename="{filename}"\r\n'
         f"Content-Type: {image_type}\r\n\r\n").encode(),
        image, b"\r\n",
    ])
    if args.reference_video:
        parts.extend([
            (f'--{boundary}\r\nContent-Disposition: form-data; name="ref_video_1"; '
             'filename="reference.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode(),
            args.reference_video.read_bytes(), b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode())
    with request(args.url, "videos", key, b"".join(parts), f"multipart/form-data; boundary={boundary}",
                 args.timeout) as response:
        return json.load(response)


def wait_for_video(args, key, job, started):
    job_id = job["id"]
    route = "videos/" + quote(job_id, safe="")
    deadline = started + args.timeout
    print(f"Video job {job_id}", file=sys.stderr)
    while job.get("status") in ("queued", "in_progress"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"video job {job_id} is still running after {args.timeout} seconds")
        time.sleep(min(1.0, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"video job {job_id} is still running after {args.timeout} seconds")
        job = json_request(args.url, route, key, timeout=remaining)
    if job.get("status") != "completed":
        raise RuntimeError(f"video job {job_id} ended with {job.get('status')}: {job.get('error')}")
    return route, job


def download_video(args, key, route):
    temp_path = None
    try:
        with request(args.url, route + "/content", key, timeout=args.timeout) as response:
            with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as output:
                temp_path = Path(output.name)
                header = response.read(12)
                if len(header) < 12 or header[4:8] != b"ftyp":
                    raise ValueError("video content did not start with an MP4 file header")
                output.write(header)
                shutil.copyfileobj(response, output)
        temp_path.replace(args.output)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def adapter(value):
    try:
        name, weight = value.rsplit("=", 1)
        strength = float(weight)
        if name.strip() and math.isfinite(strength) and abs(strength) <= 16:
            return {"name": name.strip(), "weight": strength}
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use NAME=STRENGTH, between -16 and 16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="existing PNG, JPEG, or WebP file")
    source.add_argument("--image-prompt", help="generate the source image with Krea 2")
    source.add_argument("--text-only", action="store_true", help="generate from text without an image")
    parser.add_argument("--save-image", type=Path, help="save the generated Krea PNG")
    parser.add_argument("--url", default="http://127.0.0.1:8002/v1", help="Radiance API base URL")
    parser.add_argument("--image-model", default="krea-2-raw-edit-iq8r")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-steps", type=int, default=12)
    parser.add_argument("--image-guidance", type=float)
    parser.add_argument("--image-lora", dest="image_loras", action="append", type=adapter, help="NAME=STRENGTH")
    parser.add_argument("--video-model", required=True, help="registered MiniMax H3 model ID for this task")
    parser.add_argument("--mode", choices=("first-frame", "reference"), default="first-frame")
    parser.add_argument("--reference-video", type=Path, help="source MP4 for reference mode")
    parser.add_argument("--video-lora", dest="video_loras", action="append", type=adapter,
                        help="H3 adapter as NAME=STRENGTH")
    parser.add_argument("--prompt", required=True, help="describe the motion and sound, use <Picture 1> for reference")
    parser.add_argument("--size", default="864x480")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--video-steps", type=int, help="optional Radiance H3 schedule override")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for each generation")
    parser.add_argument("--output", type=Path, required=True, help="destination MP4")
    parser.add_argument("--json", action="store_true", help="print job metadata and client timings as JSON")
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.image_steps <= 0:
        parser.error("clip duration, timeout, and image steps must be positive")
    if args.video_steps is not None and args.video_steps <= 0:
        parser.error("video steps must be positive")
    if args.image_guidance is not None and not math.isfinite(args.image_guidance):
        parser.error("image guidance must be finite")
    if not all(re.fullmatch(r"[1-9]\d*x[1-9]\d*", size) for size in (args.size, args.image_size)):
        parser.error("use WIDTHxHEIGHT for image and video sizes")
    if args.save_image and (not args.image_prompt or args.save_image.resolve() == args.output.resolve()):
        parser.error("--save-image needs --image-prompt and a path distinct from --output")
    if args.text_only and args.mode != "first-frame":
        parser.error("--mode reference requires an image")
    if args.reference_video:
        if args.mode != "reference" or args.reference_video.suffix.lower() != ".mp4":
            parser.error("--reference-video requires an MP4 and --mode reference")
        if args.reference_video.resolve() in {args.output.resolve(),
                                             args.save_image.resolve() if args.save_image else None}:
            parser.error("--reference-video must be distinct from the output paths")
    if args.source and args.source.resolve() == args.output.resolve():
        parser.error("--output must be distinct from --source")
    for path in (args.output, args.save_image):
        if path and not path.parent.is_dir():
            parser.error(f"output directory does not exist: {path.parent}")
    key = os.getenv("RADIANCE_API_KEY")
    try:
        if args.reference_video:
            with args.reference_video.open("rb") as source_video:
                if not source_video.read(1):
                    raise ValueError("the reference MP4 is empty")
        models = {item["id"] for item in json_request(args.url, "models", key)["data"]}
        if args.video_model not in models:
            raise ValueError(f"video model {args.video_model!r} is absent from /v1/models")
        image, image_type = (None, None) if args.text_only else make_image(args, key)
        started = time.monotonic()
        job = start_video(args, key, image, image_type)
        submitted = time.monotonic()
        route, job = wait_for_video(args, key, job, started)
        ready = time.monotonic()
        download_video(args, key, route)
        saved = time.monotonic()
        if args.json:
            print(json.dumps({
                "job_id": job["id"], "model": args.video_model,
                "mode": "text-only" if args.text_only else args.mode,
                "seed": job.get("seed", args.seed), "size": job.get("size", args.size),
                "requested_seconds": args.seconds, "server_clip_seconds_text": job.get("seconds"),
                "steps": job.get("steps_total"), "sampler": job.get("sampler"),
                "submit_s": submitted - started, "wait_s": ready - submitted,
                "download_s": saved - ready, "request_to_mp4_s": saved - started,
                "output": str(args.output),
            }))
        else:
            print(f"Saved {args.output} in {saved - started:.1f} s from the start of the video request")
    except (OSError, URLError, RuntimeError, ValueError, KeyError, IndexError, TypeError, binascii.Error) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
