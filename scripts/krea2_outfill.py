#!/usr/bin/env python3
"""Outfill an image through Radiance, optionally captioning it with an OpenAI-compatible VLM."""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(base_url, route, body, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url.rstrip('/')}/{route}", json.dumps(body).encode(), headers)
    try:
        with urlopen(request, timeout=900) as response:
            return json.load(response)
    except HTTPError as error:
        with error:
            raise RuntimeError(f"{route} returned HTTP {error.code}") from error


def image_url(path):
    suffix = path.suffix.lower()
    types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError("the input file must be PNG, JPEG or WebP")
    return f"data:{types[suffix]};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def caption(url, model, source, key, *, radiance=False):
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source}},
            {"type": "text", "text": (
                "Describe the visible scene in one or two sentences for an image outpainting model. "
                "Include the main subject, pose, clothing if visible, perspective, lighting, and "
                "background details that should continue outside the frame. Describe only visible details. "
                "Return only the scene description."
            )},
        ]}],
    }
    if radiance:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    response = post(url, "chat/completions", request, key)
    choice = response["choices"][0]
    finish_reason = choice["finish_reason"]
    if finish_reason != "stop":
        raise ValueError(f"the vision endpoint did not finish the scene description ({finish_reason})")
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the vision endpoint returned an empty scene description")
    return content.strip()


def describe_scene(args, source, radiance_key):
    if args.prompt is not None:
        return args.prompt
    if args.primary_caption_model:
        try:
            return caption(args.url, args.primary_caption_model, source, radiance_key, radiance=True)
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError):
            if not args.caption_url:
                raise
            print("Primary caption failed. Using the fallback endpoint.", file=sys.stderr)
    return caption(args.caption_url, args.caption_model, source, os.getenv("CAPTION_API_KEY"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8002/v1", help="Radiance API base URL")
    parser.add_argument("--model", default="krea-2")
    parser.add_argument("--primary-caption-model", help="vision-capable text model ID at --url")
    parser.add_argument("--caption-url", help="separate OpenAI-compatible vision API base URL")
    parser.add_argument("--caption-model", help="model ID at the vision endpoint")
    parser.add_argument("--prompt", help="scene description supplied directly")
    parser.add_argument("--horizontal", choices=("left", "center", "right"), default="left")
    parser.add_argument("--vertical", choices=("top", "center", "bottom"), default="top")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turbo-lora", default="krea2_turbo_lora_rank_64_bf16")
    parser.add_argument("--outfill-lora", default="krea2_outpaint_rank32")
    args = parser.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must describe the scene")
    if bool(args.caption_url) != bool(args.caption_model):
        parser.error("--caption-url and --caption-model must be supplied together")
    if args.prompt is None and not (args.primary_caption_model or args.caption_url):
        parser.error("supply --prompt, --primary-caption-model, or a separate caption endpoint")

    try:
        source = image_url(args.source)
        radiance_key = os.getenv("RADIANCE_API_KEY")
        prompt = describe_scene(args, source, radiance_key)
        print(f"Scene: {prompt}", flush=True)
        result = post(args.url, "images/edits", {
            "model": args.model,
            "image": source,
            "prompt": prompt,
            "size": "1024x1024",
            "outfill": {"horizontal": args.horizontal, "vertical": args.vertical},
            "num_inference_steps": 8,
            "guidance_scale": 0,
            "seed": args.seed,
            "loras": [
                {"name": args.turbo_lora, "weight": 0.6},
                {"name": args.outfill_lora, "weight": 1.0},
            ],
        }, radiance_key)
        data = result["data"][0]
        args.output.write_bytes(base64.b64decode(data["b64_json"], validate=True))
        print(f"Saved {args.output} (seed {data['seed']})")
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
