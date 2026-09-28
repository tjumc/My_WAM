#!/usr/bin/env python3
"""One small image/no-image probe through the same Qwen client as the pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "versions/v3_3_3"))
from observe_entities import DEFAULT_BASE_URL, DEFAULT_MODEL, build_client, data_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Existing Qwen input contact sheet")
    parser.add_argument("--without-image", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--aigc-user", default=None)
    args = parser.parse_args()
    args.api_key = None  # Use AIGC_API_KEY from the environment; keep secrets out of shell history.

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    content = [{
        "type": "text",
        "text": "What color is the round plate on the countertop in the supplied image? "
                "If no image is supplied or you cannot determine its color, answer NO_IMAGE. "
                "Reply with only the color or NO_IMAGE.",
    }]
    if not args.without_image:
        content.append({"type": "image_url", "image_url": {"url": data_url(args.image)}})

    response = build_client(args).chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(json.dumps({
        "image_supplied": not args.without_image,
        "requested_base_url": args.base_url or os.getenv("AIGC_BASE_URL", DEFAULT_BASE_URL),
        "requested_model": args.model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "http_request_id": getattr(response, "_request_id", None),
        "answer": response.choices[0].message.content,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
