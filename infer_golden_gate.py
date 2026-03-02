"""
Example script to call SAM3 inference on the Golden Gate Bridge image.

This script downloads an image from Wikimedia and sends it to the Modal endpoint
for segmentation with a text prompt.
"""

import requests
import base64
import json
from io import BytesIO
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image, ImageDraw

# Load environment variables from .env file
load_dotenv()


def infer_image_from_url(modal_url: str, image_url: str, prompt: str) -> dict:
    """
    Download an image from a URL and call the image inference endpoint.

    Args:
        modal_url: Base URL of your Modal deployment
        image_url: URL of the image to download and process
        prompt: Text prompt for segmentation

    Returns:
        Response dict with segmentation results
    """
    # Download image from URL
    print(f"Downloading image from: {image_url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()

    image_data = response.content
    image_base64 = base64.b64encode(image_data).decode()

    # Call Modal endpoint
    print(f"Calling Modal endpoint: {modal_url}")
    inference_response = requests.post(
        modal_url,
        json={
            "image_base64": image_base64,
            "prompt": prompt,
        },
    )
    inference_response.raise_for_status()

    return inference_response.json(), image_data


def save_result_png(image_data: bytes, result: dict, prompt: str, output_path: str):
    image = Image.open(BytesIO(image_data)).convert("RGBA")
    data = result.get("data", {})

    masks = data.get("masks", [])
    boxes = data.get("boxes", [])
    scores = data.get("scores", [])

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, mask in enumerate(masks):
        # mask shape: [1, H, W] — take first channel
        mask_arr = np.array(mask[0], dtype=bool)
        color = (255, 50, 50, 120)  # semi-transparent red
        mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
        colored = Image.new("RGBA", image.size, color)
        overlay.paste(colored, mask=mask_img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[i] if i < len(scores) else 0
        draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50, 255), width=3)
        draw.text((x1 + 4, y1 + 4), f"{prompt} {score:.2f}", fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    combined.save(output_path)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    # Golden Gate Bridge image URL from Wikimedia Commons
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Golden_Gate_Bridge_as_seen_from_Battery_East.jpg/1920px-Golden_Gate_Bridge_as_seen_from_Battery_East.jpg"

    # Load Modal endpoint from environment variables
    MODAL_ENDPOINT = os.getenv("IMAGE_ENDPOINT")

    # Text prompt for segmentation
    PROMPT = "bridge"

    try:
        print(f"Running SAM3 inference with prompt: '{PROMPT}'")
        print("-" * 60)

        result, image_data = infer_image_from_url(MODAL_ENDPOINT, IMAGE_URL, PROMPT)

        print("Inference completed successfully!")
        print("-" * 60)
        scores = result.get("data", {}).get("scores", [])
        boxes = result.get("data", {}).get("boxes", [])
        print(f"Detections: {len(scores)}")
        for i, (score, box) in enumerate(zip(scores, boxes)):
            print(f"  [{i}] score={score:.3f}  box={[round(v, 1) for v in box]}")

        save_result_png(image_data, result, PROMPT, "output.png")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Modal endpoint at {MODAL_ENDPOINT}")
        print("Make sure to:")
        print("1. Run 'modal run modal_app.py' in the project directory")
        print("2. Update MODAL_ENDPOINT with the URL shown in the terminal")
    except Exception as e:
        print(f"Error: {e}")
        raise
