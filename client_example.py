"""
Example client for SAM3 inference endpoints on Modal.

This demonstrates how to use the modal_app endpoints.
"""

import requests
import base64
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def infer_image(modal_url: str, image_path: str, prompt: str) -> dict:
    """
    Call the image inference endpoint.

    Args:
        modal_url: Base URL of your Modal deployment (e.g., https://your-workspace--sam3-inference-infer-image.modal.run)
        image_path: Path to the image file
        prompt: Text prompt for segmentation

    Returns:
        Response dict with masks, boxes, and scores
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode()

    # Call endpoint
    response = requests.post(
        modal_url,
        json={
            "image_base64": image_base64,
            "prompt": prompt,
        },
    )

    return response.json()


def start_video_session(modal_url: str, video_path: str) -> dict:
    """
    Start a video inference session.

    Args:
        modal_url: Base URL of your Modal deployment (e.g., https://your-workspace--sam3-inference-infer-video.modal.run)
        video_path: Path to video file or JPEG frame folder

    Returns:
        Response dict with session_id
    """
    response = requests.post(
        modal_url,
        json={
            "action": "start_session",
            "video_path": video_path,
        },
    )

    return response.json()


def add_video_prompt(modal_url: str, session_id: str, frame_index: int, prompt: str) -> dict:
    """
    Add a text prompt to a video session.

    Args:
        modal_url: Base URL of your Modal deployment
        session_id: Session ID from start_video_session
        frame_index: Frame number to add prompt to
        prompt: Text prompt for segmentation

    Returns:
        Response dict with segmentation results
    """
    response = requests.post(
        modal_url,
        json={
            "action": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "prompt": prompt,
        },
    )

    return response.json()


if __name__ == "__main__":
    # Load endpoints from environment variables
    IMAGE_ENDPOINT = os.getenv("IMAGE_ENDPOINT")
    VIDEO_ENDPOINT = os.getenv("VIDEO_ENDPOINT")

    # Example: Image inference
    print("Image Inference Example:")
    # result = infer_image(IMAGE_ENDPOINT, "path/to/image.jpg", "dog")
    # print(result)

    # Example: Video inference
    print("\nVideo Inference Example:")
    # session = start_video_session(VIDEO_ENDPOINT, "path/to/video.mp4")
    # session_id = session["data"]["session_id"]
    # result = add_video_prompt(VIDEO_ENDPOINT, session_id, 0, "person")
    # print(result)
