"""
Script to process video frames through SAM3 inference in parallel.

Reads frames from a local video file, sends them to Modal endpoint for
segmentation with a text prompt, and reconstructs the output video with
segmentation results overlaid.
"""

import cv2
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import requests

# Load environment variables from .env file
load_dotenv()


def process_frame(frame_data: dict) -> dict:
    """
    Send a single frame to the inference endpoint and return results.

    Args:
        frame_data: Dict with 'frame_num', 'frame_bytes', and 'modal_url' keys

    Returns:
        Dict with frame_num and inference results
    """
    frame_num = frame_data['frame_num']
    frame_bytes = frame_data['frame_bytes']
    modal_url = frame_data['modal_url']
    prompt = frame_data['prompt']

    try:
        frame_base64 = base64.b64encode(frame_bytes).decode()

        response = requests.post(
            modal_url,
            json={
                "image_base64": frame_base64,
                "prompt": prompt,
            },
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return {
            'frame_num': frame_num,
            'success': True,
            'result': result
        }
    except Exception as e:
        return {
            'frame_num': frame_num,
            'success': False,
            'error': str(e)
        }


def overlay_results_on_frame(frame: np.ndarray, result: dict, resized_width: int) -> np.ndarray:
    """
    Overlay segmentation masks on a frame, scaling from resized inference space to original frame.

    Args:
        frame: BGR numpy array (original frame size)
        result: Inference result dict (in resized frame coordinates)
        resized_width: Width of the frame used for inference

    Returns:
        BGR numpy array with mask overlays
    """
    try:
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame).convert("RGBA")

        data = result.get("data", {})
        masks = data.get("masks", [])

        # Calculate scale factor from resized frame to original frame
        original_width = image.size[0]
        scale_factor = original_width / resized_width

        # Create overlay for masks
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

        # Draw masks
        for i, mask in enumerate(masks):
            try:
                mask_arr = np.array(mask[0], dtype=bool)

                # Scale mask back to original frame size
                resized_height = mask_arr.shape[0]
                original_height = int(resized_height * scale_factor)
                scaled_mask = cv2.resize(
                    mask_arr.astype(np.uint8) * 255,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR
                )
                scaled_mask = scaled_mask > 127  # Convert back to binary

                # Draw scaled mask in green
                color = (50, 255, 50, 120)  # semi-transparent green (RGBA)
                mask_img = Image.fromarray(scaled_mask.astype(np.uint8) * 255, mode="L")
                colored = Image.new("RGBA", image.size, color)
                overlay.paste(colored, mask=mask_img)
            except Exception as e:
                print(f"Warning: failed to draw mask {i} ({e}), skipping")
                continue

        # Composite and convert back to BGR
        combined = Image.alpha_composite(image, overlay).convert("RGB")
        bgr_result = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

        return bgr_result
    except Exception as e:
        # If overlay fails, just return original frame
        print(f"Warning: overlay failed ({e}), returning original frame")
        return frame


if __name__ == "__main__":
    # Video file path
    VIDEO_PATH = "camera_front_wide_120fov.mp4"
    OUTPUT_PATH = "output_vehicle_segmented.mp4"
    FRAMES_DIR = "frames"
    MAX_FRAMES = 60

    # Load Modal endpoint from environment variables
    MODAL_ENDPOINT = os.getenv("IMAGE_ENDPOINT")
    if not MODAL_ENDPOINT:
        raise ValueError("IMAGE_ENDPOINT environment variable not set. Set it in .env or export it.")

    # Text prompt for segmentation
    PROMPT = "vehicle"

    # Parallel processing settings
    N_WORKERS = 10
    TIMEOUT = 60  # seconds per frame

    try:
        print(f"Processing video: {VIDEO_PATH}")
        print(f"Prompt: '{PROMPT}'")
        print(f"Max frames: {MAX_FRAMES}")
        print(f"Workers: {N_WORKERS}")
        print("-" * 60)

        # Create frames directory
        Path(FRAMES_DIR).mkdir(exist_ok=True)
        print(f"Created/using directory: {FRAMES_DIR}")

        # Open video
        if not Path(VIDEO_PATH).exists():
            raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {VIDEO_PATH}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames total")
        print("-" * 60)

        # Read frames (only first MAX_FRAMES)
        print(f"Reading first {MAX_FRAMES} frames...")
        frames_original = []
        frames_resized = []
        frame_data_list = []
        resized_width = None

        with tqdm(total=MAX_FRAMES, desc="Reading frames") as pbar:
            frame_count = 0
            while frame_count < MAX_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                # Keep original frame
                frames_original.append(frame.copy())

                # Downsize frame before encoding (aggressive resize for speed)
                frame_resized = frame.copy()
                max_width = 512
                if frame_resized.shape[1] > max_width:
                    scale = max_width / frame_resized.shape[1]
                    new_height = int(frame_resized.shape[0] * scale)
                    frame_resized = cv2.resize(frame_resized, (max_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    if resized_width is None:
                        resized_width = max_width

                frames_resized.append(frame_resized)

                # Prepare frame data for inference
                _, frame_bytes = cv2.imencode('.jpg', frame_resized)
                frame_data_list.append({
                    'frame_num': frame_count,
                    'frame_bytes': frame_bytes.tobytes(),
                    'modal_url': MODAL_ENDPOINT,
                    'prompt': PROMPT
                })

                frame_count += 1
                pbar.update(1)

        cap.release()

        if not frames_original:
            raise ValueError(f"No frames loaded from {VIDEO_PATH}")

        print(f"Loaded {len(frames_original)} frames")
        print("-" * 60)

        # Process frames in parallel
        print("Processing frames with SAM3...")
        results = {}

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            # Submit all tasks
            futures = {executor.submit(process_frame, frame_data): frame_data['frame_num']
                      for frame_data in frame_data_list}

            # Process results as they complete
            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    frame_num = result['frame_num']
                    results[frame_num] = result

                    if not result['success']:
                        print(f"\nFrame {frame_num} failed: {result['error']}")

                    pbar.update(1)

        print("-" * 60)

        # Save frames with overlays to directory
        print(f"Saving frames to {FRAMES_DIR}/...")
        output_frame_paths = []

        with tqdm(total=len(frames_original), desc="Saving frames") as pbar:
            for frame_num, frame in enumerate(frames_original):
                if frame_num in results:
                    result = results[frame_num]
                    if result['success']:
                        frame = overlay_results_on_frame(frame, result['result'], resized_width)

                frame_path = Path(FRAMES_DIR) / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                output_frame_paths.append(frame_path)
                pbar.update(1)

        print(f"Saved {len(output_frame_paths)} frames to {FRAMES_DIR}/")
        print("-" * 60)

        # Merge frames into video
        print(f"Merging frames into video: {OUTPUT_PATH}")
        output_height, output_width = frames_original[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (output_width, output_height))

        with tqdm(total=len(output_frame_paths), desc="Writing video") as pbar:
            for frame_path in output_frame_paths:
                frame = cv2.imread(str(frame_path))
                out.write(frame)
                pbar.update(1)

        out.release()
        print(f"Saved output to: {OUTPUT_PATH}")

        # Summary
        print("-" * 60)
        successful = sum(1 for r in results.values() if r['success'])
        print(f"Processing complete: {successful}/{len(frames_original)} frames successful")

    except Exception as e:
        print(f"Error: {e}")
        raise
