import modal
import io
import base64

# Create a Modal app
app = modal.App("sam3-inference")

# Volume to cache HuggingFace model weights across container restarts
hf_cache_vol = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# Create an image with SAM3 dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "python3-dev")
    .pip_install(
        "pillow",
        "fastapi",
        "pydantic",
        "numpy",
        "opencv-python",
        "huggingface_hub",
    )
    .pip_install("einops", "decord", "pycocotools", "scikit-image", "scikit-learn", "psutil")
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "python -m pip install --pre git+https://github.com/facebookresearch/sam3.git#egg=sam3",
        "python -c 'from sam3.model_builder import build_sam3_image_model; print(\"sam3 import OK\")'",
    )
)


@app.cls(
    image=image,
    gpu="l40s",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={HF_CACHE_PATH: hf_cache_vol},
    container_idle_timeout=90,
)
class SAM3ImagePredictor:

    @modal.enter()
    def load_model(self):
        import torch
        assert torch.cuda.is_available(), f"CUDA not available! torch version: {torch.__version__}"

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(device="cuda")
        self.processor = Sam3Processor(self.model)

    @modal.fastapi_endpoint(method="POST")
    async def infer(self, request_dict: dict) -> dict:
        try:
            from PIL import Image

            image_base64 = request_dict.get("image_base64")
            prompt = request_dict.get("prompt")

            if not image_base64 or not prompt:
                return {"error": "Missing required fields: 'image_base64' and 'prompt'"}

            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

            return {
                "success": True,
                "data": {
                    "masks": output["masks"].tolist() if hasattr(output["masks"], "tolist") else output["masks"],
                    "boxes": output["boxes"].tolist() if hasattr(output["boxes"], "tolist") else output["boxes"],
                    "scores": output["scores"].tolist() if hasattr(output["scores"], "tolist") else output["scores"],
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


@app.cls(
    image=image,
    gpu="l40s",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={HF_CACHE_PATH: hf_cache_vol},
    container_idle_timeout=90,
)
class SAM3VideoPredictor:

    @modal.enter()
    def load_model(self):
        import torch
        assert torch.cuda.is_available(), f"CUDA not available! torch version: {torch.__version__}"

        from sam3.model_builder import build_sam3_video_predictor

        self.video_predictor = build_sam3_video_predictor(gpus_to_use=[0])

    @modal.fastapi_endpoint(method="POST")
    async def infer(self, request_dict: dict) -> dict:
        try:
            action = request_dict.get("action")

            if action == "start_session":
                video_path = request_dict.get("video_path")
                if not video_path:
                    return {"error": "Missing 'video_path'"}
                result = self.video_predictor.handle_request(
                    request=dict(type="start_session", resource_path=video_path)
                )
                return {"success": True, "data": result}

            elif action == "add_prompt":
                session_id = request_dict.get("session_id")
                frame_index = request_dict.get("frame_index")
                prompt = request_dict.get("prompt")

                if not all([session_id, frame_index is not None, prompt]):
                    return {"error": "Missing required fields: 'session_id', 'frame_index', 'prompt'"}

                result = self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_index,
                        text=prompt,
                    )
                )
                return {"success": True, "data": result}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


@app.function()
@modal.fastapi_endpoint(method="GET")
async def health_check() -> dict:
    return {"status": "healthy", "service": "sam3-inference"}
