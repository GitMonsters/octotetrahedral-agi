from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gpu_support import (
    DeviceResolution,
    clear_device_cache,
    prepare_input_tensor,
    resolve_device,
    warmup_device,
)

if TYPE_CHECKING:
    from model import OctoTetrahedralModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = os.getenv("OCTO_CHECKPOINT_PATH", "checkpoints/arc/arc_final.pt")

class InferenceRequest(BaseModel):
    input_ids: list

def _load_state_dict(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _instantiate_model(
    checkpoint_path: str,
    resolution: DeviceResolution,
) -> OctoTetrahedralModel:
    from model import OctoTetrahedralModel

    model = OctoTetrahedralModel()
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(resolution.torch_device)
    warmup_device(resolution.selected)
    logger.info("✅ Model loaded on %s", resolution.selected)
    return model


def _run_inference(
    model: torch.nn.Module,
    input_ids: list,
    device_name: str,
) -> list[list[int]]:
    tensor = prepare_input_tensor(input_ids, device_name)
    with torch.inference_mode():
        output = model(
            input_ids=tensor,
            return_confidences=False,
        )
    return output["logits"].argmax(dim=-1).detach().cpu().tolist()


def _fallback_to_cpu(
    model: torch.nn.Module,
    resolution: DeviceResolution,
    exc: Exception,
) -> DeviceResolution:
    clear_device_cache("mps")
    model.to("cpu")
    model.eval()
    return DeviceResolution(
        requested=resolution.requested,
        selected="cpu",
        backend="cpu",
        fallback_reason=f"Metal inference failed: {exc}",
        cuda_available=resolution.cuda_available,
        mps_available=resolution.mps_available,
        mps_built=resolution.mps_built,
    )


def initialize_runtime(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    requested_device: str | None = None,
) -> tuple[OctoTetrahedralModel | None, DeviceResolution, str | None]:
    resolution = resolve_device(requested_device)
    if not os.path.exists(checkpoint_path):
        message = f"Checkpoint not found at {checkpoint_path}"
        logger.error("❌ %s", message)
        return None, resolution, message

    try:
        return _instantiate_model(checkpoint_path, resolution), resolution, None
    except Exception as exc:
        logger.error("❌ Failed to load model on %s: %s", resolution.selected, exc)
        clear_device_cache(resolution.selected)

        if resolution.selected == "mps":
            fallback_reason = f"Metal backend initialization failed: {exc}"
            fallback_resolution = DeviceResolution(
                requested=resolution.requested,
                selected="cpu",
                backend="cpu",
                fallback_reason=fallback_reason,
                cuda_available=resolution.cuda_available,
                mps_available=resolution.mps_available,
                mps_built=resolution.mps_built,
            )
            try:
                return (
                    _instantiate_model(checkpoint_path, fallback_resolution),
                    fallback_resolution,
                    None,
                )
            except Exception as fallback_exc:
                logger.error("❌ CPU fallback failed: %s", fallback_exc)
                return None, fallback_resolution, str(fallback_exc)

        return None, resolution, str(exc)


def create_app(
    model_instance: OctoTetrahedralModel | None = None,
    device_resolution: DeviceResolution | None = None,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    requested_device: str | None = None,
    initialize: bool = True,
    model_error: str | None = None,
) -> FastAPI:
    app = FastAPI(title="OctoTetrahedral AGI Inference")

    if initialize and (model_instance is None or device_resolution is None):
        model_instance, device_resolution, runtime_error = initialize_runtime(
            checkpoint_path=checkpoint_path,
            requested_device=requested_device,
        )
        if model_error is None:
            model_error = runtime_error

    if device_resolution is None:
        device_resolution = DeviceResolution(
            requested=requested_device,
            selected="cpu",
            backend="cpu",
        )

    app.state.model = model_instance
    app.state.device_resolution = device_resolution
    app.state.model_error = model_error

    @app.post("/predict")
    async def predict(request: InferenceRequest):
        """Run inference on input tokens."""
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.model_error or "Model is unavailable.",
            )

        try:
            predictions = _run_inference(
                app.state.model,
                request.input_ids,
                app.state.device_resolution.selected,
            )
            logger.info("✅ Prediction successful")
            return {
                "predictions": predictions,
                "device": app.state.device_resolution.selected,
                "success": True,
            }
        except Exception as exc:
            if app.state.device_resolution.selected == "mps":
                logger.warning(
                    "⚠️ Metal inference failed, falling back to CPU for this and future requests: %s",
                    exc,
                )
                app.state.device_resolution = _fallback_to_cpu(
                    app.state.model,
                    app.state.device_resolution,
                    exc,
                )
                predictions = _run_inference(
                    app.state.model,
                    request.input_ids,
                    "cpu",
                )
                return {
                    "predictions": predictions,
                    "device": "cpu",
                    "success": True,
                }
            logger.error("❌ Inference error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/health")
    async def health():
        """Health check."""
        resolution = app.state.device_resolution
        return {
            "status": "healthy" if app.state.model is not None else "degraded",
            "model": "OctoTetrahedralModel",
            "device": resolution.selected,
            "device_backend": resolution.backend,
            "device_fallback_reason": resolution.fallback_reason,
            "mps_available": resolution.mps_available,
            "cuda_available": resolution.cuda_available,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
