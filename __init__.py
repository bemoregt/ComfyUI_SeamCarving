"""
ComfyUI Custom Node – Seam Carving Resize
==========================================
콘텐츠 인식 비선형 리사이징 노드입니다.
Seam Carving 알고리즘으로 중요 피사체를 보존하면서 이미지 크기를 조정합니다.

입력
----
image         : ComfyUI IMAGE 텐서 (B, H, W, C)
target_width  : 출력 폭  (픽셀)
target_height : 출력 높이 (픽셀)

출력
----
image         : 리사이징된 IMAGE 텐서 (B, target_height, target_width, C)

주의
----
* seam 수 = |Δwidth| + |Δheight| 이므로 큰 변화량에서는 처리 시간이 증가합니다.
* 배치 내 각 이미지는 동일한 목표 크기로 독립 처리됩니다.
"""

from __future__ import annotations

import numpy as np
import torch

from .seam_carving import seam_carve, total_steps

try:
    import comfy.utils as _comfy_utils
    _HAS_PBAR = hasattr(_comfy_utils, "ProgressBar")
except ImportError:
    _comfy_utils = None  # type: ignore[assignment]
    _HAS_PBAR = False


class SeamCarvingNode:
    """Seam Carving을 이용한 콘텐츠 인식 이미지 리사이징 노드."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "display": "number",
                        "tooltip": "출력 이미지의 폭 (픽셀)",
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "display": "number",
                        "tooltip": "출력 이미지의 높이 (픽셀)",
                    },
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"
    CATEGORY      = "image/transform"
    DESCRIPTION   = (
        "Seam Carving으로 중요 콘텐츠를 보존하면서 이미지를 비선형 리사이징합니다.\n"
        "일반 리사이징과 달리 배경·여백을 우선 제거/확장해 피사체 왜곡을 최소화합니다."
    )

    # ------------------------------------------------------------------ #

    def apply(
        self,
        image: torch.Tensor,
        target_width: int,
        target_height: int,
    ) -> tuple[torch.Tensor]:
        """
        Parameters
        ----------
        image         : (B, H, W, C) float32 tensor, 값 범위 [0, 1]
        target_width  : 목표 폭
        target_height : 목표 높이
        """
        B, H, W, C = image.shape

        # 프로그레스 바 초기화 (ComfyUI 환경에서만 동작)
        steps_per_image = total_steps(W, H, target_width, target_height)
        total           = B * steps_per_image

        pbar = None
        if _HAS_PBAR and total > 0:
            pbar = _comfy_utils.ProgressBar(total)

        def _step():
            if pbar is not None:
                pbar.update(1)

        results: list[np.ndarray] = []
        for i in range(B):
            img_np = image[i].cpu().numpy().astype(np.float32)
            result = seam_carve(img_np, target_width, target_height, _step)
            results.append(result)

        # (B, target_height, target_width, C)
        out = torch.from_numpy(np.stack(results, axis=0))
        return (out,)


# ------------------------------------------------------------------ #
# ComfyUI 노드 등록
# ------------------------------------------------------------------ #

NODE_CLASS_MAPPINGS = {
    "SeamCarving": SeamCarvingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamCarving": "Seam Carving Resize",
}
