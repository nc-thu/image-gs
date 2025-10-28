"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
import torch

import gsplat.cuda as _C

def project_gaussians_2d_scale_rot(
    means2d: Float[Tensor, "*batch 2"],
    scales2d: Float[Tensor, "*batch 2"],
    rotation: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int]
) -> Tuple[Tensor, Tensor, Tensor, int]:

    return _ProjectGaussians2dScaleRot.apply(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        img_height,
        img_width,
        tile_bounds
    )

class _ProjectGaussians2dScaleRot(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means2d: Float[Tensor, "*batch 2"],
        scales2d: Float[Tensor, "*batch 2"],
        rotation: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int]
    ):
        num_points = means2d.shape[-2]
        if num_points < 1 or means2d.shape[-1] != 2:
            raise ValueError(f"Invalid shape for means2d: {means2d.shape}")
        (
            xys,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_scale_rot_forward(
            num_points,
            means2d,
            scales2d,
            rotation,
            img_height,
            img_width,
            tile_bounds
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        )
        return (xys, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_radii, v_conics, v_num_tiles_hit):
        print("\n" + "="*70)
        print("🔥🔥🔥 [Project CUDA Backward] 被调用了！")
        print(f"  v_xys shape: {v_xys.shape if v_xys is not None else None}")
        print(f"  v_conics shape: {v_conics.shape if v_conics is not None else None}")
        print("="*70)
        
        (
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        ) = ctx.saved_tensors
        print("  📌 调用 _C.project_gaussians_2d_scale_rot_backward (CUDA kernel)...")
        (v_cov2d, v_mean2d, v_scale, v_rot) = _C.project_gaussians_2d_scale_rot_backward(
            ctx.num_points,
            means2d,
            scales2d,
            rotation,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_conics,
        )
        print(f"  ✅ Project backward完成！")
        print(f"     v_mean2d shape: {v_mean2d.shape if v_mean2d is not None else None}")
        print(f"     v_scale shape: {v_scale.shape if v_scale is not None else None}")
        print(f"     v_rot shape: {v_rot.shape if v_rot is not None else None}")
        print("="*70 + "\n")
        

        # Return a gradient for each input.
        return (
            # means2d: Float[Tensor, "*batch 2"],
            v_mean2d,
            # scales: Float[Tensor, "*batch 2"],
            v_scale,
            #rotation: Float,
            v_rot,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
        )
