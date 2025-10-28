from __future__ import annotations

import torch
from torch import Tensor


def _cov2d_to_conic_vjp(conic: Tensor, v_conic: Tensor) -> Tensor:
    a, b, c = conic.unbind(dim=-1)
    va, vb, vc = v_conic.unbind(dim=-1)
    xg00 = a * va + b * vb
    xg01 = a * vb + b * vc
    xg10 = b * va + c * vb
    xg11 = b * vb + c * vc
    v_sigma00 = -(xg00 * a + xg01 * b)
    v_sigma01 = -(xg00 * b + xg01 * c)
    v_sigma10 = -(xg10 * a + xg11 * b)
    v_sigma11 = -(xg10 * b + xg11 * c)
    return torch.stack(
        [v_sigma00, v_sigma01 + v_sigma10, v_sigma11], dim=-1
    )


def _rotmat(theta: Tensor) -> Tensor:
    cos_r = torch.cos(theta)
    sin_r = torch.sin(theta)
    row0 = torch.stack([cos_r, -sin_r], dim=-1)
    row1 = torch.stack([sin_r, cos_r], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _rotmat_grad(theta: Tensor) -> Tensor:
    cos_r = torch.cos(theta)
    sin_r = torch.sin(theta)
    row0 = torch.stack([-sin_r, -cos_r], dim=-1)
    row1 = torch.stack([cos_r, -sin_r], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def rasterize_backward_sum(
    xys: Tensor,
    conics: Tensor,
    colors: Tensor,
    radii: Tensor,
    grad_output: Tensor,
    topk_norm: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if topk_norm:
        raise NotImplementedError("Top-k normalization backward is not implemented in the simple version")
    device = xys.device
    dtype = xys.dtype
    num_gaussians = xys.shape[0]
    channels = colors.shape[1]
    _, img_h, img_w = grad_output.shape
    grad_output = grad_output.contiguous()

    v_xy = torch.zeros_like(xys)
    v_conic = torch.zeros_like(conics)
    v_color = torch.zeros_like(colors)

    # Process gaussians in chunks to keep temporary tensors bounded.
    chunk_size = 512 if num_gaussians > 512 else num_gaussians
    if chunk_size == 0:
        return v_xy, v_conic, v_color

    all_indices = torch.arange(num_gaussians, device=device, dtype=torch.long)

    for start in range(0, num_gaussians, chunk_size):
        end = min(start + chunk_size, num_gaussians)
        chunk_indices = all_indices[start:end]

        radius_chunk = radii[chunk_indices].to(dtype=torch.long)
        if radius_chunk.numel() == 0:
            continue

        active_mask = radius_chunk > 0
        if not torch.any(active_mask):
            continue

        chunk_indices = chunk_indices[active_mask]
        radius_chunk = radius_chunk[active_mask]

        cx = xys[chunk_indices, 0]
        cy = xys[chunk_indices, 1]
        conic_chunk = conics[chunk_indices]
        color_chunk = colors[chunk_indices]

        radius_f = radius_chunk.to(dtype=dtype)

        x_min = torch.floor(cx - radius_f - 1.0).to(torch.long)
        x_max = torch.floor(cx + radius_f + 1.0).to(torch.long)
        y_min = torch.floor(cy - radius_f - 1.0).to(torch.long)
        y_max = torch.floor(cy + radius_f + 1.0).to(torch.long)

        x_min.clamp_(0, img_w - 1)
        x_max.clamp_(0, img_w - 1)
        y_min.clamp_(0, img_h - 1)
        y_max.clamp_(0, img_h - 1)

        valid_bbox = (x_max >= x_min) & (y_max >= y_min)
        if not torch.any(valid_bbox):
            continue

        chunk_indices = chunk_indices[valid_bbox]
        radius_chunk = radius_chunk[valid_bbox]
        cx = cx[valid_bbox]
        cy = cy[valid_bbox]
        conic_chunk = conic_chunk[valid_bbox]
        color_chunk = color_chunk[valid_bbox]
        x_min = x_min[valid_bbox]
        x_max = x_max[valid_bbox]
        y_min = y_min[valid_bbox]
        y_max = y_max[valid_bbox]

        if chunk_indices.numel() == 0:
            continue

        patch_w = (x_max - x_min + 1)
        patch_h = (y_max - y_min + 1)

        max_w = int(patch_w.max().item())
        max_h = int(patch_h.max().item())
        if max_w <= 0 or max_h <= 0:
            continue

        x_offsets = torch.arange(max_w, device=device, dtype=torch.long)
        y_offsets = torch.arange(max_h, device=device, dtype=torch.long)
        grid_y_off_long, grid_x_off_long = torch.meshgrid(y_offsets, x_offsets, indexing="ij")

        x_coords_idx = x_min.view(-1, 1, 1) + grid_x_off_long
        y_coords_idx = y_min.view(-1, 1, 1) + grid_y_off_long

        # Clamp coordinates so advanced indexing stays within bounds. Contributions on
        # clamped pixels are masked out later so this does not affect correctness.
        x_coords_idx = x_coords_idx.clamp(0, img_w - 1)
        y_coords_idx = y_coords_idx.clamp(0, img_h - 1)

        valid_patch = (grid_x_off_long < patch_w.view(-1, 1, 1)) & (
            grid_y_off_long < patch_h.view(-1, 1, 1)
        )

        # Build floating coordinate grids for distance computations.
        grid_x_off = grid_x_off_long.to(dtype)
        grid_y_off = grid_y_off_long.to(dtype)
        x_coords = x_min.to(dtype).view(-1, 1, 1) + grid_x_off
        y_coords = y_min.to(dtype).view(-1, 1, 1) + grid_y_off

        dx = cx.view(-1, 1, 1) - x_coords
        dy = cy.view(-1, 1, 1) - y_coords

        a = conic_chunk[:, 0].view(-1, 1, 1)
        b = conic_chunk[:, 1].view(-1, 1, 1)
        c = conic_chunk[:, 2].view(-1, 1, 1)
        sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy

        valid_sigma = torch.isfinite(sigma) & (sigma >= 0.0)
        valid_total = valid_patch & valid_sigma

        if not torch.any(valid_total):
            continue

        sigma = torch.where(
            valid_total,
            sigma,
            torch.zeros(1, device=device, dtype=dtype),
        )
        weight = torch.exp(-sigma) * valid_total.to(dtype)

        patch_grad = grad_output[:, y_coords_idx, x_coords_idx]
        patch_grad = patch_grad.permute(1, 0, 2, 3).contiguous()
        patch_grad = patch_grad.to(dtype)
        patch_grad = patch_grad * valid_total.unsqueeze(1).to(dtype)

        weighted_grad = (patch_grad * weight.unsqueeze(1)).sum(dim=(2, 3))
        v_color.index_add_(0, chunk_indices, weighted_grad)

        patch_dot = (patch_grad * color_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        v_sigma = -weight * patch_dot

        dx2 = dx * dx
        dy2 = dy * dy
        dxdy = dx * dy

        v_conic_chunk = torch.stack(
            [
                0.5 * (v_sigma * dx2).sum(dim=(1, 2)),
                (v_sigma * dxdy).sum(dim=(1, 2)),
                0.5 * (v_sigma * dy2).sum(dim=(1, 2)),
            ],
            dim=-1,
        )
        v_conic.index_add_(0, chunk_indices, v_conic_chunk)

        vx = (v_sigma * (a * dx + b * dy)).sum(dim=(1, 2))
        vy = (v_sigma * (b * dx + c * dy)).sum(dim=(1, 2))
        v_xy_chunk = torch.stack([vx, vy], dim=-1)
        v_xy.index_add_(0, chunk_indices, v_xy_chunk)

    return v_xy, v_conic, v_color


def project_backward_scale_rot(
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    grad_xy: Tensor,
    grad_conic: Tensor,
    img_height: int,
    img_width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    theta = rotation.view(-1)
    v_cov2d = _cov2d_to_conic_vjp(conic=grad_conic, v_conic=grad_conic)

    v_mean = torch.zeros_like(means2d)
    v_mean[:, 0] = grad_xy[:, 0] * img_width
    v_mean[:, 1] = grad_xy[:, 1] * img_height

    R = _rotmat(theta).to(dtype=scales2d.dtype)
    Rg = _rotmat_grad(theta).to(dtype=scales2d.dtype)

    S = torch.zeros(theta.shape[0], 2, 2, device=scales2d.device, dtype=scales2d.dtype)
    S[:, 0, 0] = scales2d[:, 0]
    S[:, 1, 1] = scales2d[:, 1]

    M = torch.matmul(R, S)
    Mt = M.transpose(-1, -2)

    scale_x_g = torch.zeros_like(S)
    scale_y_g = torch.zeros_like(S)
    scale_x_g[:, 0, 0] = 2.0 * scales2d[:, 0]
    scale_y_g[:, 1, 1] = 2.0 * scales2d[:, 1]

    sigma_x_g = torch.matmul(torch.matmul(R, scale_x_g), R.transpose(-1, -2))
    sigma_y_g = torch.matmul(torch.matmul(R, scale_y_g), R.transpose(-1, -2))

    theta_g = torch.matmul(torch.matmul(Rg, S), Mt) + torch.matmul(M, torch.matmul(S, Rg.transpose(-1, -2)))

    G11 = v_cov2d[:, 0]
    G12 = v_cov2d[:, 1]
    G22 = v_cov2d[:, 2]

    v_scale = torch.zeros_like(scales2d)
    v_scale[:, 0] = (
        G11 * sigma_x_g[:, 0, 0]
        + 2.0 * G12 * sigma_x_g[:, 0, 1]
        + G22 * sigma_x_g[:, 1, 1]
    )
    v_scale[:, 1] = (
        G11 * sigma_y_g[:, 0, 0]
        + 2.0 * G12 * sigma_y_g[:, 0, 1]
        + G22 * sigma_y_g[:, 1, 1]
    )

    v_rot = (
        G11 * theta_g[:, 0, 0]
        + 2.0 * G12 * theta_g[:, 0, 1]
        + G22 * theta_g[:, 1, 1]
    ).unsqueeze(-1)

    return v_mean, v_scale, v_rot
