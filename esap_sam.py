#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESAP-SAM binary semantic segmentation; see README.md for paper/implementation boundaries.

SAM components below are adapted from Meta Segment Anything (Apache-2.0):
https://github.com/facebookresearch/segment-anything
Revision: dca509fe793f601edb92606367a655c15ac00fdf
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
The full upstream license is reproduced in README.md.
Modifications: merged modeling modules; disambiguated attention class names;
removed relative imports/NumPy dependency; added ESAP-SAM adaptation and CLI.
"""
from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F



# ===== Upstream SAM: modeling/common.py =====
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.




class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ===== Upstream SAM: modeling/image_encoder.py =====
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.





# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class ViTAttention(nn.Module):
    """Multi-head ViTAttention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


# ===== Upstream SAM: modeling/prompt_encoder.py =====
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.





class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# ===== Upstream SAM: modeling/transformer.py =====
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.





class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = DecoderAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class DecoderAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # DecoderAttention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


# ===== Upstream SAM: modeling/mask_decoder.py =====
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.





class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# ===== ESAP-SAM: paper equations (1)--(9), with documented implementation choices =====

def _build_sam_components(backbone: str) -> nn.Module:
    """Original SAM parameter names/shapes, so official .pth files load strictly."""
    configs = {
        "vit_l": (1024, 24, 16, (5, 11, 17, 23)),
        "vit_b": (768, 12, 12, (2, 5, 8, 11)),
    }
    if backbone not in configs:
        raise ValueError(f"Unsupported backbone: {backbone}; choose vit_l or vit_b")
    width, depth, heads, global_blocks = configs[backbone]
    bank = nn.Module()
    bank.image_encoder = ImageEncoderViT(
        img_size=1024, patch_size=16, embed_dim=width, depth=depth,
        num_heads=heads, mlp_ratio=4, out_chans=256, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_rel_pos=True,
        global_attn_indexes=global_blocks, window_size=14,
    )
    bank.prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(1024, 1024), mask_in_chans=16,
    )
    bank.mask_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256,
    )
    return bank


class DirectionalConv(nn.Module):
    """A learnable k-tap line kernel with actual directional support, not a dense k*k kernel."""
    def __init__(self, channels: int, kernel_size: int, direction: str):
        super().__init__()
        self.kernel_size = kernel_size
        self.direction = direction
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if direction in {"diag45", "diag135"}:
            basis = torch.zeros(kernel_size, kernel_size, kernel_size)
            for i in range(kernel_size):
                row = kernel_size - 1 - i if direction == "diag45" else i
                basis[i, row, i] = 1
            self.register_buffer("basis", basis)

    def forward(self, x: Tensor) -> Tensor:
        p = self.kernel_size // 2
        if self.direction == "horizontal":
            return F.conv2d(x, self.weight.unsqueeze(2), padding=(0, p))
        if self.direction == "vertical":
            return F.conv2d(x, self.weight.unsqueeze(3), padding=(p, 0))
        kernel = torch.einsum("ock,khw->ochw", self.weight, self.basis)
        return F.conv2d(x, kernel, padding=p)


class EMSAdapter(nn.Module):
    """Paper §2.2: channel reduction, MSCM + four-direction MDCM, aggregation, restoration."""
    def __init__(self, encoder_channels: int, channels: int = 32, kernel_size: int = 7):
        super().__init__()
        if channels < 1 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("adapter_channels must be positive; directional_kernel must be odd >= 3")
        self.down = nn.Conv2d(encoder_channels, channels, 1)
        self.scale1 = nn.Conv2d(channels, channels, 1)
        self.scale3 = nn.Conv2d(channels, channels, 3, padding=1)
        # Paper gives GAP -> MLP -> sigmoid, but does not specify MLP width.
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(channels // 4, 1), 1),
            nn.ReLU(), nn.Conv2d(max(channels // 4, 1), channels, 1), nn.Sigmoid(),
        )
        self.directions = nn.ModuleList([
            nn.Sequential(DirectionalConv(channels, kernel_size, direction),
                          nn.BatchNorm2d(channels), nn.ReLU())
            for direction in ("horizontal", "vertical", "diag45", "diag135")
        ])
        self.aggregate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                       nn.BatchNorm2d(channels), nn.ReLU())
        self.up = nn.Conv2d(channels, encoder_channels, 1)

    def forward(self, patch_features: Tensor) -> Tensor:
        low = self.down(patch_features)
        multiscale = self.scale1(low) + self.scale3(low)
        multiscale = multiscale * self.channel_attention(multiscale)
        directional = sum(branch(low) for branch in self.directions)
        return self.up(self.aggregate(multiscale + directional))


class HybridPromptEncoder(nn.Module):
    """Paper §2.3 / Fig.5: dense C + aligned sparse 2 -> FFN -> dense C.

    Implementation choice: align frozen sparse tokens via positive/negative cosine
    similarity maps. Missing sparse prompts give two zero maps. Original geometric
    tokens also enter the decoder's token stream. See README for details.
    """
    def __init__(self, channels: int = 256, hidden: int = 512):
        super().__init__()
        if hidden < 1:
            raise ValueError("hp_hidden must be positive")
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm = LayerNorm2d(channels)
        self.fusion = nn.Sequential(nn.Linear(channels + 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, channels))

    def forward(self, image: Tensor, pe: Tensor, sparse: Tensor,
                polarity: Tensor, mask_embedding: Optional[Tensor]) -> Tensor:
        dense = self.norm(self.project(image)) + pe
        if mask_embedding is not None:
            dense = dense + mask_embedding
        b, _, h, w = dense.shape
        aligned = dense.new_zeros(b, 2, h, w)
        if sparse.shape[1]:
            query = F.normalize(dense.flatten(2).transpose(1, 2), dim=-1)
            keys = F.normalize(sparse.to(dense.dtype), dim=-1)
            similarities = query @ keys.transpose(1, 2)
            for index, sign in enumerate((1, 0)):
                valid = (polarity == sign).to(similarities.dtype)
                scores = (similarities * valid[:, None, :]).sum(-1)
                scores = scores / valid.sum(-1).clamp_min(1)[:, None]
                aligned[:, index] = scores.reshape(b, h, w)
        fused = torch.cat((dense, aligned), dim=1).permute(0, 2, 3, 1)
        return self.fusion(fused).permute(0, 3, 1, 2).contiguous()


class FeatureMaskDecoder(nn.Module):
    """SAM's two-way attention blocks and 4x upscaler, returning features for BRA.

    Native mask hypernetworks / quality head / final token-only attention are not
    retained: paper BRA supplies the prediction head. Never repeat a whole image
    batch B times as the native single-image/multiple-prompts SAM wrapper does.
    """
    def __init__(self, sam_decoder: MaskDecoder):
        super().__init__()
        self.layers = sam_decoder.transformer.layers
        self.iou_token = sam_decoder.iou_token
        self.mask_tokens = sam_decoder.mask_tokens
        self.output_upscaling = sam_decoder.output_upscaling

    def forward(self, image: Tensor, pe: Tensor, sparse: Tensor, dense: Tensor) -> Tensor:
        b, c, h, w = image.shape
        tokens = torch.cat((self.iou_token.weight, self.mask_tokens.weight), dim=0)
        tokens = torch.cat((tokens.unsqueeze(0).expand(b, -1, -1), sparse), dim=1)
        keys = (image + dense).flatten(2).transpose(1, 2)
        key_pe = pe.expand(b, -1, -1, -1).flatten(2).transpose(1, 2)
        queries = tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, tokens, key_pe)
        decoded = keys.transpose(1, 2).reshape(b, c, h, w)
        return self.output_upscaling(decoded)


class BoundaryRefinementAttention(nn.Module):
    """Paper §2.4, equations (6)--(9): additive edge attention, not multiplication."""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.preliminary = nn.Conv2d(channels, 1, 1)
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("sobel", torch.stack((sobel, sobel.T)).unsqueeze(1))
        self.edge_attention = nn.Conv2d(1, 1, 3, padding=1)
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, 1, 1))

    def forward(self, decoded: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        preliminary = self.preliminary(decoded)
        # FP32 Sobel + epsilon under sqrt: finite gradients even on flat masks.
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            gradient = F.conv2d(F.pad(preliminary.float(), (1, 1, 1, 1), mode="replicate"),
                                self.sobel.float())
            magnitude = (gradient.square().sum(1, keepdim=True) + 1e-6).sqrt()
        attention = self.edge_attention(magnitude.to(decoded.dtype)).sigmoid()
        enhanced = self.fusion(decoded + attention)
        return self.head(enhanced), preliminary, attention


class ESAPSAM(nn.Module):
    """Full ESAP-SAM, RGB image [B,3,H,W] in [0,1] -> raw logits [B,1,H,W].

    points=(xy [B,N,2], labels [B,N]), labels in {-1,0,1}; boxes [B,4]
    use (x0,y0,x1,y1) coordinates in INPUT pixels. mask_prompt [B,1,H,W]
    contains probabilities in [0,1]. All prompts are optional.
    """
    def __init__(self, backbone: str = "vit_l", sam_checkpoint: Optional[str] = None,
                 adapter_channels: int = 32, directional_kernel: int = 7,
                 hp_hidden: int = 512, allow_random_init: bool = False):
        super().__init__()
        if sam_checkpoint is None and not allow_random_init:
            raise ValueError("sam_checkpoint is required; allow_random_init=True is for smoke tests only")
        self.config = dict(backbone=backbone, adapter_channels=adapter_channels,
                           directional_kernel=directional_kernel, hp_hidden=hp_hidden)
        bank = _build_sam_components(backbone)
        if sam_checkpoint is not None:
            state = torch.load(Path(sam_checkpoint), map_location="cpu", weights_only=True)
            try:
                bank.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ValueError(f"Not a matching official SAM {backbone} checkpoint: {exc}") from exc
        self.image_encoder = bank.image_encoder
        self.prompt_encoder = bank.prompt_encoder
        self.image_encoder.requires_grad_(False)
        self.prompt_encoder.requires_grad_(False)
        width = 1024 if backbone == "vit_l" else 768
        self.ems = EMSAdapter(width, adapter_channels, directional_kernel)
        self.hp = HybridPromptEncoder(256, hp_hidden)
        self.decoder = FeatureMaskDecoder(bank.mask_decoder)
        self.bra = BoundaryRefinementAttention(32)
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        return self

    @torch.no_grad()
    def _geometry(self, batch: int, original_hw: Tuple[int, int], padded_hw: Tuple[int, int],
                  grid_hw: Tuple[int, int], device: torch.device,
                  points, boxes, mask_prompt):
        encoder = self.prompt_encoder
        h, w = original_hw
        sparse = torch.empty(batch, 0, 256, device=device)
        polarity = torch.empty(batch, 0, dtype=torch.long, device=device)
        if points is not None:
            xy, labels = (t.to(device) for t in points)
            if xy.ndim != 3 or xy.shape[0] != batch or xy.shape[2] != 2 or labels.shape != xy.shape[:2]:
                raise ValueError("points require xy [B,N,2] and labels [B,N]")
            if not torch.isin(labels, labels.new_tensor([-1, 0, 1])).all():
                raise ValueError("point labels must be -1 (padding), 0 (background), or 1 (foreground)")
            valid = labels != -1
            outside = (~torch.isfinite(xy).all(-1) | (xy[..., 0] < 0) | (xy[..., 0] >= w)
                       | (xy[..., 1] < 0) | (xy[..., 1] >= h))
            if (outside & valid).any():
                raise ValueError("Point coordinates are outside the input image")
            xy = torch.where(valid[..., None], xy.float(), torch.zeros_like(xy).float())
            embedding = encoder.pe_layer.forward_with_coords(xy + 0.5, padded_hw)
            embedding[~valid] = 0
            embedding[~valid] += encoder.not_a_point_embed.weight
            embedding[labels == 0] += encoder.point_embeddings[0].weight
            embedding[labels == 1] += encoder.point_embeddings[1].weight
            sparse = torch.cat((sparse, embedding), dim=1)
            polarity = torch.cat((polarity, labels.long()), dim=1)
        if boxes is not None:
            boxes = boxes.to(device=device, dtype=torch.float32)
            if boxes.shape != (batch, 4) or not torch.isfinite(boxes).all():
                raise ValueError("boxes must be finite [B,4]")
            if ((boxes[:, :2] < 0).any() or (boxes[:, 2] > w).any() or
                    (boxes[:, 3] > h).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
                raise ValueError("boxes must satisfy 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H")
            embedding = encoder.pe_layer.forward_with_coords(boxes.reshape(batch, 2, 2) + 0.5, padded_hw)
            embedding[:, 0] += encoder.point_embeddings[2].weight
            embedding[:, 1] += encoder.point_embeddings[3].weight
            sparse = torch.cat((sparse, embedding), dim=1)
            polarity = torch.cat((polarity, torch.ones(batch, 2, device=device, dtype=torch.long)), dim=1)
        mask_embedding = None
        if mask_prompt is not None:
            mask = mask_prompt.to(device=device, dtype=torch.float32)
            if (mask.shape != (batch, 1, h, w) or not torch.isfinite(mask).all()
                    or (mask < 0).any() or (mask > 1).any()):
                raise ValueError("mask_prompt must be [B,1,H,W] probabilities in [0,1]")
            mask = F.pad(mask, (0, padded_hw[1] - w, 0, padded_hw[0] - h))
            mask = F.interpolate(mask, size=(grid_hw[0] * 4, grid_hw[1] * 4),
                                 mode="bilinear", align_corners=False)
            mask_embedding = encoder.mask_downscaling(mask)
        pe = encoder.pe_layer(grid_hw).unsqueeze(0)
        return sparse, polarity, mask_embedding, pe

    def forward(self, images: Tensor, points=None, boxes: Optional[Tensor] = None,
                mask_prompt: Optional[Tensor] = None, return_aux: bool = False):
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("images must be floating RGB [B,3,H,W] in [0,1]")
        batch, _, h, w = images.shape
        if batch < 1 or min(h, w) < 16:
            raise ValueError("Nonempty batch with H,W >= 16 required")
        padded_hw = ((h + 15) // 16 * 16, (w + 15) // 16 * 16)
        x = (images * 255.0 - self.pixel_mean) / self.pixel_std
        x = F.pad(x, (0, padded_hw[1] - w, 0, padded_hw[0] - h))
        # Adapter is parallel to frozen ViT: no adapter gradients pass through ViT.
        with torch.no_grad():
            patches = self.image_encoder.patch_embed(x)
            grid_hw = patches.shape[1:3]
            vit = patches
            if self.image_encoder.pos_embed is not None:
                position = self.image_encoder.pos_embed.permute(0, 3, 1, 2)
                position = F.interpolate(position, size=grid_hw, mode="bicubic", align_corners=False)
                vit = vit + position.permute(0, 2, 3, 1)
            for block in self.image_encoder.blocks:
                vit = block(vit)
        fused = vit.permute(0, 3, 1, 2) + self.ems(patches.permute(0, 3, 1, 2))
        # Frozen neck still needs autograd w.r.t. its INPUT so EMS receives gradients.
        image_embedding = self.image_encoder.neck(fused)
        sparse, polarity, mask_embedding, pe = self._geometry(
            batch, (h, w), padded_hw, grid_hw, images.device, points, boxes, mask_prompt)
        dense = self.hp(image_embedding, pe, sparse, polarity, mask_embedding)
        decoded = self.decoder(image_embedding, pe, sparse, dense)
        logits, preliminary, edge = self.bra(decoded)
        def restore(tensor):
            tensor = F.interpolate(tensor, size=padded_hw, mode="bilinear", align_corners=False)
            return tensor[..., :h, :w]
        logits = restore(logits)
        if return_aux:
            return {"logits": logits, "preliminary_logits": restore(preliminary),
                    "edge_attention": restore(edge)}
        return logits

    @torch.inference_mode()
    def predict(self, images: Tensor, **prompts) -> Tensor:
        """Return probabilities. Call model.eval() before inference."""
        return self(images, **prompts).sigmoid()


def _model_cli():
    parser = argparse.ArgumentParser(description="ESAP-SAM shape/gradient smoke test")
    parser.add_argument("--backbone", choices=("vit_l", "vit_b"), default="vit_l")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = ESAPSAM(args.backbone, args.sam_checkpoint,
                    allow_random_init=args.allow_random_init).to(args.device)
    model.train(args.backward)
    x = torch.rand(args.batch_size, 3, args.size, args.size, device=args.device)
    with torch.set_grad_enabled(args.backward):
        result = model(x)
        assert result.shape == (args.batch_size, 1, args.size, args.size)
        assert torch.isfinite(result).all()
        if args.backward:
            result.square().mean().backward()
            for name, module in (("EMS", model.ems), ("HP", model.hp),
                                 ("decoder", model.decoder), ("BRA", model.bra)):
                gradients = [p.grad for p in module.parameters() if p.requires_grad]
                assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients), name
            assert all(p.grad is None for p in model.image_encoder.parameters())
            assert all(p.grad is None for p in model.prompt_encoder.parameters())
    print("output:", tuple(result.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Smoke test passed; random initialization is NOT a pretrained experiment.")


if __name__ == "__main__":
    _model_cli()
