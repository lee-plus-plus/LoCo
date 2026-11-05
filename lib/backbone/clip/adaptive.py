"""To make original CLIP model adaptive to input with variant image size"""

import torch
import torch.nn.functional as F
from .model import AttentionPool2d, VisionTransformer
from .clip import load
from typing import Union, List


def adaptive_pos_embed(positional_embedding, H, W, h, w):
    """interpolate positional_embedding: [1 + HW, C] to [1 + hw, C]"""
    HW1, C = positional_embedding.shape
    assert H * W + 1 == HW1

    if (H, W) == (h, w):
        return positional_embedding

    # update the position embedding during inference for varied input size
    dtype = positional_embedding.dtype
    positional_embedding = positional_embedding.float()
    cls_embed, pos_embed = positional_embedding[:1], positional_embedding[1:]

    pos_embed = pos_embed.reshape(1, H, W, -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, (h, w), mode="bilinear")
    pos_embed = pos_embed.reshape(1, -1, h * w).transpose(1, 2)
    positional_embedding = torch.cat([cls_embed, pos_embed[0]], 0)
    return positional_embedding.type(dtype)


class AdaptiveAttentionPool2d(AttentionPool2d):
    """To support variant input image size"""

    def forward(self, x, use_pos_embed=True):
        H = W = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        assert H * W == self.positional_embedding.shape[0] - 1
        n, c, h, w = x.shape

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        if use_pos_embed:
            positional_embedding = adaptive_pos_embed(self.positional_embedding, H=H, W=W, h=h, w=w)
            x = x + positional_embedding.to(x.dtype)[:, None, :]  # (HW+1)NC

        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class AdaptiveVisionTransformer(VisionTransformer):
    """To support variant input image size"""

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        H = W = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        assert H * W == self.positional_embedding.shape[0] - 1

        n, c, h, w = x.shape

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        # x = x + self.positional_embedding.to(x.dtype)
        positional_embedding = adaptive_pos_embed(self.positional_embedding, H=H, W=W, h=h, w=w)
        x = x + positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def create_clip(name: str):
    clip_model, preprocess = load(name, device="cpu", jit=False, download_root=None)

    if name.startswith("RN"):
        clip_model.visual.attnpool.__class__ = AdaptiveAttentionPool2d
    elif name.startswith("ViT"):
        clip_model.visual.__class__ = AdaptiveVisionTransformer

    return clip_model.float(), preprocess
