"""SC-CLIP

Bai, Sule, et al.
"Self-calibrated clip for training-free open-vocabulary segmentation."
IEEE Transactions on Image Processing (2025).
"""
from typing import Annotated, Literal

from pydantic import conint
from copy import deepcopy
from torch import nn, Tensor
import math
import torch
import torch.nn.functional as F
from lib.utils.entry import parse_from_cli
from ...backbone.clip.model import CLIP, VisionTransformer
from ...backbone import TextEncoder, tokenize, create_clip
from .templates import IMAGENET_TEMPLATES, BACKGROUND_CATEGORY
from ...utils import get_model_device, SlideWindowCropper
from ..base import BaseConfig, BaseTrainer, entry


def slide_window_patching(
    image_size: tuple[int, int], stride: tuple[int, int], crop_size: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    H0, W0 = image_size

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size

    h_grids = max(H0 - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W0 - w_crop + w_stride - 1, 0) // w_stride + 1

    crop_coords = [
        (
            y1 := h_idx * h_stride,
            x1 := w_idx * w_stride,
            y2 := min(y1 + h_crop, H0),
            x2 := min(x1 + w_crop, W0),
        )
        for h_idx in range(h_grids)
        for w_idx in range(w_grids)
    ]

    return crop_coords


def merge_slide_window_logits(crop_logits, crop_coords):
    # crop_coords: list of (y1, x1, y2, x2)
    # crop_logits: [K, B, H, W, C]
    # return: logits: [B, HW, C]
    
    _, _, H_IMAGE, W_IMAGE = [max(elem) for elem in zip(*crop_coords)]
    H_CROP, W_CROP = crop_coords[0][2] - crop_coords[0][0], crop_coords[0][3] - crop_coords[0][1]
    K, B, H, W, C = crop_logits.shape
    H2, W2 = H_IMAGE * H // H_CROP, W_IMAGE * W // W_CROP   

    logits = torch.zeros((B, H2, W2, C)).to(crop_logits.device)
    counts = torch.zeros((B, H2, W2, 1)).to(crop_logits.device)

    for (y1, x1, y2, x2), crop_logit in zip(crop_coords, crop_logits):
        y1 = y1 * H // H_CROP
        x1 = x1 * W // W_CROP
        y2 = y2 * H // H_CROP
        x2 = x2 * W // W_CROP

        logits[:, y1:y2, x1:x2, :] += crop_logit.view(B, H, W, C)
        counts[:, y1:y2, x1:x2, :] += 1
    assert (counts == 0).sum() == 0

    logits = logits / counts
    return logits  # [B, H2, W2, C]


@torch.no_grad()
def local_outliner_factor(x, n_neighbors=30, contamination=0.05):
    """Local Outlier Factor in pure PyTorch to flag outlier samples."""
    if x.dim() == 3:
        N, B, D = x.shape
        return [local_outliner_factor(x[:, b, :], n_neighbors, contamination) for b in range(B)]

    assert x.dim() == 2
    # x: [N, D]

    # Pairwise squared L2 distance matrix
    distances = torch.norm(x[:, None] - x[None, :], dim=2, p=2) ** 2

    # k-NN distances/indices (drop self-neighbor)
    knn_distances, knn_indices = torch.topk(distances, k=n_neighbors + 1, largest=False)
    knn_distances, knn_indices = knn_distances[:, 1:], knn_indices[:, 1:]

    # Reachability distance: max of neighbor distance and k-distance
    k_distances = knn_distances[:, -1].unsqueeze(1).expand_as(knn_distances)
    reach_distances = torch.max(knn_distances, k_distances)

    # Local reachability density (lrd)
    lrd = n_neighbors / torch.nan_to_num(reach_distances.mean(dim=1), nan=1e-6)

    # lof score = mean lrd ratio vs. neighbors
    lrd_ratios = lrd[knn_indices] / lrd.unsqueeze(1)
    lof_scores = lrd_ratios.mean(dim=1)

    # Choose threshold by contamination percentile
    threshold = torch.quantile(lof_scores.to(torch.float32), 1 - contamination)

    outlier_mask = lof_scores > threshold
    outlier_indices = torch.where(outlier_mask)[0]  # [num_outliers]

    # return outlier_indices, lof_scores
    return outlier_indices


@torch.no_grad()
def mean_interpolation(x, top_indices, H=None, W=None):
    if x.dim() == 3:
        N, B, C = x.shape
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return mean_interpolation(x, top_indices, H=H, W=W).reshape(B, C, H * W).permute(2, 0, 1)

    assert x.dim() == 4, "x should be 4-D tensor"
    B, C, H, W = x.shape

    if B != 1:
        return torch.cat([mean_interpolation(x[b : b + 1, :, :, :], top_indices[b], H, W) for b in range(B)], dim=0)

    if len(top_indices) == 0:
        return x

    device = x.device
    dtype = x.dtype

    kernel = torch.ones(C, 1, 3, 3, device=device, dtype=dtype)
    kernel[:, 0, 1, 1] = 0
    mask = torch.ones((H, W), device=device, dtype=dtype)
    indices = torch.tensor(top_indices, dtype=torch.long, device=device)
    mask[indices[:, 0], indices[:, 1]] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)
    masked_x = x * mask
    padded_x = F.pad(masked_x, (1, 1, 1, 1), mode="constant", value=0)
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
    neighbor_sum = F.conv2d(padded_x, kernel, groups=C)
    valid_neighbors = F.conv2d(padded_mask, kernel[:, :1, :, :], groups=1)
    valid_neighbor_mask = (valid_neighbors > 0).to(dtype)
    safe_valid_neighbors = valid_neighbors.clone()
    safe_valid_neighbors[safe_valid_neighbors == 0] = 1
    mean_neighbors = neighbor_sum / safe_valid_neighbors
    top_indices_mask = torch.zeros((H, W), device=device, dtype=dtype)
    top_indices_mask[indices[:, 0], indices[:, 1]] = 1
    top_indices_mask = top_indices_mask.unsqueeze(0).unsqueeze(0)
    update_mask = top_indices_mask * valid_neighbor_mask
    x = x * (1 - update_mask) + mean_neighbors * update_mask

    return x


class ScClipVisionTransformer(VisionTransformer):

    def post_init(
        self,
        anomaly_resolve: bool = True,
        multi_layer_fuse: bool = True,
        attn_enhance: bool = True,
        feat_aggregate: bool = True,
        cls_token_require: bool = False,
    ):
        self.patch_size = self.conv1.kernel_size[0]
        self.similarity_threshold = 0.4
        self.mid_layer1_idx = 8
        self.mid_layer2_idx = 3
        self.multi_start_idx = 3
        self.multi_end_idx = 10
        self.cls_token_fusing_factor = 0.3

        self.anomaly_resolve = anomaly_resolve
        self.multi_layer_fuse = multi_layer_fuse
        self.attn_enhance = attn_enhance
        self.feat_aggregate = feat_aggregate
        self.cls_token_require = cls_token_require

    def forward(self, x: Tensor, return_all: bool = True) -> Tensor:
        x = self.conv1(x)
        H, W = x.shape[-2], x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, H, W).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.custom_transformer(x, H, W)

        x = x.permute(1, 0, 2)
        if return_all:
            x = self.ln_post(x)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def _similarity_and_mask(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        sim = torch.einsum("nbd, mbd -> bnm", feats, feats)
        masked_sim = torch.where(sim < self.similarity_threshold, 0, sim)
        return sim, masked_sim

    def _forward_last_block(self, x: Tensor, cls_token: Tensor, sim: Tensor) -> Tensor:
        blk = self.transformer.resblocks[-1]
        return self.custom_attn(blk.attn, blk.ln_1(x), sim=sim) + self.cls_token_fusing_factor * cls_token

    def custom_transformer(self, x: Tensor, H: int, W: int) -> Tensor:
        N, B, D = x.shape
        assert N == H * W + 1

        layer_feats = []
        for idx, blk in enumerate(self.transformer.resblocks[:-1], start=1):
            x = blk(x)
            layer_feats.append(x)

            if idx == len(self.transformer.resblocks) - 1:
                x0 = x.clone()

                if self.anomaly_resolve:
                    outlier_indices = [
                        [(i // W, i % W) for i in elem]
                        for elem in local_outliner_factor(x[1:, ...], n_neighbors=30, contamination=0.05)
                    ]
                    x[1:, ...] = mean_interpolation(x[1:, ...], outlier_indices, H=H, W=W)

        if self.multi_layer_fuse:
            multi_layer_raw_feats = torch.stack(
                layer_feats[self.multi_start_idx : self.multi_end_idx], dim=0
            ).mean(dim=0)

        x1_cls_token, x1 = x[:1, ...], x[1:, ...]
        if self.multi_layer_fuse:
            x2_cls_token, x2 = multi_layer_raw_feats[:1, ...], multi_layer_raw_feats[1:, ...]

        mid1_feats = layer_feats[self.mid_layer1_idx][1:, ...]
        if self.anomaly_resolve:
            mid1_feats = mean_interpolation(mid1_feats, outlier_indices, H=H, W=W)
        mid1_feats = F.normalize(mid1_feats, dim=2)
        mid1_sim, masked_mid1_sim = self._similarity_and_mask(mid1_feats)

        mid2_feats = layer_feats[self.mid_layer2_idx][1:, ...]
        mid2_feats = F.normalize(mid2_feats, dim=2)
        _, masked_mid2_sim = self._similarity_and_mask(mid2_feats)

        if self.feat_aggregate:
            x1 = self.adaptively_aggregate(x1, sim=masked_mid1_sim)
        x1 = self._forward_last_block(x1, cls_token=x1_cls_token, sim=mid1_sim)
        if self.feat_aggregate:
            x1 = self.adaptively_aggregate(x1, sim=masked_mid2_sim)

        if self.multi_layer_fuse:
            # Keep ScClip behavior: x2 does not do the first adaptive aggregation.
            x2 = self._forward_last_block(x2, cls_token=x2_cls_token, sim=mid1_sim)
            if self.feat_aggregate:
                x2 = self.adaptively_aggregate(x2, sim=masked_mid2_sim)

        if self.multi_layer_fuse:
            x_dense = x1 + x2
        else:
            x_dense = x1

        if self.cls_token_require:
            blk = self.transformer.resblocks[-1]

            x_cls_token = x0[:1, ...]
            x_cls_token = x_cls_token + blk.attn(blk.ln_1(x_cls_token), blk.ln_1(x0), blk.ln_1(x0), need_weights=False)[0]
            x_cls_token = x_cls_token + blk.mlp(blk.ln_2(x_cls_token))
            x = torch.cat([x_cls_token, x_dense], dim=0)

        else:
            x = x_dense

        return x

    def custom_attn(self, attn_layer: nn.Module, x: Tensor, sim: Tensor) -> Tensor:
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim**-0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        sim = (sim - sim.mean(dim=[1, 2], keepdim=True)) * 3.0
        sim[sim < 0.0] = float("-inf")
        sim = sim.repeat_interleave(num_heads, dim=0)

        kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale

        if self.attn_enhance:
            attn_weights = 0.5 * F.softmax(sim, dim=-1) + 0.5 * F.softmax(kk_attn, dim=-1)
        else:
            attn_weights = F.softmax(kk_attn, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        return attn_output

    def adaptively_aggregate(self, x: Tensor, sim: Tensor) -> Tensor:
        sim = sim / (sim.sum(dim=-1, keepdim=True) + 1e-6)
        return torch.einsum("bnm, mbd -> nbd", sim, x)

    def interpolate_pos_encoding(self, x: Tensor, H: int, W: int) -> Tensor:
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1

        if npatch == N and W == H:
            return self.positional_embedding

        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=((H + 0.1) / math.sqrt(N), (W + 0.1) / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(H) == patch_pos_embed.shape[-2] and int(W) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def to_scclip_vit(vit_model: nn.Module) -> nn.Module:
    assert isinstance(vit_model, VisionTransformer)
    vit_model = deepcopy(vit_model)
    vit_model.__class__ = ScClipVisionTransformer
    vit_model.post_init(anomaly_resolve=False)
    return vit_model


class ScClipVisualEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.module = to_scclip_vit(clip_model.visual)

    def forward(self, image, normalize=True):
        x_local = self.module(image)  # [B, HW, D]
        x_global = x_local.mean(dim=1)  # [B, D]

        if normalize:
            x_global = x_global / x_global.norm(dim=-1, keepdim=True)
            x_local = x_local / x_local.norm(dim=-1, keepdim=True)

        return x_global, x_local  # [B, D], [B, HW, D]


class ScClipModel(nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        classnames: list[str],
        # area_threshold=0.01,
        logit_scale=40.0,
        crop_size=224,
        stride=112,
        target_size=56,
    ):
        super().__init__()
        self.visual_encoder = ScClipVisualEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.classnames = classnames
        self.query_names, self.query_indices = self.init_queries(classnames)
        self.query_features = None

        # self.area_threshold = area_threshold
        self.logit_scale = torch.tensor(logit_scale)
        self.crop_size = (crop_size, crop_size) if type(crop_size) == int else crop_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.target_size = (target_size, target_size) if type(target_size) == int else target_size

    def init_queries(self, classnames):
        """Initialize query names and indices, including background categories."""
        query_names = classnames + BACKGROUND_CATEGORY
        query_indices = list(range(len(self.classnames))) + [len(self.classnames)] * len(BACKGROUND_CATEGORY)
        return query_names, torch.tensor(query_indices)

    def to_segmentation_scores(self, logits, logit_scale):
        """Mapping query-wise logits to class-wise logits, by max-pooling over queries belonging to the same class."""
        assert logits.shape[-1] == len(self.query_names)

        if logits.dim() > 2:
            scores = self.to_segmentation_scores(logits.view(-1, logits.shape[-1]), logit_scale=logit_scale)
            return scores.reshape(*logits.shape[:-1], scores.shape[-1])

        B, C0 = logits.shape
        C0, C = len(self.query_names), max(self.query_indices) + 1

        scores = (logit_scale * logits).softmax(dim=-1)  # [B, C0]

        # mapping from original queries to class indices
        query_labels = nn.functional.one_hot(self.query_indices, num_classes=C).to(logits.device)  # [C0, C]
        raw_scores = scores[:, :, None] * query_labels[None, :, :]  # [B, C0, C]
        # max-pooling over queries belonging to the same class
        scores = raw_scores.max(dim=1).values  # [B, C]
        assert scores.shape == torch.Size([B, C])

        return scores  # [B, C]

    def filter_small_areas(self, scores, area_threshold):
        assert scores.shape[-1] == max(self.query_indices) + 1
        B, N, C = scores.shape
        C0, C = len(self.query_names), max(self.query_indices) + 1

        preds = nn.functional.one_hot(scores.argmax(dim=-1), num_classes=C).to(scores.device)  # [B, N, C]

        preds_area = preds[:, :, :-1].sum(dim=1, keepdim=True)  # prone background, [B, 1, C-1]
        scores[:, :, :-1] *= ((preds_area / N) > area_threshold).float()

        return scores

    def to_multi_label_scores(self, scores):
        assert scores.shape[-1] == max(self.query_indices) + 1

        B, N, C = scores.shape
        C0, C = len(self.query_names), max(self.query_indices) + 1

        # eliminate non-top scores per spatial location
        # clf_scores = torch.where(scores == scores.max(dim=0, keepdim=True).values, scores, 0.0)
        clf_scores = scores
        clf_scores = clf_scores[:, :, : len(self.classnames)]
        clf_scores = clf_scores.max(dim=1).values

        return clf_scores  # [B, num_classes]

    @torch.no_grad()
    def encode_query(self, normalize: bool = True) -> Tensor:
        device = get_model_device(self.text_encoder)

        # using default prompt templates for ImageNet
        query_features = []
        for name in self.query_names:
            prompted_t = torch.cat([tokenize(template.format(name)) for template in IMAGENET_TEMPLATES])
            class_embedding = F.normalize(self.text_encoder(prompted_t.to(device)), dim=-1).mean(dim=0)
            query_features.append(class_embedding)
        query_features = torch.stack(query_features, dim=1).to(device).T

        if normalize:
            query_features = F.normalize(query_features, dim=-1)

        return query_features

    def forward_feature(self, image, target_size=None):
        """forward to extract global and local features
        image (B, 3, H0, W0)
        ↓  visual_encoder
        global_logits (B, C0), local_logits (B, H, W, C0) 
        ↓  interpolate
        global_logits (B, C0), local_logits (B, H1, W1, C0)
        """

        B, _, H0, W0 = image.shape

        if self.query_features is None:
            self.query_features = self.encode_query()

        query_features = self.query_features
        img_feats_global, img_feats_local = self.visual_encoder(image, normalize=True)

        B, HW, C0 = img_feats_local.shape
        H, W = H0 // 16, W0 // 16
        assert H * W == HW

        img_feats_local = img_feats_local.view(B, H, W, C0)
        global_logits = img_feats_global @ query_features.T  # [B, C0]
        local_logits = img_feats_local @ query_features.T  # [B, H, W, C0]

        if target_size is not None:
            H1, W1 = target_size
            local_logits = F.interpolate(
                local_logits.permute(0, 3, 1, 2), size=target_size, mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)

        return global_logits, local_logits  # [B, C0], [B, H1, W1, C0]

    def forward_slide(self, image):
        """forward with slide-window cropping strategy
        
        image (B, 3, H0, W0)
        ↓  slide-window cropping
        crop_images (K*B, 3, h_crop, w_crop)
        ↓  forward_feature
        crop_global_logits (K, B, C0), crop_local_logits (K, B, H1, W1, C0)
        ↓  merge
        logits_global (B, C0), logits_local (B, H2, W2, C0)
        """
        B, _, H0, W0 = image.shape

        # crop image into patches with slide-window strategy, and forward each patch
        crop_coords = slide_window_patching(
            image_size=(image.shape[2], image.shape[3]),
            stride=self.stride,
            crop_size=self.crop_size,
        )
        K, C0, H1, W1 = len(crop_coords), len(self.query_names), self.target_size[0], self.target_size[1]
        crop_images = torch.stack([image[:, :, y1:y2, x1:x2] for (y1, x1, y2, x2) in crop_coords], dim=0)
        crop_images = crop_images.view(-1, 3, self.crop_size[0], self.crop_size[1])  # [K * B, 3, H0, W0]

        crop_global_logits, crop_local_logits = self.forward_feature(crop_images, target_size=self.target_size)
        crop_global_logits = crop_global_logits.view(K, B, C0)
        crop_local_logits = crop_local_logits.view(K, B, H1, W1, C0)

        # merge cropped logits back to full-image logits
        H2, W2 = H1 * H0 // self.crop_size[0], W1 * W0 // self.crop_size[1]
        logits_global = crop_global_logits.mean(dim=0)  # [B, C0]
        logits_local = merge_slide_window_logits(crop_local_logits, crop_coords=crop_coords)  # [B, H2, W2, C0]

        return logits_global, logits_local

    def forward_dense(self, image):
        if self.query_features is None:
            self.query_features = self.encode_query()

        logits_global, logits_local = self.forward_slide(image)  # [B, C0], [B, H2, W2, C0]

        B, H2, W2, C0 = logits_local.shape
        seg_logits = logits_local.view(B, -1, C0)  # [B, HW, C0]

        scores = self.to_segmentation_scores(seg_logits, logit_scale=self.logit_scale)  # [B, HW, C]

        scores = scores.view(B, H2, W2, -1)  # [B, H2, W2, C]

        return 

    def forward(self, image):
        if self.query_features is None:
            self.query_features = self.encode_query()

        logits_global, logits_local = self.forward_slide(image)  # [B, C0], [B, H2, W2, C0]

        B, H2, W2, C0 = logits_local.shape
        seg_logits = logits_local.view(B, -1, C0)  # [B, HW, C0]

        scores = self.to_segmentation_scores(seg_logits, logit_scale=self.logit_scale)  # [B, HW, C]

        # scores = self.filter_small_areas(scores, area_threshold=self.area_threshold)  # [B, HW, C]

        clf_scores = self.to_multi_label_scores(scores)  # [B, C]
        return clf_scores

    def learnable_parameters(self):
        return []


class ScClipConfig(BaseConfig):
    # Dataset and input settings
    image_size: Literal[224, 448] | None = 448

    # Training settings
    batch_size: Annotated[int, conint(ge=1)] = 1

    # Model settings
    backbone: Literal["ViT-B/16"] = "ViT-B/16"
    # area_threshold: float = 0.01
    logit_scale: float = 40.0
    crop_size: Literal[112, 224, 448] = 224
    stride: Literal[56, 112, 224] = 112
    target_size: Literal[56, 112, 224] = 56


class ScClipTrainer(BaseTrainer):
    def create_model(self, dataset, cfg):
        classnames = list(dataset.category_name.values())
        clip_model, _ = create_clip(cfg.backbone)
        return ScClipModel(
            clip_model=clip_model,
            classnames=classnames,
            # area_threshold=cfg.area_threshold,
            logit_scale=cfg.logit_scale,
            target_size=cfg.target_size,
        )

    @torch.inference_mode()
    def forward(self, batch, mode):
        image, label, indices = batch
        image, label = image.to(self.device), label.to(self.device)

        with self.autocast():
            # preds, scores, logits = self.model(image)
            preds = self.model(image)

        return {
            "indices": indices,
            "preds": preds.float(),
            "target": label.long(),
            # "scores": scores.float(),  # spatial scores
            # "logits": logits.float(),  # raw spatial logits
        }


def main():
    entry(config=ScClipConfig, trainer=ScClipTrainer, train=False, test=True)
