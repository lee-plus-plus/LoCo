import torch
import torch.nn.functional as F
from torch import nn, Tensor
from copy import deepcopy
from typing import Literal
from torchvision.models.resnet import ResNet
from .clip import tokenize, _tokenizer
from .clip.model import ModifiedResNet, VisionTransformer, CLIP
from .clip.adaptive import adaptive_pos_embed


class DenseResNet(ResNet):
    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x: [B, C, H, W]
        x_global = self.avgpool(x)
        x_global = torch.flatten(x_global, start_dim=1, end_dim=-1)
        x_global = self.fc(x_global)

        x_local = torch.flatten(x.permute(2, 3, 0, 1), start_dim=0, end_dim=1)
        x_local = self.fc(x_local)
        x_local = x_local.permute(1, 0, 2)  # [B, HW, D]

        return x_global, x_local  # [B, HW, D], [B, D]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)  # [B, HW, D], [B, D]


class DenseModifiedResNet(ModifiedResNet):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        b, c, h, w = x.shape

        x_local = x.reshape(b, c, h * w).permute(2, 0, 1)  # [hw, b, c]
        # x_local = torch.cat([x_local.mean(dim=0, keepdim=True), x_local], dim=0)  # [hw+1, b, c]

        # pos_embed = self.attnpool.positional_embedding
        # pos_embed = adaptive_pos_embed(pos_embed, H=7, W=7, h=h, w=w)  # [hw+1, c]
        # x_local = x_local + pos_embed[:, None, :]

        x_local = F.linear(x_local, self.attnpool.v_proj.weight, self.attnpool.v_proj.bias)
        x_local = F.linear(x_local, self.attnpool.c_proj.weight, self.attnpool.c_proj.bias)

        x_global = self.attnpool(x, use_pos_embed=True)  # no positional embedding

        return x_local, x_global  # [HW, B, D], [B, D]


class DenseVisionTransformer(VisionTransformer):
    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        H = W = int((self.positional_embedding.shape[0] - 1) ** 0.5)
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

        pos_embed = self.positional_embedding
        pos_embed = adaptive_pos_embed(pos_embed, H=H, W=W, h=h, w=w)
        x = x + pos_embed.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x[:, :, :])  # return both cls token and image tokens

        if self.proj is not None:
            x = x @ self.proj

        x_global, x_local = x[:, 0, :], x[:, 1:, :].permute(1, 0, 2)
        return x_local, x_global  # [HW, B, D], [B, D]


def to_dense_resnet(model: ResNet) -> DenseResNet:
    model = deepcopy(model)
    model.__class__ = DenseResNet
    return model


def to_dense_modified_resnet(model: ModifiedResNet) -> DenseModifiedResNet:
    model = deepcopy(model)
    model.__class__ = DenseModifiedResNet
    return model


def to_dense_vit(vit: VisionTransformer) -> DenseVisionTransformer:
    vit = deepcopy(vit)
    vit.__class__ = DenseVisionTransformer
    return vit


def to_dense_visual_encoder(visual_encoder: nn.Module):
    if "ModifiedResNet" in visual_encoder.__class__.__name__:
        visual_encoder = to_dense_modified_resnet(visual_encoder)
    elif "VisionTransformer" in visual_encoder.__class__.__name__:
        visual_encoder = to_dense_vit(visual_encoder)
    elif "ResNet" in visual_encoder.__class__.__name__:
        visual_encoder = to_dense_resnet(visual_encoder)
    
    return visual_encoder


class VisualEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.module = deepcopy(clip_model.visual)

    def forward(self, image, normalize=False):
        x = self.module(image)
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x


class DenseVisualEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.module = to_dense_visual_encoder(clip_model.visual)

    def forward(self, image, normalize=False, spatialize=False):
        x_local, x_global = self.module(image)
        x_local = x_local.permute(1, 0, 2)

        if normalize:
            x_local = x_local / x_local.norm(dim=-1, keepdim=True)
            x_global = x_global / x_global.norm(dim=-1, keepdim=True)

        if spatialize:
            B, HW, D = x_local.shape
            H = W = int(HW**0.5)
            x_local = x_local.view(B, H, W, D)

        return x_global, x_local  # [B, D], [B, HW, D]


class TextEncoder(nn.Module):
    """Text Encoder for Learnable Prompt (optional)"""

    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer = deepcopy(clip_model.transformer)
        self.token_embedding = deepcopy(clip_model.token_embedding)
        self.positional_embedding = deepcopy(clip_model.positional_embedding)
        self.ln_final = deepcopy(clip_model.ln_final)
        self.text_projection = deepcopy(clip_model.text_projection)
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts, prompts=None, *, eot_only=True, normalize=False):
        """return encoded text embedding

        prompts: embedding sequence
        tokenized_prompts: token sequence, for locating SOS and EOS position
        """
        if prompts is None:
            prompts = self.token_embedding(tokenized_prompts).type(self.dtype)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # [B, L, D] -> [L, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [L, B, D] -> [B, L, D]
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]

        if eot_only:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x_eot = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]  # [B, D]
            z_eot = x_eot @ self.text_projection  # [B, D]
            if normalize:
                z_eot = z_eot / z_eot.norm(dim=-1, keepdim=True)
            return z_eot
        else:
            z = x @ self.text_projection  # [B, L, D]
            z_eot = z[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]  # [B, D]
            if normalize:
                z_eot = z_eot / z_eot.norm(dim=-1, keepdim=True)
                z = z / z.norm(dim=-1, keepdim=True)
            return z_eot, z


class DenseTextEncoder(TextEncoder):
    def __init__(self, clip_model: CLIP):
        super().__init__(clip_model)

    def forward(self, tokenized_prompts, prompts=None, normalize=False):
        x_global, x_local = super().forward(tokenized_prompts, prompts, eot_only=False, normalize=normalize)

        return x_global, x_local  # [B, D], [B, L, D]


class PromptLearner(nn.Module):
    """Genereate Learnable Prompt
    prompts: [SOS] + [contex] + [CLS] + [EOS]

    token_prefix: (num_classes, 1, dim)
    contex: (num_classes, n_context, dim)
    token_suffix: (num_classes, *, dim)
    tokenized_prompts: (77,)
    """

    def __init__(
        self,
        clip_model: CLIP,
        classnames: list[str],
        *,
        position: Literal["front", "middle", "end"] = "end",
        n_context: int = 16,
        context_init: str | None = None,
        csc: bool = False,
    ):
        super().__init__()
        num_classes = len(classnames)
        dtype = clip_model.dtype
        dim_ctx = clip_model.ln_final.weight.shape[0]

        # generate intial prompt context: 'ctx_vectors' as vectors and 'prompt_prefix' for readable string
        # if context_init is given, use use given words to initialize context vectors,
        # othervise, use randomly generated context vectors
        if context_init:
            context_init = context_init.replace("_", " ")
            n_context = len(context_init.split(" "))
            prompt_prefix = context_init
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenize(context_init)).type(dtype)

            if csc:
                ctx_vectors = embedding[0:1, 1 : 1 + n_context, :].repeat(num_classes, 1, 1)
            else:
                ctx_vectors = embedding[0, 1 : 1 + n_context, :]

        else:
            if csc:
                # Initializing class-specific contexts
                ctx_vectors = torch.empty(num_classes, n_context, dim_ctx, dtype=dtype)
            else:
                # Initializing a generic context
                ctx_vectors = torch.empty(n_context, dim_ctx, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_context)

        self.prompt_prefix = prompt_prefix
        self.context = nn.Parameter(ctx_vectors)  # to be optimized
        self.n_context = n_context  # length of self.context
        self.num_classes = num_classes

        self.classnames = [name.replace("_", " ") for name in classnames]
        n_classname = [len(_tokenizer.encode(name)) for name in self.classnames]
        prompts = [prompt_prefix + " " + name + "." for name in self.classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_context :, :])  # CLS, '.', EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.n_classname = n_classname
        self.position = position

    def forward(self):
        # prompts = torch.cat([self.token_prefix, self.context, self.token_suffix])
        prefix = self.token_prefix
        suffix = self.token_suffix
        context = self.context
        if context.dim() == 2:
            context = context.unsqueeze(0).expand(self.num_classes, -1, -1)

        prompts = []
        for i in range(self.num_classes):
            n_classname = self.n_classname[i]
            half_n_ctx = self.n_context // 2
            prefix_i = prefix[i : i + 1, :, :]  # SOS
            class_i = suffix[i : i + 1, :n_classname, :]  # classname
            suffix_i = suffix[i : i + 1, n_classname:, :]  # '.', EOS, ...
            ctx_i_half1 = context[i : i + 1, :half_n_ctx, :]  # learnable context (half-front)
            ctx_i_half2 = context[i : i + 1, half_n_ctx:, :]  # learnable context (half-back)

            if self.position == "end":
                prompt = torch.cat([prefix_i, ctx_i_half1, ctx_i_half2, class_i, suffix_i], dim=1)
            elif self.position == "middle":
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
            elif self.position == "front":
                prompt = torch.cat([prefix_i, class_i, ctx_i_half1, ctx_i_half2, suffix_i], dim=1)
            else:
                raise ValueError(f"unsupported position ‘{self.position}’")
            prompts.append(prompt)

        prompts = torch.cat(prompts, dim=0)
        return prompts


class DualPromptLearner(nn.Module):
    """Genereate Learnable Prompt
    prompts: [SOS] + [contex] + [CLS] + [EOS]

    token_prefix: (num_classes, 1, dim)
    contex: (num_classes, n_context, dim)
    token_suffix: (num_classes, *, dim)
    tokenized_prompts: (77,)
    """

    def __init__(
        self,
        clip_model: CLIP,
        classnames: list[str],
        *,
        position: Literal["front", "middle", "end"] = "end",
        n_context: int = 16,
        context_init: tuple[str, str] | None = None,
        csc: bool = False,
    ):
        super().__init__()
        num_classes = len(classnames)
        dtype = next(clip_model.parameters()).dtype
        dim_ctx = clip_model.ln_final.weight.shape[0]

        # generate intial prompt context: 'ctx_vectors' as vectors and 'prompt_prefix' for readable string
        # if context_init is given, use use given words to initialize context vectors,
        # othervise, use randomly generated context vectors
        if context_init:
            context_init_pos, context_init_neg = (elem.replace("_", " ").strip() for elem in context_init)

            assert len(context_init_pos.split(" ")) == len(context_init_neg.split(" "))
            n_context = len(context_init_pos.split(" "))
            prompt_prefix = context_init_pos

            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(tokenize(context_init_pos)).type(dtype)
                embedding_neg = clip_model.token_embedding(tokenize(context_init_neg)).type(dtype)

            if csc:  # Initializing class-specific contexts
                ctx_vectors_pos = embedding_pos[0:1, 1 : 1 + n_context, :].repeat(num_classes, 1, 1)
                ctx_vectors_neg = embedding_neg[0:1, 1 : 1 + n_context, :].repeat(num_classes, 1, 1)
            else:  # Initializing a generic context
                ctx_vectors_pos = embedding_pos[0, 1 : 1 + n_context, :]
                ctx_vectors_neg = embedding_neg[0, 1 : 1 + n_context, :]

        else:
            if csc:  # Initializing class-specific contexts
                ctx_vectors_pos = torch.empty(num_classes, n_context, dim_ctx, dtype=dtype)
                ctx_vectors_neg = torch.empty(num_classes, n_context, dim_ctx, dtype=dtype)
            else:  # Initializing a generic context
                ctx_vectors_pos = torch.empty(n_context, dim_ctx, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_context, dim_ctx, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix = " ".join(["X"] * n_context)

        self.prompt_prefix = prompt_prefix
        self.context_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.context_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.n_context = n_context  # length of self.context
        self.num_classes = num_classes

        self.classnames = [name.replace("_", " ") for name in classnames]
        n_classname = [len(_tokenizer.encode(name)) for name in self.classnames]
        prompts = [prompt_prefix + " " + name + "." for name in self.classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).detach()

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_context :, :])  # CLS, '.', EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor (list of int)
        self.n_classname = n_classname
        self.position = position

        self.temperature = nn.Parameter(torch.tensor(3.91, dtype=dtype))
        self.spatial_T = nn.Parameter(torch.tensor(3.0, dtype=dtype))

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        context_pos = self.context_pos
        context_neg = self.context_neg
        if context_pos.dim() == 2:
            context_pos = context_pos.unsqueeze(0).expand(self.num_classes, -1, -1)
        if context_neg.dim() == 2:
            context_neg = context_neg.unsqueeze(0).expand(self.num_classes, -1, -1)

        prompts_pos, prompts_neg = [], []
        for i in range(self.num_classes):
            n_classname = self.n_classname[i]
            prefix_i = prefix[i : i + 1, :, :]  # SOS
            class_i = suffix[i : i + 1, :n_classname, :]  # classname
            suffix_i = suffix[i : i + 1, n_classname:, :]  # '.', EOS, ...
            ctx_pos_i = context_pos[i : i + 1]
            ctx_neg_i = context_neg[i : i + 1]

            prompts_pos.append(self.concat_prompt(prefix_i, class_i, suffix_i, ctx_pos_i, position=self.position))
            prompts_neg.append(self.concat_prompt(prefix_i, class_i, suffix_i, ctx_neg_i, position=self.position))

        prompts_pos = torch.cat(prompts_pos, dim=0)
        prompts_neg = torch.cat(prompts_neg, dim=0)

        # return prompts_pos, prompts_neg, self.temperature, self.spatial_T
        return prompts_pos, prompts_neg

    def concat_prompt(self, prefix_i, class_i, suffix_i, context_i, position):
        half_n_ctx = context_i.shape[1] // 2
        ctx_i_half1 = context_i[:, :half_n_ctx, :]
        ctx_i_half2 = context_i[:, half_n_ctx:, :]

        if position == "end":
            prompt = torch.cat([prefix_i, ctx_i_half1, ctx_i_half2, class_i, suffix_i], dim=1)
        elif position == "middle":
            prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
        elif position == "front":
            prompt = torch.cat([prefix_i, class_i, ctx_i_half1, ctx_i_half2, suffix_i], dim=1)
        return prompt


class PromptTuningClassifier(nn.Module):
    def __init__(self, text_encoder: nn.Module, prompt_learner: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder
        self.prompt_learner = prompt_learner

    def forward(self, image_features):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_prototypes = self.text_encoder(tokenized_prompts, prompts)

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_prototypes = text_prototypes / text_prototypes.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits = image_features @ text_prototypes.t()  # [B, num_classes]

        return logits