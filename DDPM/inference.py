# inference.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple

import base64
from PIL import Image
import io
import os

import torch
from torchvision.utils import make_grid

from utils.tools import load_yaml
from utils.engine import DDPMSampler
from model.UNet import UNet


def pil_to_base64_png(img: Image.Image) -> str:
    # convert PIL image to base64 PNG string for sending over HTTP
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_cfg_values(cfg: Dict[str, Any]) -> Tuple[int, int, Tuple[float, float], int, int, float]:
    # pull out all the values we need from the yaml, with fallbacks in case keys differ
    img_size = (
        cfg.get("img_size")
        or cfg.get("image_size")
        or cfg.get("data", {}).get("img_size")
        or cfg.get("dataset", {}).get("img_size")
        or 64
    )

    in_channels = (
        cfg.get("in_channels")
        or cfg.get("model", {}).get("in_channels")
        or 3
    )

    beta_start = (
        cfg.get("beta_start")
        or cfg.get("diffusion", {}).get("beta_start")
        or cfg.get("trainer", {}).get("beta_start")
        or 1e-4
    )
    beta_end = (
        cfg.get("beta_end")
        or cfg.get("diffusion", {}).get("beta_end")
        or cfg.get("trainer", {}).get("beta_end")
        or 2e-2
    )
    beta = (float(beta_start), float(beta_end))

    T = (
        cfg.get("T")
        or cfg.get("diffusion", {}).get("T")
        or cfg.get("trainer", {}).get("T")
        or 1000
    )
    T = int(T)

    num_class = (
        cfg.get("num_class")
        or cfg.get("model", {}).get("num_class")
        or cfg.get("dataset", {}).get("num_class")
        or 12
    )
    num_class = int(num_class)

    default_guidance_scale = (
        cfg.get("guidance_scale")
        or cfg.get("sampling", {}).get("guidance_scale")
        or cfg.get("diffusion", {}).get("guidance_scale")
        or 5.0
    )
    default_guidance_scale = float(default_guidance_scale)

    return int(img_size), int(in_channels), beta, T, num_class, default_guidance_scale


def _build_unet(cfg: Dict[str, Any]) -> UNet:
    # build UNet from config, falls back to sensible defaults if keys are missing
    m = cfg.get("model", {})
    return UNet(
        in_channels=m.get("in_channels", cfg.get("in_channels", 3)),
        model_channels=m.get("model_channels", 128),
        out_channels=m.get("out_channels", 3),
        num_res_blocks=m.get("num_res_blocks", 2),
        num_class=m.get("num_class", cfg.get("num_class", 12)),
        attention_resolutions=tuple(m.get("attention_resolutions", (8, 16))),
        dropout=m.get("dropout", 0.0),
        channel_mult=tuple(m.get("channel_mult", (1, 2, 2, 2))),
        conv_resample=m.get("conv_resample", True),
        num_heads=m.get("num_heads", 4),
    )


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # handles the different ways checkpoints can be saved
    # tries the most common wrapper keys first, then assumes its a raw state dict
    if isinstance(ckpt, dict):
        for key in ("ema_state_dict", "ema", "model_ema", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unrecognized checkpoint format. Inspect keys in your .pth file.")


def _clean_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # strips prefixes like "module." (DDP) or "model." (wrappers) so load_state_dict works
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "trainer.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_model(ckpt_path: str, config_path: str = "config.yaml", device: str = "cpu"):
    # loads everything and returns a ready-to-use DDPMSampler
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_yaml(config_path)
    _, _, beta, T, _, _ = _get_cfg_values(cfg)

    m = cfg["Model"]
    model = UNet(
        in_channels=m["in_channels"],
        model_channels=m["model_channels"],
        out_channels=m["out_channels"],
        num_res_blocks=m["num_res_blocks"],
        num_class=m["num_class"],
        attention_resolutions=tuple(m["attention_resolutions"]),
        dropout=m["dropout"],
        channel_mult=tuple(m["channel_mult"]),
        conv_resample=m["conv_resample"],
        num_heads=m["num_heads"],
    )
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)
    state = _clean_keys(state)

    # strict=False so it doesn't blow up if there are minor key mismatches

    missing, unexpected = model.load_state_dict(state, strict=False)
    
    sampler = DDPMSampler(model=model, beta=beta, T=T).to(device)
    sampler.eval()
    return sampler


@torch.no_grad()
def generate_images(
    model,                          # DDPMSampler
    device: str,
    num_images: int = 4,
    seed: int = 0,
    guidance_scale: float = 5.0,
    img_size: int = 64,
    in_channels: int = 3,
    breed_id: Optional[int] = None,   # single-breed mode
    breed_a: Optional[int] = None,    # mix mode - first breed
    breed_b: Optional[int] = None,    # mix mode - second breed
    mix_ratio: float = 0.5,           # how much of breed_a vs breed_b (0=all A, 1=all B)
    progress_callback=None,           # optional _ProgressCallback from server.py, can be None
) -> List[Image.Image]:

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    x_T = torch.randn(num_images, in_channels, img_size, img_size, device=device, generator=g)

    class_labels = None

    # two-breed mix mode
    if breed_a is not None and breed_b is not None:
        labels_a = torch.full((num_images,), int(breed_a), device=device, dtype=torch.long)
        labels_b = torch.full((num_images,), int(breed_b), device=device, dtype=torch.long)
        mix_ratio = float(mix_ratio)
        mix_ratio = max(0.0, min(1.0, mix_ratio))  # clamp just in case
        class_labels = (labels_a, labels_b, mix_ratio)

    # single breed mode
    elif breed_id is not None:
        class_labels = torch.full((num_images,), int(breed_id), device=device, dtype=torch.long)

    # patch tqdm before the sampler runs so we get step-level progress
    # the sampler uses tqdm internally and we can't easily hook into it otherwise
    if progress_callback is not None:
        import utils.engine as _engine_mod
        _real_tqdm = _engine_mod.tqdm

        class _PatchedTqdm(_real_tqdm):
            # intercepts tqdm updates and forwards them to our progress callback
            def update(self, n=1):
                super().update(n)
                progress_callback.update(n=n, step=self.n)

        # patch the tqdm name inside engine.py specifically - that's where the loop lives
        # patching tqdm.tqdm globally doesnt work because engine does "from tqdm import tqdm"
        _engine_mod.tqdm = _PatchedTqdm
        try:
            x_0 = model(
                x_T=x_T,
                class_labels=class_labels,
                guidance_scale=float(guidance_scale),
                only_return_x_0=True,
            )
        finally:
            # always restore even if the model crashes
            _engine_mod.tqdm = _real_tqdm
    else:
        # no progress tracking, just run normally
        x_0 = model(
            x_T=x_T,
            class_labels=class_labels,
            guidance_scale=float(guidance_scale),
            only_return_x_0=True,
        )

    # convert output tensors to PIL images
    imgs: List[Image.Image] = []
    for i in range(x_0.shape[0]):
        x = x_0[i].detach().float().cpu()
        # model outputs in [-1, 1] range so we remap to [0, 1]
        if x.min() < 0:
            x = (x.clamp(-1, 1) + 1) / 2.0
        x = x.clamp(0, 1)
        arr = (x * 255).byte().permute(1, 2, 0).numpy()
        imgs.append(Image.fromarray(arr, mode="RGB"))

    return imgs
