import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def detectmap_proc(detected_map, h, w, rgbbgr_mode=False):
    detected_map = HWC3(detected_map)

    if rgbbgr_mode:  # or module == "normal_map" 
        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float() / 255.0
    else:
        control = torch.from_numpy(detected_map.copy()).float() / 255.0
        
    control = rearrange(control, 'h w c -> c h w')
    control = Resize((h,w), interpolation=InterpolationMode.BICUBIC)(control)

    return control
