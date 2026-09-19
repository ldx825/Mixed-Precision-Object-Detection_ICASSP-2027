"""Fake quantization utilities for F-LBQ-compatible diagnostic reconstruction.

Discovery phase design (see DECISIONS.md D004/D007):
- Quantization unit: Swin block linear groups (attn.qkv, attn.proj, mlp.fc1, mlp.fc2)
- Quantizer: symmetric min-max uniform, per-tensor, weights only (bias stays FP)
- Perturbation: quantize exactly one group to b bits, everything else FP
- Restore: snapshot/restore of parameter data (bit-exact recovery)
"""
import torch
import torch.nn as nn


def quantize_weight_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric min-max uniform fake-quant (per-tensor). bits>=32 returns clone."""
    if bits >= 32:
        return w.clone()
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max().item() / qmax
    if scale <= 0:
        return w.clone()
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    return q


def swin_block_groups(model) -> list[tuple[str, list[str]]]:
    """Enumerate Swin-T block weight groups of an mmdet Mask R-CNN model.

    mmdet 3.x SwinBlock layout: attn.w_msa.{qkv,proj}, ffn.layers.{0.0,1}.
    Returns list of (group_name, [param_name, ...]) for every block's
    qkv / proj / fc1 / fc2 weights (norm/bias excluded).
    """
    pd = dict(model.named_parameters())
    backbone = model.backbone
    stages_attr = "stages" if hasattr(backbone, "stages") else "layers"
    stages = getattr(backbone, stages_attr)
    groups = []
    for li, stage in enumerate(stages):
        for bi, blk in enumerate(stage.blocks):
            prefix = f"backbone.{stages_attr}.{li}.blocks.{bi}"
            candidates = [
                ("qkv", f"{prefix}.attn.w_msa.qkv.weight"),
                ("qkv", f"{prefix}.attn.qkv.weight"),
                ("proj", f"{prefix}.attn.w_msa.proj.weight"),
                ("proj", f"{prefix}.attn.proj.weight"),
                ("fc1", f"{prefix}.ffn.layers.0.0.weight"),
                ("fc1", f"{prefix}.mlp.fc1.weight"),
                ("fc2", f"{prefix}.ffn.layers.1.weight"),
                ("fc2", f"{prefix}.mlp.fc2.weight"),
            ]
            seen = set()
            for tag, pname in candidates:
                if tag in seen or pname not in pd:
                    continue
                seen.add(tag)
                groups.append((f"L{li}B{bi}.{tag}", [pname]))
    return groups


def all_group_names(model) -> list[str]:
    return [g for g, _ in swin_block_groups(model)]


def snapshot_params(model, param_names: list[str]) -> dict:
    pd = dict(model.named_parameters())
    return {n: pd[n].data.clone() for n in param_names}


def restore_params(model, snap: dict):
    pd = dict(model.named_parameters())
    for n, t in snap.items():
        pd[n].data.copy_(t)
    torch.cuda.synchronize()


class GroupQuantApplier:
    """Apply/restore per-group weight quantization on a model (in-place)."""

    def __init__(self, model):
        self.model = model
        self.groups = dict(swin_block_groups(model))
        # snapshot of every group param (for restore)
        all_params = [p for _, plist in self.groups.items() for p in plist]
        self._snap = snapshot_params(model, all_params)

    def apply(self, group_name: str, bits: int):
        """Quantize one group's weights. group_name from swin_block_groups."""
        self.restore()
        pd = dict(self.model.named_parameters())
        for pname in self.groups[group_name]:
            pd[pname].data.copy_(quantize_weight_tensor(pd[pname].data, bits))
        torch.cuda.synchronize()

    def apply_many(self, group_bits: dict[str, int]):
        """Quantize multiple groups with specified bits (e.g., uniform quant)."""
        self.restore()
        pd = dict(self.model.named_parameters())
        for gname, bits in group_bits.items():
            for pname in self.groups[gname]:
                pd[pname].data.copy_(quantize_weight_tensor(pd[pname].data, bits))
        torch.cuda.synchronize()

    def apply_uniform(self, bits: int):
        self.apply_many({g: bits for g in self.groups})

    def restore(self):
        restore_params(self.model, self._snap)

    def group_param_count(self) -> dict:
        pd = dict(self.model.named_parameters())
        return {g: sum(pd[p].numel() for p in plist) for g, plist in self.groups.items()}
