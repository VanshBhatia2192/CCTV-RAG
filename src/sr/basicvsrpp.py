"""BasicVSR++ (Chan et al., CVPR 2022) in plain PyTorch.

Same architecture and parameter names as the mmedit/MMagic `BasicVSRPlusPlus` generator, so the official
REDS4 checkpoint loads directly (see `load_weights`). The only op mmcv supplied, modulated deformable
convolution, comes from `torchvision.ops.deform_conv2d` here, so nothing needs compiling and it runs on any
GPU torch supports (H100 included).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def flow_warp(x, flow, padding_mode="zeros"):
    """Warp x (n,c,h,w) by flow (n,h,w,2) given in pixels, (dx, dy) order."""
    _, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack((gx, gy), 2).type_as(x) + flow
    gx = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gx, gy), 3), mode="bilinear", padding_mode=padding_mode, align_corners=True)


class ResidualBlockNoBN(nn.Module):
    def __init__(s, c=64):
        super().__init__()
        s.conv1, s.conv2 = nn.Conv2d(c, c, 3, 1, 1), nn.Conv2d(c, c, 3, 1, 1)
        s.relu = nn.ReLU(inplace=True)

    def forward(s, x):
        return x + s.conv2(s.relu(s.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(s, cin, cout=64, num_blocks=30):
        super().__init__()
        s.main = nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                               nn.Sequential(*[ResidualBlockNoBN(cout) for _ in range(num_blocks)]))

    def forward(s, x):
        return s.main(x)


class PixelShufflePack(nn.Module):
    def __init__(s, cin, cout, scale, k=3):
        super().__init__()
        s.scale = scale
        s.upsample_conv = nn.Conv2d(cin, cout * scale * scale, k, padding=(k - 1) // 2)

    def forward(s, x):
        return F.pixel_shuffle(s.upsample_conv(x), s.scale)


class _Conv(nn.Module):                      # mirrors mmcv ConvModule's `.conv` key
    def __init__(s, cin, cout, act=True):
        super().__init__()
        s.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        s.act = act

    def forward(s, x):
        x = s.conv(x)
        return F.relu(x) if s.act else x


class SPyNetBasicModule(nn.Module):
    def __init__(s):
        super().__init__()
        s.basic_module = nn.Sequential(_Conv(8, 32), _Conv(32, 64), _Conv(64, 32), _Conv(32, 16), _Conv(16, 2, act=False))

    def forward(s, x):
        return s.basic_module(x)


class SPyNet(nn.Module):
    """Optical flow network used for alignment. Returns flow from ref to supp, (dx, dy)."""
    def __init__(s):
        super().__init__()
        s.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        s.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(s, ref, supp):
        n, _, h, w = ref.shape
        ref, supp = [(ref - s.mean) / s.std], [(supp - s.mean) / s.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for lv in range(len(ref)):
            up = flow if lv == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supp[lv], up.permute(0, 2, 3, 1), padding_mode="border")
            flow = up + s.basic_module[lv](torch.cat([ref[lv], warped, up], 1))
        return flow

    def forward(s, ref, supp):
        h, w = ref.shape[2:4]
        hu, wu = -(-h // 32) * 32, -(-w // 32) * 32
        ref = F.interpolate(ref, size=(hu, wu), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(hu, wu), mode="bilinear", align_corners=False)
        flow = F.interpolate(s.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        return torch.stack((flow[:, 0] * (w / wu), flow[:, 1] * (h / hu)), 1)


class SecondOrderDeformableAlignment(nn.Module):
    """Flow-guided modulated deformable conv over the n-1 and n-2 propagated features."""
    def __init__(s, cin, cout, deform_groups=16, max_residue_magnitude=10):
        super().__init__()
        s.deform_groups, s.max_residue_magnitude = deform_groups, max_residue_magnitude
        s.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        s.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(s.weight, a=5 ** 0.5)
        lr = lambda: nn.LeakyReLU(0.1, inplace=True)
        s.conv_offset = nn.Sequential(nn.Conv2d(3 * cout + 4, cout, 3, 1, 1), lr(), nn.Conv2d(cout, cout, 3, 1, 1), lr(),
                                      nn.Conv2d(cout, cout, 3, 1, 1), lr(), nn.Conv2d(cout, 27 * deform_groups, 3, 1, 1))
        nn.init.zeros_(s.conv_offset[-1].weight); nn.init.zeros_(s.conv_offset[-1].bias)

    def forward(s, x, extra, flow_1, flow_2):
        o1, o2, mask = torch.chunk(s.conv_offset(torch.cat([extra, flow_1, flow_2], 1)), 3, dim=1)
        off_1, off_2 = torch.chunk(s.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), 1)), 2, dim=1)
        off_1 = off_1 + flow_1.flip(1).repeat(1, off_1.size(1) // 2, 1, 1)
        off_2 = off_2 + flow_2.flip(1).repeat(1, off_2.size(1) // 2, 1, 1)
        offset = torch.cat([off_1, off_2], 1)
        # torchvision has no low-precision kernel for this op, so run it in fp32 under autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            return deform_conv2d(x.float(), offset.float(), s.weight.float(), s.bias.float(), padding=(1, 1),
                                 mask=torch.sigmoid(mask.float()))


class BasicVSRPlusPlus(nn.Module):
    """Input (n, t, 3, h, w) in [0, 1], output (n, t, 3, 4h, 4w). h and w must be >= 64."""
    MODULES = ("backward_1", "forward_1", "backward_2", "forward_2")

    def __init__(s, mid_channels=64, num_blocks=7, max_residue_magnitude=10):
        super().__init__()
        s.mid_channels = mid_channels
        s.spynet = SPyNet()
        s.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        s.deform_align, s.backbone = nn.ModuleDict(), nn.ModuleDict()
        for i, m in enumerate(s.MODULES):
            s.deform_align[m] = SecondOrderDeformableAlignment(2 * mid_channels, mid_channels, 16, max_residue_magnitude)
            s.backbone[m] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)
        s.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        s.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2)
        s.upsample2 = PixelShufflePack(mid_channels, 64, 2)
        s.conv_hr, s.conv_last = nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 3, 3, 1, 1)
        s.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        s.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def compute_flow(s, lqs):
        n, t, c, h, w = lqs.shape
        a, b = lqs[:, :-1].reshape(-1, c, h, w).float(), lqs[:, 1:].reshape(-1, c, h, w).float()
        with torch.autocast(device_type=lqs.device.type, enabled=False):   # flow is precision-sensitive
            backward = s.spynet(a, b).view(n, t - 1, 2, h, w)
            forward = s.spynet(b, a).view(n, t - 1, 2, h, w)
        return forward, backward

    def propagate(s, feats, flows, name):
        n, t, _, h, w = flows.shape
        frame_idx, flow_idx = list(range(t + 1)), list(range(-1, t))
        if name.startswith("backward"):
            frame_idx = frame_idx[::-1]; flow_idx = frame_idx
        prop = flows.new_zeros(n, s.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            cur = feats["spatial"][idx]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2, flow_n2, cond_n2 = torch.zeros_like(prop), torch.zeros_like(flow_n1), torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[name][-2]
                    flow_n2 = flow_n1 + flow_warp(flows[:, flow_idx[i - 1]], flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, cur, cond_n2], 1)
                prop = s.deform_align[name](torch.cat([prop, feat_n2], 1), cond, flow_n1, flow_n2).to(cur.dtype)
            feat = [cur] + [feats[k][idx] for k in feats if k not in ("spatial", name)] + [prop]
            prop = prop + s.backbone[name](torch.cat(feat, 1))
            feats[name].append(prop)
        if name.startswith("backward"):
            feats[name] = feats[name][::-1]
        return feats

    def forward(s, lqs):
        n, t, c, h, w = lqs.shape
        assert h >= 64 and w >= 64, f"BasicVSR++ needs input of at least 64x64, got {h}x{w}"
        assert t >= 2, "BasicVSR++ needs at least 2 frames"
        f = s.feat_extract(lqs.reshape(-1, c, h, w)).view(n, t, -1, h, w)
        feats = {"spatial": [f[:, i] for i in range(t)]}
        flows_fwd, flows_bwd = s.compute_flow(lqs)
        for m in s.MODULES:
            feats[m] = []
            feats = s.propagate(feats, flows_bwd if m.startswith("backward") else flows_fwd, m)
        out = []
        for i in range(t):
            hr = torch.cat([feats["spatial"][i]] + [feats[m][i] for m in s.MODULES], 1)
            hr = s.reconstruction(hr)
            hr = s.lrelu(s.upsample1(hr)); hr = s.lrelu(s.upsample2(hr)); hr = s.lrelu(s.conv_hr(hr))
            out.append(s.conv_last(hr) + s.img_upsample(lqs[:, i]))
        return torch.stack(out, 1)


def load_weights(model, path, strict=True):
    """Load an mmedit/MMagic BasicVSR++ checkpoint (keys `generator.*`) or one saved by training/sr/train.py."""
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck.get("params", ck.get("model", ck))) if isinstance(ck, dict) else ck
    if any(k.startswith("generator.") for k in sd):
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return model


def build(weights=None, device="cpu"):
    m = BasicVSRPlusPlus()
    if weights:
        load_weights(m, weights)
    return m.to(device).eval()
