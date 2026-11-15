import importlib, sys, types
from pathlib import Path
import torch, torch.nn as nn, yaml

OG = Path("third_party/OpenGait")
CK = Path("models/pretrained/opengait")
sys.path.insert(0, str(OG / "opengait"))
MODELS = {
    "deepgaitv2_grew": ("configs/deepgaitv2/DeepGaitV2_grew.yaml", "GREW/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-180000.pt"),
    "gaitbase_grew": ("configs/gaitbase/gaitbase_da_grew.yaml", "GREW/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-180000.pt"),
    "deepgaitv2_gait3d": ("configs/deepgaitv2/DeepGaitV2_gait3d.yaml", "Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt"),
}

def _import(mod):
    # modeling/models/__init__.py imports every model in the repo (many extra deps): stub the package
    if "modeling.models" not in sys.modules:
        importlib.import_module("modeling")
        pkg = types.ModuleType("modeling.models")
        pkg.__path__ = [str(OG / "opengait/modeling/models")]
        sys.modules["modeling.models"] = pkg
    return importlib.import_module(f"modeling.models.{mod}")

def load(name, dev="cpu", ckpt=None, pretrained=True):
    """ckpt overrides the default OpenGait checkpoint path (same {'model': state_dict} format);
    pretrained=False builds a randomly initialised model (smoke tests / from-scratch)."""
    cfg_rel, ck_rel = MODELS[name]
    cfg = yaml.safe_load(open(OG / cfg_rel))
    mc = cfg["model_cfg"]
    cls = getattr(_import({"DeepGaitV2": "deepgaitv2", "Baseline": "baseline"}[mc["model"]]), mc["model"])
    m = cls.__new__(cls)
    nn.Module.__init__(m)          # skip BaseModel.__init__ (needs torch.distributed + dataset)
    m.build_network(mc)
    if pretrained:
        ck = torch.load(Path(ckpt) if ckpt else CK / ck_rel, map_location="cpu", weights_only=False)
        sd = {k.removeprefix("module."): v for k, v in ck["model"].items()}
        m.load_state_dict(sd, strict=True)
    else:
        m.init_parameters()
    return m.to(dev).eval(), cfg

@torch.no_grad()
def embed(m, x):
    """x: float [n, s, 64, 44] in 0..1 -> embeddings [n, 256, 16]"""
    return m(([x], None, None, None, None))["inference_feat"]["embeddings"]
