import numpy as np, torch
from PIL import Image

NAME = "openai/clip-vit-base-patch32"  # 512-d, matches vector_db.DIMS
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_m = _p = None

def _load():
    global _m, _p
    if _m is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            _m = CLIPModel.from_pretrained(NAME, local_files_only=True).to(DEV).eval()
            _p = CLIPProcessor.from_pretrained(NAME, local_files_only=True)
        except Exception:
            try:
                import open_clip
                _m, _, _p = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEV)
                _m.eval()
            except Exception:
                from transformers import CLIPModel, CLIPProcessor
                _m = CLIPModel.from_pretrained(NAME).to(DEV).eval()
                _p = CLIPProcessor.from_pretrained(NAME)
    return _m, _p

@torch.no_grad()
def embed_images(paths, bs=32):
    m, p = _load(); out = []
    if hasattr(m, "encode_image"):
        for i in range(0, len(paths), bs):
            ims = torch.stack([p(Image.open(x).convert("RGB")) for x in paths[i:i + bs]]).to(DEV)
            f = m.encode_image(ims)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    else:
        for i in range(0, len(paths), bs):
            ims = [Image.open(x).convert("RGB") for x in paths[i:i + bs]]
            f = m.visual_projection(m.vision_model(**p(images=ims, return_tensors="pt").to(DEV)).pooler_output)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def embed_text(texts):
    m, p = _load()
    if hasattr(m, "encode_text"):
        import open_clip
        tok = open_clip.get_tokenizer("ViT-B-32")
        t = tok(list(texts)).to(DEV)
        f = m.encode_text(t)
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
    else:
        f = m.text_projection(m.text_model(**p(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEV)).pooler_output)
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
