# ============================================================
# XAI-HDetect FastAPI Backend
# ============================================================

import os
import io
import re
import gc
import math
import base64
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import joblib
from PIL import Image
from collections import Counter
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Download models from HF Hub at startup ────────────────────
from huggingface_hub import hf_hub_download

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
REPO_ID    = "ayeshaissadeen/xai-hdetect-models"
MODELS_DIR = "/tmp/xai_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_model_path(filename):
    local = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(local):
        print(f"Downloading {filename} from HF Hub...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
            token=HF_TOKEN,
        )
    return local

# ── Imports from transformers ─────────────────────────────────
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="XAI-HDetect API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline (loaded once at startup) ──────────────────
pipeline = None

# ── Helper functions ──────────────────────────────────────────
SPECIAL_TOKENS = {"</s>", "<s>", "[PAD]", "[pad]"}
HEX_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

SKIP_WORDS = {
    "", ".", ",", "!", "?", ";", ":", "-", "—", "(", ")", "[", "]",
    "the", "a", "an", "and", "or", "but", "at", "for", "of", "by",
    "from", "up", "about", "into", "through", "during", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may",
    "might", "it", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they"
}

REL_CUE_WORDS = {
    "in", "on", "under", "over", "between", "behind", "beside", "next",
    "to", "left", "right", "front", "of", "above", "below", "with",
    "holding", "near"
}

def clean_token(tok):
    return str(tok).replace("▁", "").strip().lower()

def should_skip_token(tok_clean):
    if tok_clean in SPECIAL_TOKENS: return True
    if HEX_RE.match(tok_clean): return True
    if len(tok_clean) <= 1: return True
    if tok_clean in REL_CUE_WORDS: return False
    return tok_clean in SKIP_WORDS

def is_bad_subword_piece(raw_tok):
    t_raw = str(raw_tok).strip()
    if t_raw.startswith("<0x") or t_raw in {"0A", "<0x0A>"}: return True
    bare = t_raw.replace("▁", "").strip().lower()
    if bare in SPECIAL_TOKENS: return True
    if HEX_RE.match(bare): return True
    if len(bare) <= 1: return True
    if bare.isalpha() and not t_raw.startswith("▁") and len(bare) <= 3: return True
    return False

def attn_summary_features(attn_row):
    if attn_row is None or attn_row.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    a = attn_row.astype(np.float32)
    s = float(a.sum())
    p = a / s if s > 0 else np.full_like(a, 1.0 / max(len(a), 1))
    p = np.clip(p, 1e-12, 1.0)
    ent = float(-(p * np.log(p)).sum())
    return [float(a.mean()), float(a.max()), ent, float(a.var())]

def build_word_spans(tokens):
    toks = list(tokens)
    spans = []
    start = 0
    parts = []
    for i, raw in enumerate(toks):
        is_word_start = str(raw).startswith("▁") or i == 0
        if is_word_start and i > 0:
            spans.append((start, i, "".join(parts)))
            start = i
            parts = [clean_token(raw)]
        else:
            parts.append(clean_token(raw))
    if parts:
        spans.append((start, len(toks), "".join(parts)))
    return spans

def patch_indices_to_box(indices, grid_size=24, image_size=336, min_box_size=32):
    if not indices:
        return (0, 0, image_size, image_size)
    rows = [idx // grid_size for idx in indices]
    cols = [idx % grid_size for idx in indices]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    pw, ph = image_size / grid_size, image_size / grid_size
    x0 = int(math.floor(c0 * pw))
    y0 = int(math.floor(r0 * ph))
    x1 = int(math.ceil((c1 + 1) * pw))
    y1 = int(math.ceil((r1 + 1) * ph))
    x0 = max(0, min(x0, image_size - 1))
    y0 = max(0, min(y0, image_size - 1))
    x1 = max(x0 + 1, min(x1, image_size))
    y1 = max(y0 + 1, min(y1, image_size))
    return (x0, y0, x1, y1)

def build_crop_from_attention(attn_vec, image, topk=16):
    a = np.asarray(attn_vec, dtype=np.float32)
    if a.ndim != 1 or a.size != 576:
        return image.copy()
    s = float(a.sum())
    p = a / s if s > 0 else np.full_like(a, 1.0 / len(a))
    top_idx = np.argsort(p)[-topk:]
    box = patch_indices_to_box(top_idx.tolist())
    return image.crop(box)

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_to_overlay(attn_map_flat, image_pil, label=None):
    attn = attn_map_flat.reshape(24, 24)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heat = cv2.resize(attn, (336, 336), interpolation=cv2.INTER_CUBIC)
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    img = np.array(image_pil.resize((336, 336)))[:, :, ::-1]
    overlay = cv2.addWeighted(img, 0.55, heat_color, 0.45, 0)
    if label:
        cv2.putText(overlay, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(overlay[:, :, ::-1])

# ── Span grouping ─────────────────────────────────────────────
SPAN_EDGE_SKIP = {
    "", ".", ",", "!", "?", ";", ":", "-", "—", "(", ")", "[", "]",
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on",
    "at", "for", "with", "as", "by", "from", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "there", "here"
}

def detokenize_span_tokens(raw_tokens):
    text = "".join(str(t) for t in raw_tokens)
    text = text.replace("▁", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

def is_valid_clean_span(tokens, indices):
    if not indices: return False
    words = []
    for i in indices:
        tc = clean_token(tokens[i])
        if tc and tc not in SPECIAL_TOKENS and not HEX_RE.match(tc):
            words.append(tc)
    if not words: return False
    meaningful = [w for w in words if len(w) > 2 and w not in SPAN_EDGE_SKIP]
    return bool(meaningful)

def group_hallucinated_spans(tokens, hallucinated_indices):
    if not hallucinated_indices: return []
    spans, current = [], []
    for i in sorted(set(hallucinated_indices)):
        if not current:
            current = [i]
        elif i == current[-1] + 1:
            current.append(i)
        else:
            spans.append(current)
            current = [i]
    if current:
        spans.append(current)

    span_objects = []
    for span in spans:
        idxs = list(span)
        while idxs:
            lc = clean_token(tokens[idxs[0]])
            if lc in SPAN_EDGE_SKIP or is_bad_subword_piece(tokens[idxs[0]]):
                idxs.pop(0)
            else:
                break
        while idxs:
            rc = clean_token(tokens[idxs[-1]])
            if rc in SPAN_EDGE_SKIP or is_bad_subword_piece(tokens[idxs[-1]]):
                idxs.pop()
            else:
                break
        if not is_valid_clean_span(tokens, idxs): continue
        span_text = detokenize_span_tokens([tokens[i] for i in idxs])
        if span_text:
            span_objects.append({"text": span_text, "indices": idxs})
    return span_objects


# ── Span-level readable explanation builder ───────────────────
_REL_W   = {"left","right","near","next","behind","front","between","along",
            "towards","beside","above","below","under","over","inside",
            "outside","on","in","located"}
_SCENE_W = {"street","road","park","room","kitchen","office","sidewalk",
            "beach","field","building","city"}
_NUM_W   = {"one","two","three","four","five","six","seven","eight","nine",
            "ten","several","some","many","few"}
_ATTR_W  = {"standing","walking","running","located","parked","wearing",
            "holding","closer","farther"}

def build_span_explanation(span_text, mean_p, explanations, span_indices):
    s_lower = span_text.lower()
    words   = re.findall(r"[a-z0-9\']+", s_lower)

    if any(w in _REL_W for w in words):
        semantic_type = "relationship"
    elif any(w in _SCENE_W for w in words):
        semantic_type = "scene"
    elif any(w in _NUM_W or w.isdigit() for w in words):
        semantic_type = "object"
    elif any(w in _ATTR_W for w in words):
        semantic_type = "attribute"
    else:
        semantic_type = "general"

    content_reason = {
        "object":       "The span makes a concrete object or count claim.",
        "relationship": "The span expresses spatial or relational information.",
        "scene":        "The span adds scene-level context.",
        "attribute":    "The span adds a descriptive detail or modifier.",
        "general":      "The span adds content that may not be visually supported.",
    }[semantic_type]

    if mean_p >= 0.80:
        conf_phrase = "with strong confidence"
    elif mean_p >= 0.65:
        conf_phrase = "with moderate confidence"
    else:
        conf_phrase = "with relatively weak confidence"

    span_exps    = [e for e in explanations if e["token_index"] in span_indices]
    driver_names = set()
    for ex in span_exps:
        for name, _ in (ex.get("top_shap") or [])[:4]:
            driver_names.add(name)

    lang_sig   = any(x in driver_names for x in ["llava_embedding_total","logprob","log_prob"])
    vis_sig    = any(x in driver_names for x in ["clip_score","clip_region_score",
                    "clip_region_minus_global","attn_mean","attn_max","attn_entropy","attn_var"])
    attn_unc   = "attn_entropy" in driver_names
    weak_local = "clip_region_minus_global" in driver_names

    if lang_sig and vis_sig:
        det_reason = "The detector found mismatch between language patterns and visual grounding."
    elif lang_sig:
        det_reason = "The detector suggests the model relied more on language patterns than visual evidence."
    elif vis_sig:
        det_reason = "The detector suggests weak visual grounding."
    else:
        det_reason = "The detector found weak support for this span."

    extra = []
    if attn_unc:
        extra.append("attention was diffuse")
    if weak_local:
        extra.append("local region grounding was weak")

    readable = (
        f"Span \'{span_text}\' is identified as a hallucination candidate {conf_phrase} "
        f"(mean score={mean_p:.3f}). "
        f"{content_reason} {det_reason}"
    )
    if extra:
        readable += " In particular, " + "; ".join(extra[:2]) + "."
    return readable

# ── Taxonomy classifier wrapper ───────────────────────────────
class TaxonomyClassifierSentenceTransformer:
    def __init__(self, model_path: str, meta_path: str, rel_tau: float = 0.20):
        from sentence_transformers import SentenceTransformer

        self.clf  = joblib.load(model_path)
        self.meta = joblib.load(meta_path)

        self.classes_ = list(getattr(self.clf, "classes_", [])) or self.meta.get("classes")
        if not self.classes_:
            raise ValueError("Could not load taxonomy classes from model/meta.")

        self.rel_tau = float(rel_tau)

        self.model_name = self.meta.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.sent_model = SentenceTransformer(self.model_name)

        if "relationship" not in self.classes_:
            raise ValueError(f"'relationship' not in classes: {self.classes_}")

    @staticmethod
    def make_context(caption: str, span_text: str) -> str:
        if span_text and span_text in caption:
            return caption.replace(span_text, f" [TOK] {span_text} [/TOK] ", 1)
        return f"{caption} [TOK] {span_text} [/TOK]"

    def embed_context(self, context: str) -> np.ndarray:
        emb = self.sent_model.encode(
            [context],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        return emb

    def predict_from_context(self, context: str):
        emb   = self.embed_context(context)
        proba = self.clf.predict_proba(emb)[0]
        probs = dict(zip(self.classes_, proba))

        if probs.get("relationship", 0.0) >= self.rel_tau:
            pred = "relationship"
        else:
            pred = self.classes_[int(np.argmax(proba))]

        return pred, probs

    def predict(self, caption: str, span_text: str):
        context = self.make_context(caption, span_text)
        pred, probs = self.predict_from_context(context)
        return pred, probs, context

# ── SHAP explainer wrapper ────────────────────────────────────
class SHAPWrapper:
    def __init__(self, lgbm_model):
        import shap
        self.booster  = lgbm_model.booster_
        self.explainer = shap.TreeExplainer(self.booster)

    def explain_one(self, x_row):
        sv = self.explainer.shap_values(x_row.reshape(1, -1))
        shap_row = sv[1][0] if isinstance(sv, list) else sv[0]
        groups = {
            "entropy":               float(shap_row[0]),
            "logprob":               float(shap_row[1]),
            "clip_score":            float(shap_row[2]),
            "clip_region_score":     float(shap_row[3]),
            "clip_region_minus_global": float(shap_row[4]),
            "llava_embedding_total": float(np.sum(shap_row[5:4101])),
            "attn_mean":             float(shap_row[4101]),
            "attn_max":              float(shap_row[4102]),
            "attn_entropy":          float(shap_row[4103]),
            "attn_var":              float(shap_row[4104]),
        }
        top = sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        return top

# ── Grad-CAM ──────────────────────────────────────────────────
GRADCAM_SKIP = {
    "", ".", ",", "the", "a", "an", "and", "or", "of", "to", "in",
    "on", "is", "are", "was", "were", "be", "with", "by", "for",
    "one", "some", "several", "many", "few", "other", "while"
}

def get_vit_layers(llava_model):
    for path in [
        ["vision_tower", "vision_model", "encoder", "layers"],
        ["model", "vision_tower", "vision_model", "encoder", "layers"],
        ["vision_tower", "vision_tower", "vision_model", "encoder", "layers"],
        ["model", "vision_tower", "vision_tower", "vision_model", "encoder", "layers"],
    ]:
        obj, ok = llava_model, True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok: return obj
    raise AttributeError("Could not find ViT encoder layers")

def compute_gradcam(llava_model, processor, device, image_pil, prompt,
                    full_ids, gen_ids, token_idx, prompt_len, target_layer=-4):
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["vit"] = out[0] if isinstance(out, (tuple, list)) else out

    def bwd_hook(m, gi, go):
        gradients["vit"] = go[0] if isinstance(go, (tuple, list)) else go

    layers = get_vit_layers(llava_model)
    layer  = layers[target_layer]
    fh = layer.register_forward_hook(fwd_hook)
    bh = layer.register_full_backward_hook(bwd_hook)

    try:
        image_pil_resized = image_pil.convert("RGB").resize((336, 336))
        conv = processor(images=image_pil_resized, text=prompt, return_tensors="pt").to(device)

        token_id  = int(gen_ids[token_idx].item())
        abs_pos   = prompt_len + token_idx

        llava_model.zero_grad(set_to_none=True)
        out = llava_model(
            input_ids=full_ids.unsqueeze(0),
            pixel_values=conv.pixel_values,
            use_cache=False, return_dict=True,
        )
        score = out.logits[0, abs_pos - 1, token_id].float()
        del out
        torch.cuda.empty_cache()
        score.backward()

        acts  = activations["vit"][0][1:, :]
        grads = gradients["vit"][0][1:, :]
        if acts.shape[0] != 576:
            return None

        weights = grads.mean(dim=-1)
        cam = F.relu((weights.unsqueeze(-1) * acts).sum(dim=-1))
        cam = cam.reshape(24, 24).detach().float().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    finally:
        fh.remove()
        bh.remove()
        torch.cuda.empty_cache()

def overlay_gradcam(image_pil, cam, alpha=0.45):
    img = image_pil.convert("RGB").resize((336, 336))
    heat = cv2.resize(cam, (336, 336), interpolation=cv2.INTER_CUBIC)
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    arr = np.array(img)[:, :, ::-1]
    overlay = cv2.addWeighted(arr, 1 - alpha, heat_color, alpha, 0)
    return Image.fromarray(overlay[:, :, ::-1])

# ── Main pipeline run ─────────────────────────────────────────

def run_pipeline(image_pil, question):
    global pipeline
    p = pipeline

    image_pil = image_pil.convert("RGB")
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}]
    }]
    prompt = p["processor"].apply_chat_template(conversation, add_generation_prompt=True)
    inputs = p["processor"](images=image_pil, text=prompt, return_tensors="pt").to(p["device"])

    with torch.no_grad():
        gen = p["llava"].generate(
            **inputs,
            max_new_tokens=200,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    prompt_len = len(inputs.input_ids[0])
    full_ids   = gen.sequences[0]
    gen_ids    = full_ids[prompt_len:]
    caption    = p["processor"].decode(gen_ids, skip_special_tokens=True)
    tokens     = p["processor"].tokenizer.convert_ids_to_tokens(gen_ids)
    T          = len(tokens)
    eps        = 1e-10

    # Language-prior baseline
    try:
        blank = Image.new("RGB", (336, 336), (0, 0, 0))
        bi    = p["processor"](images=blank, text=prompt, return_tensors="pt").to(p["device"])
        bfull = torch.cat([bi.input_ids, gen_ids.unsqueeze(0)], dim=1)
        with torch.no_grad():
            bout = p["llava"](input_ids=bfull, pixel_values=bi.pixel_values, use_cache=False)
        blog = bout.logits[0, prompt_len - 1:-1, :]
        prior_logprobs = np.zeros(T, dtype=np.float32)
        for i in range(T):
            lp = torch.log_softmax(blog[i].float(), dim=-1)[gen_ids[i]].item()
            prior_logprobs[i] = lp if np.isfinite(lp) else -50.0
        del bout, blog, bi, bfull
        torch.cuda.empty_cache()
    except Exception:
        prior_logprobs = np.zeros(T, dtype=np.float32)

    # Entropy + logprob
    token_entropies = np.zeros(T, dtype=np.float32)
    token_logprobs  = np.zeros(T, dtype=np.float32)
    for i in range(T):
        logits = gen.scores[i][0]
        probs  = torch.softmax(logits, dim=-1)
        token_entropies[i] = float(-(probs * torch.log(probs + eps)).sum().item())
        token_logprobs[i]  = float(torch.log_softmax(logits, dim=-1)[gen_ids[i]].item())

    causal_scores = token_logprobs - prior_logprobs

    # CLIP
    all_clean = [clean_token(t) for t in tokens]
    toks_uniq = list(dict.fromkeys(t for t in all_clean if t and t not in SPECIAL_TOKENS))
    if toks_uniq:
        ci = p["clip_processor"](text=toks_uniq, images=image_pil,
                                  return_tensors="pt", padding=True).to(p["clip_device"])
        with torch.no_grad():
            cout = p["clip_model"](**ci)
        te = F.normalize(cout.text_embeds, p=2, dim=-1)
        ie = F.normalize(cout.image_embeds, p=2, dim=-1)
        sims = (te @ ie.T).squeeze(1).cpu().numpy()
        clip_map = {t: float(s) for t, s in zip(toks_uniq, sims)}
    else:
        clip_map = {}

    # Attention
    V = 576
    attn_summ    = np.zeros((T, 4), dtype=np.float32)
    attn_maps_576 = []

    def best_vis_start(row, V):
        if row.shape[0] <= V: return 0
        c = np.cumsum(row, dtype=np.float64)
        ws = c[V-1:] - np.concatenate(([0.0], c[:-V]))
        return int(np.argmax(ws))

    for i in range(T):
        img_attn = np.zeros(V, dtype=np.float32)
        try:
            a_last = gen.attentions[i][-1]
            a_avg  = a_last.mean(dim=1)[0]
            row    = a_avg[-1].detach().cpu().numpy().astype(np.float32)
            start  = best_vis_start(row, V)
            slc    = row[start:min(start + V, row.shape[0])]
            if slc.shape[0] < V:
                slc = np.pad(slc, (0, V - slc.shape[0]))
            img_attn = np.maximum(slc[:V], eps)
        except Exception:
            pass
        attn_maps_576.append(img_attn)
        attn_summ[i] = np.array(attn_summary_features(img_attn), dtype=np.float32)

    # Hidden states — free memory immediately after extraction
    with torch.no_grad():
        hout = p["llava"](
            input_ids=full_ids.unsqueeze(0),
            pixel_values=inputs.pixel_values,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
    emb = hout.hidden_states[-1][0][-T:, :].detach().float().cpu().numpy()
    del hout
    torch.cuda.empty_cache()
    gc.collect()

    # Region CLIP
    clip_scores           = np.zeros(T, dtype=np.float32)
    clip_region_scores    = np.zeros(T, dtype=np.float32)
    clip_region_minus     = np.zeros(T, dtype=np.float32)
    image_resized         = image_pil.resize((336, 336))
    word_spans            = build_word_spans(tokens)

    for start, end, wt in word_spans:
        wt = wt.strip().lower()
        if not wt: continue
        gs = float(clip_map.get(wt, 0.0))
        for i in range(start, end):
            clip_scores[i] = gs
        attn_word = np.mean(np.stack(attn_maps_576[start:end], axis=0), axis=0)
        crop = build_crop_from_attention(attn_word, image_resized)
        if should_skip_token(wt):
            rs = 0.0
        else:
            try:
                ci2 = p["clip_processor"](text=[wt], images=crop, return_tensors="pt",
                                           padding=True, truncation=True).to(p["clip_device"])
                with torch.no_grad():
                    co2 = p["clip_model"](**ci2)
                te2 = F.normalize(co2.text_embeds, p=2, dim=-1)
                ie2 = F.normalize(co2.image_embeds, p=2, dim=-1)
                rs = float((te2 @ ie2.T).squeeze().item())
            except Exception:
                rs = 0.0
        crop.close()
        for i in range(start, end):
            clip_region_scores[i] = rs
            clip_region_minus[i]  = rs - gs

    # Build X
    X = np.concatenate([
        token_entropies.reshape(-1, 1),
        token_logprobs.reshape(-1, 1),
        clip_scores.reshape(-1, 1),
        clip_region_scores.reshape(-1, 1),
        clip_region_minus.reshape(-1, 1),
        emb,
        attn_summ,
    ], axis=1).astype(np.float32)

    # Detect
    thr     = 0.65
    raw_p   = p["detector"].predict_proba(X)[:, 1].astype(float)
    iso     = p["detector_payload"].get("isotonic_calibrator", None)
    tprobs  = iso.predict(raw_p).astype(float) if iso is not None else raw_p

    grounding    = (clip_region_scores + np.clip(clip_region_minus, 0, 1)) / 2
    adaptive_thr = np.where(grounding < 0.15, thr * 0.80,
                   np.where(grounding > 0.35, thr * 1.20, thr))
    labels       = (tprobs >= adaptive_thr).astype(int)
    halluc_idx   = np.where(labels == 1)[0].tolist()
    halluc_display = [i for i in halluc_idx if not is_bad_subword_piece(tokens[i])]

    # CLIP post-filter
    additional = [
        i for i, tok in enumerate(tokens)
        if i not in halluc_idx
        and clean_token(tok) in clip_map
        and clip_map[clean_token(tok)] < 0.165
        and not should_skip_token(clean_token(tok))
    ]
    if additional:
        halluc_idx     = sorted(set(halluc_idx + additional))
        halluc_display = sorted(set(halluc_display + [
            i for i in additional if not is_bad_subword_piece(tokens[i])
        ]))

    # ── SHAP ──────────────────────────────────────────────────
    explanations = []
    if halluc_display and p.get("shap_explainer"):
        X_hall = X[halluc_display]
        for row_i, tok_idx in enumerate(halluc_display):
            top = p["shap_explainer"].explain_one(X_hall[row_i])
            explanations.append({
                "token_index":    tok_idx,
                "token":          tokens[tok_idx],
                "p_hallucinated": round(float(tprobs[tok_idx]), 4),
                "top_shap":       [(k, round(float(v), 4)) for k, v in top],
            })

    # ── Taxonomy + span-level readable explanations ───────────
    hall_spans    = group_hallucinated_spans(tokens, halluc_display)
    taxonomy_spans = []

    REL_SPANS   = {"to the right of","to the left of","in front of","next to",
                   "behind","under","above","on top of","beside","between"}
    SCENE_SPANS = {"indoors","outdoors","inside","outside","kitchen","bathroom",
                   "office","harbor","restaurant","bedroom"}

    for sp in hall_spans:
        st = sp["text"].strip()
        if not st: continue
        s_low = st.lower()
        if s_low in REL_SPANS:
            tax = "relationship"
        elif s_low in SCENE_SPANS:
            tax = "scene"
        else:
            mean_p = float(np.mean([tprobs[i] for i in sp["indices"]]))
            tag    = ("[HIGH-CONFIDENCE-HALLUCINATION]" if mean_p > 0.75
                      else "[MEDIUM-CONFIDENCE-HALLUCINATION]" if mean_p > 0.50
                      else "[LOW-CONFIDENCE-HALLUCINATION]")
            aug    = f"{tag} {caption}"
            tax, tax_scores, _ = p["taxonomy"].predict(aug, st)
        mean_p         = float(np.mean([tprobs[i] for i in sp["indices"]]))
        span_causal    = float(np.mean([causal_scores[i] for i in sp["indices"]
                                        if i < len(causal_scores)]))
        span_grounding = float(np.mean([grounding[i] for i in sp["indices"]
                                        if i < len(grounding)]))
        verdict = ("visual_grounding_failure"    if span_causal > 0.10
                   else "language_prior_dominant" if span_causal < -0.05
                   else "ambiguous")
        readable = build_span_explanation(st, mean_p, explanations, sp["indices"])
        taxonomy_spans.append({
            "text":            st,
            "indices":         sp["indices"],
            "taxonomy":        tax,
            "confidence":      round(mean_p, 4),
            "causal_score":    round(span_causal, 4),
            "grounding_score": round(span_grounding, 4),
            "causal_verdict":  verdict,
            "explanation":     readable,
        })

    # ── Grad-CAM — top token only ──────────────────────────────
    gradcam_b64    = None
    gradcam_token  = None
    gradcam_prob   = None

    candidates = [
        {"idx": i, "prob": float(tprobs[i]), "tok": clean_token(tokens[i])}
        for i in range(len(tokens))
        if float(tprobs[i]) >= 0.55
        and clean_token(tokens[i]) not in GRADCAM_SKIP
        and len(clean_token(tokens[i])) > 1
        and not is_bad_subword_piece(tokens[i])
    ]
    candidates.sort(key=lambda x: x["prob"], reverse=True)

    if candidates:
        best = candidates[0]
        try:
            torch.cuda.empty_cache()
            gc.collect()
            cam = compute_gradcam(
                p["llava"], p["processor"], p["device"],
                image_pil, prompt, full_ids, gen_ids,
                best["idx"], prompt_len, target_layer=-4
            )
            if cam is not None:
                overlay         = overlay_gradcam(image_pil, cam)
                gradcam_b64     = pil_to_base64(overlay)
                gradcam_token   = best["tok"]
                gradcam_prob    = round(best["prob"], 4)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
        finally:
            torch.cuda.empty_cache()

    # ── Detected tokens list ───────────────────────────────────
    detected_tokens = [
        {
            "index": idx,
            "token": clean_token(tokens[idx]),
            "prob":  round(float(tprobs[idx]), 4),
            "entropy": round(float(token_entropies[idx]), 4),
            "causal_score": round(float(causal_scores[idx]), 4),
        }
        for idx in halluc_display
        if float(tprobs[idx]) >= 0.50
    ]

    return {
        "caption":          caption,
        "detected_tokens":  detected_tokens,
        "spans":            taxonomy_spans,
        "explanations":     explanations,
        "gradcam_image":    gradcam_b64,
        "gradcam_token":    gradcam_token,
        "gradcam_prob":     gradcam_prob,
        "model_stats": {
            "auroc":        0.8597,
            "f1":           0.636,
            "ece":          0.0081,
            "calibrated":   True,
        }
    }

# ── Startup: load pipeline ────────────────────────────────────
@app.on_event("startup")
async def load_pipeline():
    global pipeline
    print("Loading pipeline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    llava.eval()
    print("LLaVA loaded")

    clip_device = "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("CLIP loaded")

    det_path  = get_model_path("hallucination_detector_lgbm_token_5000.pkl")
    payload   = joblib.load(det_path)
    detector  = payload["model"]

    tax_path  = get_model_path("taxonomy_lgbm_sentformer.pkl")
    meta_path = get_model_path("taxonomy_meta_sentformer.pkl")
    taxonomy  = TaxonomyClassifierSentenceTransformer(tax_path, meta_path, rel_tau=0.20)
    print("Detector and taxonomy loaded")

    try:
        import shap
        shap_exp = SHAPWrapper(detector)
        print("SHAP loaded")
    except Exception:
        shap_exp = None
        print("SHAP not available")

    pipeline = {
        "device":           device,
        "processor":        processor,
        "llava":            llava,
        "clip_device":      clip_device,
        "clip_model":       clip_model,
        "clip_processor":   clip_proc,
        "detector":         detector,
        "detector_payload": payload,
        "taxonomy":         taxonomy,
        "shap_explainer":   shap_exp,
    }
    print("Pipeline ready")

# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    question: str = Form(default="Describe the image in detail.")
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = run_pipeline(img, question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))