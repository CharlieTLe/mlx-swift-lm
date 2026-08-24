"""Dump Muse-Glimmer reference intermediates for the Swift port's parity tests.

Runs the Python MLX reference (mlx_vlm.models.muse_glimmer) on a fixed prompt and
image, writing every checkpoint the Swift test compares against to a single
safetensors file, plus a JSON manifest with the greedy token sequence.

The dumped `pixel_values` are the *post-resize* tensor. The Swift test loads that
rather than re-deriving it from the JPEG, so a CoreImage-vs-PIL resampling
difference cannot be mistaken for a model bug; the resize is checked separately.

Usage:  python muse_parity_dump.py <image> <out.safetensors> <out.json>
"""

import json
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.muse_glimmer.vision import gelu

REPO = "mlx-community/Muse-Glimmer-30B-4bit"
PROMPT = "Describe this image."
MAX_TOKENS = 64


def main(image_path, tensors_path, manifest_path):
    model, processor = load(REPO)
    config = model.config

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], text=text)

    pixel_values = mx.array(inputs["pixel_values"])
    grid_thw = mx.array(inputs["image_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    if input_ids.ndim == 1:
        input_ids = input_ids[None]

    out = {
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw.astype(mx.int32),
        "input_ids": input_ids.astype(mx.int32),
    }

    # ---- vision tower, stage by stage -------------------------------------
    vt = model.vision_tower
    dtype = vt.patch_embedder.patch_embedding.weight.dtype
    px = pixel_values.astype(dtype)
    grid = grid_thw.tolist()

    from mlx_vlm.models.muse_glimmer.vision import (
        _cu_seqlens,
        _position_ids,
        _window_index,
    )

    full_split = _cu_seqlens(grid)[1:-1]
    window_index, window_cu = _window_index(grid, vt.config.pos_emb_height)
    window_split = window_cu[1:-1]

    h = vt.ln_pre(vt.patch_embedder(px, grid))
    out["vision_ln_pre"] = h
    positions = _position_ids(grid)
    if window_index is not None:
        h = h[window_index]
        positions = positions[window_index]
        out["window_index"] = window_index.astype(mx.int32)
    cos, sin = vt._rotary(positions)
    out["vision_rope_cos"] = cos
    out["vision_rope_sin"] = sin

    for idx, block in enumerate(vt.layers):
        split = full_split if vt.config.layer_types[idx] == "full_attention" else window_split
        h = block(h, split, cos, sin)
        # Layers 0-3 cover one full window/full cycle; 49 is the last layer.
        if idx in (0, 1, 2, 3, 49):
            out[f"vision_layer_{idx}"] = h

    if window_index is not None:
        h = h[mx.argsort(window_index)]
    h = vt.ln_post(h)
    out["vision_ln_post"] = h
    merged = vt._pixel_shuffle(h, grid)
    out["vision_merged"] = merged

    # ---- bridge -----------------------------------------------------------
    adapted = model.vision_adapter(merged)
    out["vision_adapter"] = adapted
    projected = model.vision_projection(adapted)
    out["vision_projection"] = projected
    features = model.perception_emb_norm(projected)
    out["image_features"] = features

    # ---- text stack -------------------------------------------------------
    lm = model.language_model.model
    embeds = lm.embed_norm(lm.embed_tokens(input_ids))
    out["embed_norm"] = embeds

    spliced = model.get_input_embeddings(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw
    ).inputs_embeds
    out["inputs_embeds"] = spliced

    from mlx_vlm.models.muse_glimmer.language import create_attention_mask

    hs = spliced
    cache = [None] * len(lm.layers)
    full_mask = create_attention_mask(hs, cache[lm.full_attention_idx])
    sliding_mask = None
    if lm.sliding_attention_idx is not None:
        sliding_mask = create_attention_mask(
            hs, cache[lm.sliding_attention_idx], window_size=lm.sliding_window
        )
    for idx, layer in enumerate(lm.layers):
        mask = sliding_mask if layer.is_sliding else full_mask
        hs = layer(hs, mask=mask, cache=None)
        # 0-2 sliding, 3 is the first full/NoPE layer, 51 is the last.
        if idx in (0, 1, 2, 3, 4, 7, 51):
            out[f"text_layer_{idx}"] = hs
    hs = lm.norm(hs)
    out["text_norm"] = hs

    raw_logits = model.language_model.lm_head(hs)
    out["logits_raw"] = raw_logits
    scaled = raw_logits * model.language_model.output_multiplier
    out["logits_scaled"] = scaled
    cap = model.language_model.final_logit_softcapping
    out["logits"] = mx.tanh(scaled / cap) * cap

    # ---- greedy decode ----------------------------------------------------
    # A manual loop rather than mlx_vlm.generate, so the comparison is on exact
    # token ids instead of re-encoded text.
    cache = model.make_cache()
    first = model(
        input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, cache=cache
    ).logits
    eos = config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    token = int(mx.argmax(first[0, -1]).item())
    greedy_ids = [token]
    for _ in range(MAX_TOKENS - 1):
        if token in eos:
            break
        step = model.language_model(mx.array([[token]]), cache=cache).logits
        token = int(mx.argmax(step[0, -1]).item())
        greedy_ids.append(token)

    tokenizer = processor.tokenizer
    text_out = tokenizer.decode(greedy_ids)

    mx.eval(list(out.values()))
    mx.save_safetensors(
        tensors_path, {k: v.astype(mx.float32) if v.dtype != mx.int32 else v
                       for k, v in out.items()}
    )

    manifest = {
        "repo": REPO,
        "prompt": PROMPT,
        "image": image_path,
        "chat_text": text,
        "input_ids": input_ids.tolist()[0],
        "image_grid_thw": grid_thw.tolist(),
        "generated_text": text_out,
        "generated_ids": list(map(int, greedy_ids)),
        "top1": int(mx.argmax(out["logits"][0, -1]).item()),
        "top5": [int(i) for i in np.argsort(
            -np.array(out["logits"][0, -1].astype(mx.float32))
        )[:5]],
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in out.items()},
        "layer_types_text": config.text_config.layer_types,
        "layer_types_vision": config.vision_config.layer_types,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print("wrote", tensors_path, manifest_path)
    print("top1", manifest["top1"], "top5", manifest["top5"])
    print("generated:", repr(text_out[:300]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
