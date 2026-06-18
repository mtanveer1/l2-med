import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen2VLForConditionalGeneration,
)
from transformers import CLIPProcessor
from typing import Any, Dict

from qwen_vl_utils import process_vision_info


class DLCLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        clip_model,
        clip_processor,
        image_features,
        tokenizer,
        window_size: int = 8,
        penalty_scale: float = 1.0,
        top_k: int = 50,
        token_context_weight: float = 0.5,
        model_type: str = "qwen2vl",
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.penalty_scale = penalty_scale
        self.top_k = top_k
        self.token_context_weight = token_context_weight
        self.model_type = model_type
        self.stride = max(1, int(stride))

        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.image_features = F.normalize(self._to_feat_tensor(image_features), p=2, dim=-1)
        self.tokenizer = tokenizer

        self.similarity_buffer: List[float] = []
        self.buffer_warmup = 3

    @staticmethod
    def _to_feat_tensor(feat_output: torch.Tensor) -> torch.Tensor:
        if isinstance(feat_output, torch.Tensor):
            return feat_output
        if hasattr(feat_output, "image_embeds"):
            return feat_output.image_embeds
        if hasattr(feat_output, "text_embeds"):
            return feat_output.text_embeds
        if hasattr(feat_output, "pooler_output"):
            return feat_output.pooler_output
        if hasattr(feat_output, "last_hidden_state"):
            return feat_output.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported feature output type for normalization")

    def _compute_relative_sim(self, current_sim: float) -> float:
        if len(self.similarity_buffer) >= self.buffer_warmup:
            return float(torch.mean(torch.tensor(self.similarity_buffer)))
        return current_sim

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        current_len = input_ids.shape[1]

        if self.model_type == "llava":
            return scores

        if current_len < self.window_size:
            return scores

        # Apply stride: only recalibrate every N tokens to speed up
        if (current_len % self.stride) != 0:
            return scores

        start_pos = max(0, current_len - self.window_size)
        recent_ids = input_ids[0, start_pos:current_len]
        recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True).strip()

        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        token_strs = [self.tokenizer.decode([tid]) for tid in topk_indices[0]]

        valid_tokens = []
        valid_indices = []
        for idx, token in enumerate(token_strs):
            if not token.strip():
                continue
            valid_tokens.append(token)
            valid_indices.append(idx)

        if not valid_indices:
            return scores

        all_texts: List[str] = []
        all_texts.append(recent_text if recent_text else "none")
        context_token_texts = [recent_text + token for token in valid_tokens]
        all_texts.extend(context_token_texts)
        all_texts.extend(valid_tokens)

        with torch.no_grad():
            max_len = 77
            if hasattr(self.clip_processor, "tokenizer") and hasattr(self.clip_processor.tokenizer, "model_max_length"):
                try:
                    max_len = int(self.clip_processor.tokenizer.model_max_length)
                except Exception:
                    max_len = 77

            # Encode all token-only features each step (no cache)
            tok_inputs = self.clip_processor(
                text=valid_tokens,
                padding=True,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            ).to(self.clip_model.device)
            tok_feats = self.clip_model.get_text_features(**tok_inputs)
            tok_feats = self._to_feat_tensor(tok_feats)
            tok_feats = F.normalize(tok_feats, p=2, dim=-1)

            # Encode recent text and context-texts
            base_inputs = self.clip_processor(
                text=[all_texts[0]] + [recent_text + t for t in valid_tokens],
                padding=True,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            ).to(self.clip_model.device)
            base_text_features = self.clip_model.get_text_features(**base_inputs)
            base_text_features = self._to_feat_tensor(base_text_features)
            base_text_features = F.normalize(base_text_features, p=2, dim=-1)

            text_features = base_text_features[0:1]
            context_token_features = base_text_features[1:]
            token_features = tok_feats

            s_t = (self.image_features @ text_features.T).item()
            self.similarity_buffer.append(s_t)

            if len(self.similarity_buffer) <= self.buffer_warmup:
                rel_s_t = s_t
            else:
                rel_s_t = self._compute_relative_sim(s_t)

            base_sim = rel_s_t

            sim_context = (context_token_features @ self.image_features.T).squeeze()
            sim_token = (token_features @ self.image_features.T).squeeze()

        w = self.token_context_weight
        sim_v = w * sim_context + (1 - w) * sim_token

        dynamic_lambda = self.penalty_scale * (1.0 - base_sim) ** 2

        relative_sim = (sim_v - base_sim) / (1 - base_sim + 1e-6)
        penalties = torch.sigmoid(relative_sim)

        full_adjusted_scores = topk_scores[0].clone()
        for i, (idx, penalty) in enumerate(zip(valid_indices, penalties)):
            full_adjusted_scores[idx] = topk_scores[0][idx] * torch.exp(dynamic_lambda * penalty)

        for idx, score in enumerate(full_adjusted_scores):
            scores[0, topk_indices[0, idx]] = score

        return scores


class OpenClipProcessorAdapter:
    def __init__(self, preprocess, tokenizer_func, device):
        self.preprocess = preprocess
        self._tokenize = tokenizer_func
        self.device = device
        self.tokenizer = type("TokCfg", (), {"model_max_length": 256})()

    def __call__(self, text=None, images=None, return_tensors: str = "pt", padding=True, max_length: int = 256, truncation=True):
        batch: Dict[str, Any] = {}
        if text is not None:
            if isinstance(text, list):
                toks = self._tokenize(text, context_length=max_length)
            else:
                toks = self._tokenize([text], context_length=max_length)
            batch["input_ids"] = toks
        if images is not None:
            from PIL import Image
            pil_list = []
            if isinstance(images, list):
                for im in images:
                    if isinstance(im, Image.Image):
                        pil_list.append(im)
                    elif isinstance(im, dict) and "image" in im and isinstance(im["image"], Image.Image):
                        pil_list.append(im["image"])
            else:
                pil_list = [images]
            import torch
            pixel_values = torch.stack([self.preprocess(img) for img in pil_list])
            batch["pixel_values"] = pixel_values
        return _DeviceDict(batch, self.device)

    def to(self, device):
        self.device = device
        return self


class _DeviceDict(dict):
    def __init__(self, d: Dict[str, Any], device):
        super().__init__(d)
        self._device = device

    def to(self, device):
        self._device = device
        for k, v in list(self.items()):
            try:
                self[k] = v.to(device)
            except Exception:
                pass
        return self


def load_models(model_path: str, clip_model_path: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    device = next(model.parameters()).device
    try:
        clip_model = AutoModel.from_pretrained(
            clip_model_path,
            trust_remote_code=True,
        ).to(device)
        try:
            clip_processor = AutoProcessor.from_pretrained(clip_model_path, trust_remote_code=True)
        except Exception:
            try:
                clip_processor = AutoProcessor.from_pretrained(
                    clip_model_path, trust_remote_code=True, use_fast=False
                )
            except Exception:
                clip_processor = CLIPProcessor.from_pretrained(clip_model_path, trust_remote_code=True)
    except Exception:
        import open_clip
        from transformers import AutoTokenizer
        hub_id = clip_model_path if clip_model_path.startswith("hf-hub:") else f"hf-hub:{clip_model_path}"
        repo_id = hub_id.split(":", 1)[1] if hub_id.startswith("hf-hub:") else hub_id
        model_oc, preprocess = open_clip.create_model_from_pretrained(hub_id)
        hf_tok = AutoTokenizer.from_pretrained(repo_id)

        def tokenizer_func(texts, context_length=256):
            enc = hf_tok(
                texts,
                padding="max_length",
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            return enc.input_ids
        model_oc.to(device)

        class OpenClipWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.device = device

            @torch.no_grad()
            def get_image_features(self, pixel_values=None, **kwargs):
                return self.m.encode_image(pixel_values)

            @torch.no_grad()
            def get_text_features(self, input_ids=None, **kwargs):
                return self.m.encode_text(input_ids)

        clip_model = OpenClipWrapper(model_oc)
        clip_processor = OpenClipProcessorAdapter(preprocess, tokenizer_func, device)

    return model, processor, clip_model, clip_processor


def prepare_message(item, template: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{item['image']}"},
                {"type": "text", "text": template.format(Question=item["problem"])}
            ],
        }
    ]


def run_raw(model, processor, messages, max_new_tokens: int, do_sample: bool) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    generated_ids = model.generate(
        **inputs,
        use_cache=False,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text[0]


def run_raw_batch(model, processor, batch_messages, max_new_tokens: int, do_sample: bool) -> List[str]:
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, _ = process_vision_info(batch_messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    generated_ids = model.generate(
        **inputs,
        use_cache=False,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    outputs = []
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
        trimmed = out_ids[len(in_ids):]
        text = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs.append(text)
    return outputs


def run_dlc(
    model,
    processor,
    clip_model,
    clip_processor,
    messages,
    raw_image,
    max_new_tokens: int,
    window_size: int,
    penalty_scale: float,
    visual_top_k: int,
    do_sample: bool,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        clip_inputs = clip_processor(images=raw_image, return_tensors="pt").to(model_device)
        image_features = clip_model.get_image_features(**clip_inputs)

    logits_processor = LogitsProcessorList(
        [
            DLCLogitsProcessor(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_features=image_features,
                tokenizer=processor.tokenizer,
                window_size=window_size,
                penalty_scale=penalty_scale,
                top_k=visual_top_k,
                model_type="qwen2vl",
            )
        ]
    )

    generated_ids = model.generate(
        **inputs,
        use_cache=False,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        logits_processor=logits_processor,
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs DLC inference with ground-truth answers")
    parser.add_argument("--model-path", default="Med-R1-VQA_Anatomy_Identification/VQA_Anatomy_Identification")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt-path", default="test.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--visual-top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", default="dlc_compare.jsonl")
    args = parser.parse_args()

    model, processor, clip_model, clip_processor = load_models(args.model_path, args.clip_model_path)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_template = (
        "{Question} First output the thinking process in <think> </think> "
        "and final choice (A, B, C, D ...) in <answer> </answer> tags."
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = len(data)
    bsz = args.batch_size
    with open(args.output, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, total, bsz), desc="Compare raw vs DLC"):
            batch_items = data[start : start + bsz]
            batch_messages = [prepare_message(item, question_template) for item in batch_items]

            # Batched raw generation
            raw_outs = run_raw_batch(
                model=model,
                processor=processor,
                batch_messages=batch_messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
            )

            for offset, (item, raw_out) in enumerate(zip(batch_items, raw_outs)):
                idx = start + offset + 1
                raw_image = Image.open(item["image"]).convert("RGB")

                dlc_out = run_dlc(
                    model=model,
                    processor=processor,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    messages=batch_messages[offset],
                    raw_image=raw_image,
                    max_new_tokens=args.max_new_tokens,
                    window_size=args.window_size,
                    penalty_scale=args.penalty_scale,
                    visual_top_k=args.visual_top_k,
                    do_sample=args.sample,
                )

                changed = raw_out != dlc_out

                record = {
                    "image": item["image"],
                    "problem": item["problem"],
                    "solution": item.get("solution"),
                    "raw_output": raw_out,
                    "dlc_output": dlc_out,
                    "changed": changed,
                }

                fout.write(json.dumps(record) + "\n")
                fout.flush()

                marker = "[DIFF]" if changed else "[SAME]"
                print(f"[{idx}/{total}] {marker} image={item['image']}")
                if item.get("solution"):
                    print(f"  solution: {item['solution']}")
                print("  raw   :", raw_out)
                print("  dlc   :", dlc_out)
                print("")

if __name__ == "__main__":
    main()
