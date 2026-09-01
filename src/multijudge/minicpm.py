from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .adapters import InputMode, JudgeAdapter
from .parsers import parse_canonical_decision
from .policy import CanonicalPolicy, render_policy_prompt
from .recording import build_native_record
from .schema import (
    CanonicalSample,
    JudgeServiceIdentity,
    canonical_json,
    sha256_text,
)
from .soft_scores import (
    BINARY_CODEBOOKS,
    binary_codebook_evidence,
    build_soft_record,
    choice_distribution,
    combine_symmetric_binary_codebooks,
    render_binary_code_prompt,
)


ADAPTER_VERSION = "minicpm-v4.5-binary-judge-v2"
SOFT_ADAPTER_VERSION = "minicpm-v4.5-soft-score-v1"
MAX_SLICE_NUMS = 4
MAX_TOTAL_SLICES = 16


def slices_per_image(image_count: int) -> int:
    if image_count < 1:
        raise ValueError("MiniCPM requires at least one image")
    if image_count > MAX_TOTAL_SLICES:
        raise ValueError(
            f"MiniCPM image count exceeds slice budget: {image_count} > "
            f"{MAX_TOTAL_SLICES}"
        )
    return min(MAX_SLICE_NUMS, MAX_TOTAL_SLICES // image_count)


def build_minicpm_content(
    sample: CanonicalSample,
    policy_prompt: str,
) -> tuple[list[Any], str]:
    content: list[Any] = [
        policy_prompt
        + "\n\nDIALOGUE EVIDENCE follows. Image markers identify the visual "
        "evidence supplied at each turn."
    ]
    semantic: list[dict[str, str]] = [
        {"type": "text", "text": str(content[0])}
    ]
    image_index = 0
    for turn_index, turn in enumerate(sample.turns, start=1):
        indices = list(
            range(image_index + 1, image_index + 1 + len(turn.image_paths))
        )
        markers = " ".join(f"[IMAGE_{value:02d}]" for value in indices)
        header = (
            f"[TURN_{turn_index:02d}][{turn.role.upper()}] {markers}"
        ).rstrip()
        content.append(header)
        semantic.append({"type": "text", "text": header})
        for path in turn.image_paths:
            image_index += 1
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
            content.append(image)
            semantic.append(
                {"type": "image", "image": f"IMAGE_BINARY_{image_index:02d}"}
            )
        text = turn.content.strip()
        content.append(text)
        semantic.append({"type": "text", "text": text})
    return content, sha256_text(canonical_json(semantic))


class MiniCPMJudgeAdapter(JudgeAdapter):
    def __init__(
        self,
        *,
        model_path: Path,
        model_revision: str,
        policy: CanonicalPolicy,
        max_new_tokens: int = 32,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_path = model_path.resolve()
        self.policy = policy
        self.policy_prompt = render_policy_prompt(policy)
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.identity = JudgeServiceIdentity(
            organization="OpenBMB",
            model_id="OpenBMB/MiniCPM-V-4_5",
            model_revision=model_revision,
            policy_sha256=policy.sha256,
            adapter_version=ADAPTER_VERSION,
        )
        self.soft_identity = JudgeServiceIdentity(
            organization="OpenBMB",
            model_id="OpenBMB/MiniCPM-V-4_5",
            model_revision=model_revision,
            policy_sha256=policy.sha256,
            adapter_version=SOFT_ADAPTER_VERSION,
        )
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MiniCPM requires torch and transformers"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM qualification requires a CUDA GPU")
        self._model = (
            AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .cuda()
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self._torch = torch

    def _prepare_soft_inputs(
        self,
        content: list[Any],
        *,
        slice_budget: int,
    ) -> dict[str, Any]:
        assert self._model is not None
        assert self._processor is not None
        assert self._tokenizer is not None
        images = []
        rendered_content = []
        for value in content:
            if isinstance(value, Image.Image):
                images.append(value)
                rendered_content.append("(<image>./</image>)")
            elif isinstance(value, str):
                rendered_content.append(value)
            else:
                raise TypeError(
                    f"Unsupported MiniCPM message content: {type(value)!r}"
                )
        messages = [
            {
                "role": "user",
                "content": "\n".join(rendered_content),
            }
        ]
        prompt = self._processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(
            [prompt],
            [images],
            max_slice_nums=slice_budget,
            return_tensors="pt",
            max_length=16384,
        ).to(self._model.device)
        inputs.pop("image_sizes", None)
        return {
            "inputs": inputs,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "rendered_prompt": prompt,
        }

    def _score_next_token(
        self,
        content: list[Any],
        *,
        slice_budget: int,
        candidates: tuple[str, ...],
        vision_hidden_states: Any | None,
    ) -> dict[str, Any]:
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        prepared = self._prepare_soft_inputs(
            content,
            slice_budget=slice_budget,
        )
        self._torch.cuda.synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated, next_vision_hidden_states = self._model.generate(
                **prepared["inputs"],
                tokenizer=self._tokenizer,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                stream=False,
                decode_text=False,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        self._torch.cuda.synchronize()
        if not generated.scores:
            raise RuntimeError("MiniCPM soft scoring returned no scores")
        return {
            "distribution": choice_distribution(
                generated.scores[0][0],
                self._tokenizer,
                candidates,
            ),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "input_tokens": prepared["input_tokens"],
            "rendered_prompt_sha256": sha256_text(
                prepared["rendered_prompt"]
            ),
            "vision_hidden_states": next_vision_hidden_states,
        }

    def soft_judge(
        self,
        sample: CanonicalSample,
        *,
        input_mode: InputMode,
        panel_dir: Path,
    ) -> dict[str, Any]:
        del panel_dir
        if input_mode != "native":
            raise ValueError("MiniCPM soft scoring supports only native input")
        self.load()
        assert self._torch is not None
        torch = self._torch
        slice_budget = slices_per_image(len(sample.image_paths))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        total_started = time.perf_counter()

        codebook_results = []
        prompt_hashes = []
        input_tokens = []
        vision_hidden_states = None
        for codebook in BINARY_CODEBOOKS:
            prompt = render_binary_code_prompt(
                self.policy,
                safe_code=codebook["safe_code"],
                unsafe_code=codebook["unsafe_code"],
            )
            content, semantic_hash = build_minicpm_content(sample, prompt)
            scored = self._score_next_token(
                content,
                slice_budget=slice_budget,
                candidates=("0", "1"),
                vision_hidden_states=vision_hidden_states,
            )
            vision_hidden_states = scored["vision_hidden_states"]
            codebook_results.append(
                binary_codebook_evidence(
                    scored["distribution"],
                    codebook_id=codebook["codebook_id"],
                    safe_code=codebook["safe_code"],
                    unsafe_code=codebook["unsafe_code"],
                )
            )
            prompt_hashes.append(
                sha256_text(
                    canonical_json(
                        {
                            "semantic": semantic_hash,
                            "rendered": scored["rendered_prompt_sha256"],
                        }
                    )
                )
            )
            input_tokens.append(scored["input_tokens"])

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - total_started) * 1000.0
        return build_soft_record(
            sample=sample,
            identity=self.soft_identity,
            policy=self.policy,
            soft_evidence=combine_symmetric_binary_codebooks(
                codebook_results[0],
                codebook_results[1],
            ),
            prompt_hashes=prompt_hashes,
            latency_ms=latency_ms,
            peak_vram_gib=torch.cuda.max_memory_allocated() / (1024**3),
            input_tokens=input_tokens,
            runtime={
                "dtype": "bfloat16",
                "attn_implementation": self.attn_implementation,
                "do_sample": False,
                "max_new_tokens": 1,
                "candidate_normalization": "candidate_set_softmax",
                "max_slice_nums": MAX_SLICE_NUMS,
                "effective_max_slice_nums": slice_budget,
                "max_total_slices": MAX_TOTAL_SLICES,
                "vision_features_reused_across_codebooks": True,
            },
        )

    def judge(
        self,
        sample: CanonicalSample,
        *,
        input_mode: InputMode,
        panel_dir: Path,
    ) -> dict[str, Any]:
        del panel_dir
        if input_mode != "native":
            raise ValueError("MiniCPM supports only native multi-image input")
        self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._processor is not None
        assert self._model is not None
        torch = self._torch

        content, prompt_sha256 = build_minicpm_content(
            sample,
            self.policy_prompt,
        )
        slice_budget = slices_per_image(len(sample.image_paths))
        messages = [{"role": "user", "content": content}]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            raw_output = self._model.chat(
                msgs=messages,
                tokenizer=self._tokenizer,
                processor=self._processor,
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=1,
                repetition_penalty=1.0,
                stream=False,
                max_slice_nums=slice_budget,
                enable_thinking=False,
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = parse_canonical_decision(
            str(raw_output),
            self.policy.categories,
        )
        return build_native_record(
            sample=sample,
            identity=self.identity,
            policy=self.policy,
            generation={
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "max_new_tokens": self.max_new_tokens,
                "attn_implementation": self.attn_implementation,
                "dtype": "bfloat16",
                "max_slice_nums": MAX_SLICE_NUMS,
                "effective_max_slice_nums": slice_budget,
                "max_total_slices": MAX_TOTAL_SLICES,
                "enable_thinking": False,
            },
            prompt_sha256=prompt_sha256,
            raw_output=str(raw_output),
            parsed=parsed,
            latency_ms=latency_ms,
            peak_vram_gib=torch.cuda.max_memory_allocated() / (1024**3),
        )

    def close(self) -> None:
        self._tokenizer = None
        self._processor = None
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        gc.collect()
