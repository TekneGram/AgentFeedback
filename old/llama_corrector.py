from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any
import os
import re

import json
from typing import Optional

# Server mode dependency
import requests

try:
    from llama_cpp import Llama # local mode dependency
except Exception:
    Llama = None

Backend = Literal["local", "server"]

@dataclass
class LlamaConfig:
    backend: Backend = "local"
    # --- local (llama-cpp-python) ---
    model_path: str = ""
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    n_batch: int = 512

    # --- server (llama.cpp OpenAI-compatible) ---
    server_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    server_model: str = "llama"
    server_timeout_s: int = 120

    # generation
    max_tokens: int = 128
    temperature: float = 0.0

class LlamaCorrector:
    """
    Keeps a single LlaMA instance alive (local) OR uses a persistent server.
    Includes a simple in-memory cache to avoid re-correcting identical sentences.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        self.cfg = cfg
        self._cache: Dict[str,str] = {}

        self.llm = None

        if self.cfg.backend == "local":
            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python is not installed but backend='local' was chosen.\n"
                    "Install: pip install llama-cpp-python"
                )
            if not self.cfg.model_path:
                raise ValueError("For backend='local', you must set cfg.model_path to your GGUF path.")
            
            n_threads = self.cfg.n_threads or (os.cpu_count() or 4)

            # Load ONCE for the whole run
            self.llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_thread=n_threads,
                n_batch=self.cfg.n_batch,
                verbose=False,
            )

        elif self.cfg.backend == "server":
            # No local model; we'll call HTTP
            pass

        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

    
    def _build_llama_instruct_prompt(self, sentence: str) -> str:
        # Llama 3.1 Instruct chat template
        # Return only the corrected sentence
        return(
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a careful English writing assistant.\n"
            "Fix grammar and word choice errors but keep the original meaning.\n"
            "Return ONLY the corrected sentence. No explanations. No quotes.\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{sentence}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    def _postprocess_one_line(self, text: str) -> str:
        # Keep first non-empty line, strip quotes/spaces
        text = text.strip()
        text = text.replace("\r\n", "\n")
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        out = lines[0] if lines else ""

        # Remove wrapping quotes if any
        out = re.sub(r'^\s*["“”]\s*', "", out)
        out = re.sub(r'\s*["“”]\s*$', "", out)

        return out.strip()
    
    def correct_one(self, sentence: str) -> str:
        sentence_key = sentence.strip()
        if not sentence_key:
            return sentence
        
        if sentence_key in self._cache:
            return self._cache[sentence_key]
        
        if self.cfg.backend == "local":
            prompt = self._build_llama_instruct_prompt(sentence_key)
            resp = self.llm(
                prompt,
                max_tokens = self.cfg.max_tokens,
                temperature = self.cfg.temperature,
                stop=["<|eot_id|>", "\n\n"],
            )
            text = resp["choices"][0]["text"]
            corrected = self._postprocess_one_line(text)
        else:
            payload = {
                "model": self.cfg.server_model,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful English writing assistant. "
                            "Fix grammar and word choice errors but keep the original meaning. "
                            "Return ONLY the corrected sentence. No explanations. No quotes."
                        ),
                    },
                    {"role": "user", "content": sentence_key},
                ],
            }
            r = requests.post(self.cfg.server_url, json=payload, timeout=self.cfg.server_timeout_s)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            corrected = self._postprocess_one_line(text)
        
        # Fallback: if model returns empty, keep original
        if not corrected:
            corrected = sentence_key

        self._cache[sentence_key] = corrected
        return corrected
    
    def correct_many(self, sentences: List[str]) -> List[str]:
        return [self.correct_one(s) for s in sentences]


    def _chat_once(self, system_msg: str, user_msg: str, max_tokens: int) -> str:
        """
        Backend-agnostic chat call. Returns assistant text.
        """
        if self.cfg.backend == "local":
            if self.llm is None:
                raise RuntimeError("Local backend selected but self.llm is None (model not loaded).")

            prompt = (
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{system_msg}\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{user_msg}\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )

            resp = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.cfg.temperature,
                stop=["<|eot_id|>"],
            )
            return (resp["choices"][0]["text"] or "").strip()

        # server
        payload = {
            "model": self.cfg.server_model,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        r = requests.post(self.cfg.server_url, json=payload, timeout=self.cfg.server_timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def cause_effect_feedback(self, paragraph: str, phrases_used: list[str]) -> str:
        """
        Implements your rules:
        - 0 or 1 phrase: advice to increase cause-effect language
        - 2+ phrases: comment on effective use (and tell the model which phrases were used)
        """
        para = (paragraph or "").strip()
        if not para:
            return ""

        # Cache by mode + text + phrases
        key = f"CE::{len(phrases_used)}::{','.join([p.lower() for p in phrases_used])}::{para}"
        if key in self._cache:
            return self._cache[key]

        if len(phrases_used) <= 1:
            system_msg = (
                "You are an English writing coach. Focus ONLY on cause-effect language.\n"
                "Give practical, paragraph-level advice to improve cause-effect language.\n"
                "Be concise and concrete. No long theory.\n"
                "Refer to the writer as you / your."
            )
            user_msg = (
                "Task: The paragraph uses little cause-effect language.\n"
                "1) Briefly explain why more cause-effect signaling could improve clarity.\n"
                "2) By using cause-effect language, suggest 1 revised sentence or additional sentence.\n\n"
                f"Paragraph:\n{para}"
            )
            out = self._chat_once(system_msg, user_msg, max_tokens=256)
        else:
            system_msg = (
                "You are an English writing coach. Focus ONLY on cause-effect language.\n"
                "Comment on effectiveness. Do not suggest improvements.\n"
                "Be concise.\n"
                "Refer to the writer as you / your."
            )
            used_list = ", ".join(phrases_used)
            user_msg = (
                "Task: The writer used cause-effect language.\n"
                f"Cause-effect expressions detected: {used_list}\n\n"
                "1) Comment in one sentence on how effectively these expressions clarify cause and effect.\n\n"
                f"Paragraph:\n{para}"
            )
            out = self._chat_once(system_msg, user_msg, max_tokens=256)

        self._cache[key] = out
        return out
    
    def compare_contrast_feedback(self, paragraph: str, phrases_used: list[str]) -> str:
        """
        Rules:
        - If no compare/contrast language: encourage comparing two viewpoints + simple example
        - If present: praise what was done well (and mention detected phrases)
        """
        para = (paragraph or "").strip()
        if not para:
            return ""

        key = f"CC::{','.join(p.lower() for p in phrases_used)}::{para}"
        if key in self._cache:
            return self._cache[key]

        if len(phrases_used) == 0:
            system_msg = (
                "You are an English writing coach. Focus ONLY on compare/contrast language.\n"
                "Be concise, practical, and beginner-friendly.\n"
                "Refer to the writer as you / your."
            )
            user_msg = (
                "Task: The paragraph does not use compare/contrast language.\n"
                "1) Encourage the writer to compare two points of view.\n"
                "2) Provide one compare/contrast expression that would fit.\n"
                "3) Refer to the writer as you / your.\n\n"
                f"Paragraph:\n{para}"
            )
            out = self._chat_once(system_msg, user_msg, max_tokens=256)
        else:
            used_list = ", ".join(phrases_used)
            system_msg = (
                "You are an English writing coach. Focus ONLY on compare/contrast language.\n"
                "Praise what is effective.\n"
                "Be concise."
            )
            user_msg = (
                "Task: The writer used compare/contrast language.\n"
                f"Detected compare/contrast expressions: {used_list}\n\n"
                "In one sentence, say what was done well (clarity, contrast, balanced comparison).\n"
                "Refer to the writer as you / your.\n\n"
                f"Paragraph:\n{para}"
            )
            out = self._chat_once(system_msg, user_msg, max_tokens=256)

        self._cache[key] = out
        return out

    def topic_sentence_feedback(self, paragraph: str) -> str:
        para = (paragraph or "").strip()
        if not para:
            return ""
        
        system_msg = (
            "You are a beginner friendly English writing coach. Focus ONLY on the topic sentence.\n"
            "Be concise, practical, and beginner-friendly.\n"
            "Refer to the writer as you / your"
        )
        user_msg = (
            "Task: The writer may or may not have an effective topic sentence.\n"
            "1) Check whether the first sentence introduces the main idea of the writing.\n"
            "2) In one sentence, explain that and, *only* if necessary, offer a recommended change to the first sentence.\n\n"
            f"Paragraph:\n{para}"
        )
        out = self._chat_once(system_msg, user_msg, max_tokens=256)

        # self._cache[key] = out
        return out
    
    def conclusion_sentence_feedback(self, paragraph: str) -> str:
        para = (paragraph or "").strip()
        if not para:
            return ""
        
        system_msg = (
            "You are a beginner friendly English writing coach. Focus ONLY on the conclusion sentence.\n"
            "Be concise, practical, and beginner-friendly.\n"
            "Refer to the writer as you / your."
        )
        user_msg = (
            "Task: The writer may or may not have an effective conclusion sentence.\n"
            "1) Check whether the final sentence summarizes the main idea and key points of the writing.\n"
            "2) In one sentence, explain that and, *only* if necessary, offer a recommended change to the final sentence.\n\n"
            f"Paragraph:\n{para}"
        )
        out = self._chat_once(system_msg, user_msg, max_tokens=256)

        # self._cache[key] = out
        return out
    
    def praise_sentence(self, paragraph: str) -> str:
        para = (paragraph or "").strip()
        if not para:
            return ""
        
        system_msg = (
            "You are a judge of writing. You write what was done well in the writing."
        )

        user_msg = (
            "Task: Read the paragraph. In one sentence, praise the writer for something done well. Refer to the writer directly with you/your."
            f"Paragraph:\n{para}"
        )

        out = self._chat_once(system_msg, user_msg, max_tokens=128)
        return out
    
    # def personalize_feedback(self, feedback: str) -> str:
    #     fb = (feedback or "").strip()
    #     if not fb:
    #         return ""
        
    #     system_msg = (
    #         "You are an amiable feedback editor.\n"
    #         "You make impersonal feedback sound more personalized.\n"
    #     )
    #     user_msg = (
    #         "Task: The feedback is not personalized. Make it personalized with more words like you.\n"
    #         "Return the feedback using the same <<<PARA>>> separators and the same number of blocks. Do not add, remove or reorder blocks. Do not add any phrase like Here is the rewritten feedback. \n\n"
    #         f"Feedback:\n{fb}"
    #     )
    #     out = self._chat_once(system_msg, user_msg, max_tokens=1024)

    #     # self._cache[key] = out
    #     return out
    
    def _extract_json_object(self, text: str) -> str:
        """
        If the model outputs extra text, extract the first {...} JSON object.
        """
        t = (text or "").strip()
        if not t:
            raise ValueError("Empty LLM output (expected JSON).")

        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not find a JSON object in LLM output.")

        return t[start:end+1]

    def _extract_first_json(self, text: str) -> str:
        """
        Extract the first JSON object OR array from model output.
        Handles stray pre/post text.
        """
        t = (text or "").strip()
        if not t:
            raise ValueError("Empty LLM output (expected JSON).")

        # find first { or [
        i_obj = t.find("{")
        i_arr = t.find("[")
        if i_obj == -1 and i_arr == -1:
            raise ValueError("Could not find JSON start '{' or '[' in LLM output.")

        start = i_obj if (i_obj != -1 and (i_arr == -1 or i_obj < i_arr)) else i_arr
        open_char = t[start]
        close_char = "}" if open_char == "{" else "]"

        # naive matching by last close char (works well for our use case)
        end = t.rfind(close_char)
        if end == -1 or end <= start:
            raise ValueError("Could not find JSON end in LLM output.")

        return t[start:end + 1]

    def _coerce_paragraphs_list(self, paras: Any) -> List[str]:
        """
        Force model output 'paragraphs' into List[str], preserving empties.
        """
        if not isinstance(paras, list):
            raise ValueError("'paragraphs' is not a list in LLM output.")

        out: List[str] = []
        for p in paras:
            if p is None:
                out.append("")
            elif isinstance(p, str):
                out.append(p.replace("\r\n", "\n").rstrip())
            else:
                out.append(str(p).replace("\r\n", "\n").rstrip())
        return out

    def _repair_to_input_shape(self, inp_paras: List[str], out_paras: List[str]) -> List[str]:
        """
        Repair output to match input structure WITHOUT losing content.

        Key fix:
        - If the input has only ONE non-empty slot, but the model outputs multiple paragraphs,
        MERGE them back into that one slot instead of taking only the first one.
        """
        target_len = len(inp_paras)

        inp_nonempty_idx = [i for i, p in enumerate(inp_paras) if (p or "").strip() != ""]
        inp_empty_idx = [i for i, p in enumerate(inp_paras) if (p or "").strip() == ""]

        # Normalize output
        norm_out: List[str] = []
        for p in out_paras:
            if p is None:
                norm_out.append("")
            else:
                norm_out.append(str(p).replace("\r\n", "\n").rstrip())

        repaired = [""] * target_len

        # Preserve empty slots exactly
        for i in inp_empty_idx:
            repaired[i] = ""

        if not inp_nonempty_idx:
            return repaired

        # --- SPECIAL CASE: only one content slot in input ---
        # Merge ALL model output paragraphs into that slot, preserving blank lines.
        if len(inp_nonempty_idx) == 1:
            merged = "\n\n".join(
                [p.strip() if p.strip() else "" for p in norm_out]
            ).strip()

            if not merged:
                merged = (inp_paras[inp_nonempty_idx[0]] or "").strip()

            repaired[inp_nonempty_idx[0]] = merged
            return repaired

        # --- GENERAL CASE: multiple slots ---
        out_nonempty = [p.strip() for p in norm_out if (p or "").strip() != ""]

        # Fill each input content slot with the next output content paragraph
        for k, i in enumerate(inp_nonempty_idx):
            if k < len(out_nonempty):
                repaired[i] = out_nonempty[k]
            else:
                repaired[i] = (inp_paras[i] or "").rstrip()

        # If model produced extra content paragraphs, append them to the LAST content slot
        if len(out_nonempty) > len(inp_nonempty_idx):
            leftovers = out_nonempty[len(inp_nonempty_idx):]
            if leftovers:
                last_i = inp_nonempty_idx[-1]
                repaired[last_i] = repaired[last_i].rstrip() + "\n\n" + "\n\n".join(leftovers).rstrip()

        return repaired


    def personalize_feedback(self, feedback_json: str, expected_count: Optional[int] = None) -> str:
        fb = (feedback_json or "").strip()
        if not fb:
            return ""

        # ---- Parse and validate INPUT ----
        try:
            inp_obj = json.loads(fb)
        except Exception:
            raise ValueError("personalize_feedback() expects valid JSON input.")

        if not isinstance(inp_obj, dict) or "paragraphs" not in inp_obj:
            raise ValueError("personalize_feedback() expects JSON with top-level key 'paragraphs'.")

        inp_paras_raw = inp_obj["paragraphs"]
        if not isinstance(inp_paras_raw, list):
            raise ValueError("Input JSON 'paragraphs' must be a list.")

        inp_paras = self._coerce_paragraphs_list(inp_paras_raw)

        # Optional sanity check (non-fatal): expected_count should match input
        if expected_count is not None and expected_count != len(inp_paras):
            # Don't crash; just ignore and proceed with input as truth
            pass

        system_msg = (
            "You are a friendly feedback editor.\n"
            "Goal: make the feedback sound more personal by addressing the student as 'you'/'your'.\n"
            "IMPORTANT STYLE RULES:\n"
            "- Be polite.\n"
            "- Keep criticism but make them polite.\n"
        )

        user_msg = (
            "Task:\n"
            "You will receive feedback as JSON with keys:\n"
            "  schema: string\n"
            "  paragraphs: array of strings (some may be empty for spacing)\n\n"
            "Return JSON ONLY (no extra text) with the SAME keys.\n"
            "Keep the paragraphs array the SAME LENGTH as input and keep empty strings in the same positions.\n"
            "Rewrite only the NON-empty strings to be more personalized and polite.\n\n"
            "INPUT JSON:\n"
            f"{fb}"
        )

        raw = self._chat_once(system_msg, user_msg, max_tokens=1024)
        print(f"This is the raw output from the llm: {raw}")

        # ---- Parse OUTPUT JSON (robust extraction) ----
        json_text = self._extract_first_json(raw)
        out_any = json.loads(json_text)

        # Support either dict with paragraphs or bare array fallback
        if isinstance(out_any, dict):
            out_obj = out_any
            out_paras_raw = out_obj.get("paragraphs", None)
            if out_paras_raw is None:
                raise ValueError("LLM output JSON missing 'paragraphs'.")
        elif isinstance(out_any, list):
            # model returned just the array; wrap it
            out_obj = {"schema": "essaylens_feedback_v1", "paragraphs": out_any}
            out_paras_raw = out_any
        else:
            raise ValueError("LLM output JSON was neither object nor array.")

        out_paras = self._coerce_paragraphs_list(out_paras_raw)

        # ---- REPAIR to match input shape (fix your exact bug) ----
        repaired = self._repair_to_input_shape(inp_paras, out_paras)

        # Force schema
        out_obj["schema"] = "essaylens_feedback_v1"
        out_obj["paragraphs"] = repaired

        return json.dumps(out_obj, ensure_ascii=False)