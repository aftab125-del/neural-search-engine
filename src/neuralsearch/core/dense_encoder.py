"""
Ultra-fast, CPU-optimized Dense Bi-Encoder utilizing ONNX Runtime
and Hugging Face Tokenizers for sub-15ms vector embeddings.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Union
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer as HFTokenizer

DEFAULT_MODEL_DIR = Path("D:/neural-search-engine/.cache/models/all-MiniLM-L6-v2")


class DenseEncoder:
    """
    Quantized ONNX Bi-Encoder for computing 384-dimensional dense text embeddings.
    Applies mean pooling and L2 normalization.
    """

    def __init__(
        self,
        model_path: Union[str, Path] | None = None,
        tokenizer_path: Union[str, Path] | None = None,
        max_length: int = 256,
        num_threads: int = 4,
    ):
        base_dir = DEFAULT_MODEL_DIR
        self.model_path = Path(model_path) if model_path else base_dir / "onnx" / "model_quantized.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else base_dir / "tokenizer.json"
        self.max_length = max_length

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tokenizer_path}")

        # Load HF Tokenizer
        self.tokenizer = HFTokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=self.max_length)

        # Configure ONNX Runtime session for optimal CPU inference
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Determine expected session input names
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, text: str) -> np.ndarray:
        """Encodes a single string into an L2-normalized 384-dim float32 vector."""
        vectors = self.encode_batch([text])
        return vectors[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a batch of strings into an L2-normalized (N, 384) float32 matrix.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Tokenize batch
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        # Build ONNX feed dictionary
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        # Run ONNX inference
        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]  # Shape: (batch_size, seq_len, hidden_dim)

        # Mean Pooling with attention mask
        # Expand attention mask to match hidden dimensions: (batch_size, seq_len, 1)
        expanded_mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * expanded_mask, axis=1)
        sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        # L2 Normalization
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = mean_pooled / norms

        return normalized.astype(np.float32)
