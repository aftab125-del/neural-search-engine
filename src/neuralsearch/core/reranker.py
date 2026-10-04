"""
Neural Cross-Encoder Reranker using quantized ONNX Runtime for deep
query-document cross-attention and relevance scoring.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer as HFTokenizer

DEFAULT_RERANKER_DIR = Path("D:/neural-search-engine/.cache/models/ms-marco-MiniLM-L-6-v2")


class CrossEncoderReranker:
    """
    Evaluates query-document pairs jointly with full cross-attention.
    Significantly outperforms bi-encoder embeddings at ranking top candidates.
    """

    def __init__(
        self,
        model_path: Union[str, Path] | None = None,
        tokenizer_path: Union[str, Path] | None = None,
        max_length: int = 512,
        num_threads: int = 4,
    ):
        base_dir = DEFAULT_RERANKER_DIR
        self.model_path = Path(model_path) if model_path else base_dir / "onnx" / "model_quantized.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else base_dir / "tokenizer.json"
        self.max_length = max_length

        if not self.model_path.exists():
            raise FileNotFoundError(f"Cross-encoder ONNX model not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Cross-encoder tokenizer not found: {self.tokenizer_path}")

        self.tokenizer = HFTokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=self.max_length)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Computes relevance probability in [0, 1] for a list of (query, document) pairs.
        """
        if not pairs:
            return []

        # Tokenize pair sequences: [CLS] query [SEP] document [SEP]
        encodings = self.tokenizer.encode_batch(pairs)

        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids for enc in encodings], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, feed)
        logits = outputs[0].flatten()

        # Convert logits to calibrated probabilities via sigmoid
        probabilities = [round(self._sigmoid(float(logit)), 4) for logit in logits]
        return probabilities
