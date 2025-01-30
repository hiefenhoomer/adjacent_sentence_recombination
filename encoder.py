from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
from typing import List


class Encoder:
    def __init__(self, encoder_name: str) -> None:
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder_name = encoder_name

        if not torch.cuda.is_available():
            raise RuntimeError('No GPU available!')

        self.device = "cuda"

        self.encoder = self.encoder.to(self.device)

    def encode_n(self, text: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            embeddings = self.encoder(**tokens)
            embeddings = embeddings.last_hidden_state[:, 0, :]

        embeddings = F.normalize(embeddings, p=2)

        return embeddings

    @staticmethod
    def get_adjacent_similarity_statistics(embeddings: torch.Tensor) -> (float, List[float]):
        similarity_tensors = F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1)
        return similarity_tensors
