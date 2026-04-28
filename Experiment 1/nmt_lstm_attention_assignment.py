# -*- coding: utf-8 -*-
"""Multi-layer LSTM machine translation assignment.

The script trains a German-to-English encoder-decoder model on the local
IWSLT14 files.  It contains three models:

1. MultiLayerLSTMSeq2Seq: stacked LSTM encoder + stacked LSTM decoder.
2. AttentionSeq2Seq: the same stacked LSTM backbone with Bahdanau attention.
3. AttnResLSTMSeq2Seq: a LSTM adaptation of Block Attention Residuals
   (arXiv:2603.15031v1), using learned depth-wise attention over completed
   block representations and the current intra-block partial sum.

Small defaults make the program easy to smoke-test on a laptop.  On a server,
increase --max-train-samples, --epochs, --embed-size and --hidden-size.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    import nltk
except ImportError:  # pragma: no cover - only used when nltk is missing.
    nltk = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
except ImportError:  # pragma: no cover - only used when nltk is missing.
    SmoothingFunction = None
    corpus_bleu = None


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<start>"
EOS_TOKEN = "<end>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def set_seed(seed: int) -> None:
    """Make training as reproducible as possible."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str, language: str) -> List[str]:
    """Tokenize one sentence without requiring downloaded NLTK punkt data."""
    text = text.strip()
    if not text:
        return []

    if nltk is not None:
        try:
            return nltk.word_tokenize(text, language=language, preserve_line=True)
        except LookupError:
            pass

    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def read_parallel_corpus(
    source_path: Path,
    target_path: Path,
    max_len: int,
    max_samples: Optional[int],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Read, clean and tokenize a parallel source-target corpus."""
    source_sentences: List[List[str]] = []
    target_sentences: List[List[str]] = []

    with source_path.open("r", encoding="utf-8", errors="replace") as src_file:
        source_lines = src_file.readlines()
    with target_path.open("r", encoding="utf-8", errors="replace") as tgt_file:
        target_lines = tgt_file.readlines()

    for source_line, target_line in zip(source_lines, target_lines):
        source_line = source_line.strip()
        target_line = target_line.strip()

        if not source_line or not target_line:
            continue
        if source_line.startswith("<") or target_line.startswith("<"):
            continue

        source_tokens = simple_tokenize(source_line, language="german")
        target_tokens = simple_tokenize(target_line, language="english")

        if 0 < len(source_tokens) <= max_len - 1 and 0 < len(target_tokens) <= max_len - 1:
            source_sentences.append(source_tokens)
            target_sentences.append(target_tokens)

        if max_samples is not None and len(source_sentences) >= max_samples:
            break

    return source_sentences, target_sentences


@dataclass
class Vocabulary:
    """Bidirectional token/id mapping."""

    token_to_id: Dict[str, int]

    @property
    def id_to_token(self) -> Dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    @classmethod
    def build(cls, sentences: Sequence[Sequence[str]], max_size: int) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(sentence)

        token_to_id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }

        for token, _ in counter.most_common(max(0, max_size - len(token_to_id))):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Sequence[str], add_eos: bool = True) -> List[int]:
        ids = [self.token_to_id.get(token, UNK_ID) for token in tokens]
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int]) -> List[str]:
        id_to_token = self.id_to_token
        tokens: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx == EOS_ID:
                break
            if idx in {PAD_ID, BOS_ID}:
                continue
            tokens.append(id_to_token.get(idx, UNK_TOKEN))
        return tokens


def pad_to_length(ids: Sequence[int], max_len: int) -> List[int]:
    """Pad or truncate an id sequence to a fixed length."""
    ids = list(ids[:max_len])
    return ids + [PAD_ID] * (max_len - len(ids))


class IWSLTTranslationDataset(Dataset):
    """Torch Dataset that returns source ids, lengths and decoder targets."""

    def __init__(
        self,
        source_sentences: Sequence[Sequence[str]],
        target_sentences: Sequence[Sequence[str]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
        max_len: int,
        reverse_source: bool = True,
    ) -> None:
        self.source_tokens = [list(sentence) for sentence in source_sentences]
        self.target_tokens = [list(sentence) for sentence in target_sentences]
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        self.reverse_source = reverse_source

        self.examples = []
        for source_tokens, target_tokens in zip(self.source_tokens, self.target_tokens):
            source_ids = source_vocab.encode(source_tokens, add_eos=True)
            if reverse_source:
                source_ids = list(reversed(source_ids))
            target_ids = target_vocab.encode(target_tokens, add_eos=True)
            decoder_input_ids = [BOS_ID] + target_ids[:-1]

            self.examples.append(
                (
                    pad_to_length(source_ids, max_len),
                    min(len(source_ids), max_len),
                    pad_to_length(decoder_input_ids, max_len),
                    pad_to_length(target_ids, max_len),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_ids, source_len, decoder_input_ids, target_ids = self.examples[index]
        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(source_len, dtype=torch.long),
            torch.tensor(decoder_input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


class MultiLayerLSTMSeq2Seq(nn.Module):
    """Stacked LSTM encoder-decoder without attention."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size, padding_idx=PAD_ID)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, target_vocab_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        source_embedded = self.source_embedding(source_ids)
        _, encoder_hidden = self.encoder(source_embedded)

        decoder_embedded = self.target_embedding(decoder_input_ids)
        decoder_outputs, _ = self.decoder(decoder_embedded, encoder_hidden)
        logits = self.output_layer(self.dropout(decoder_outputs))
        return logits

    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        max_len: int,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        return_attention: bool = False,
    ) -> torch.Tensor:
        source_embedded = self.source_embedding(source_ids)
        _, hidden = self.encoder(source_embedded)

        current_ids = torch.full(
            (source_ids.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        predictions = []

        for _ in range(max_len):
            decoder_embedded = self.target_embedding(current_ids)
            decoder_output, hidden = self.decoder(decoder_embedded, hidden)
            logits = self.output_layer(self.dropout(decoder_output))
            current_ids = logits.argmax(dim=-1)
            predictions.append(current_ids)
            if torch.all(current_ids.squeeze(1) == eos_id):
                break

        return torch.cat(predictions, dim=1)


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by the Attention Residuals module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        rms = values.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return values / rms * self.weight


class DepthAttentionResidual(nn.Module):
    """Learned attention over residual streams along the depth axis.

    The paper uses a trainable pseudo-query to attend over residual history
    instead of simply adding the latest residual stream.  This module implements
    the core operation shown in Fig. 2 of Attention Residuals:

    V = stack(blocks + [partial_block])
    K = RMSNorm(V)
    alpha = softmax(w dot K, dim=depth)
    h = sum_depth alpha * V
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.pseudo_query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.pseudo_query, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, states: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked_states = torch.stack(list(states), dim=0)
        keys = self.norm(stacked_states)
        scores = torch.einsum("h,sbth->sbt", self.pseudo_query, keys)
        weights = torch.softmax(scores, dim=0)
        combined = torch.sum(weights.unsqueeze(-1) * stacked_states, dim=0)
        return combined, weights


class AttnResLSTMStack(nn.Module):
    """A stack of one-layer LSTMs connected by Block Attention Residuals."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        block_size: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.residual_mixers = nn.ModuleList([DepthAttentionResidual(hidden_size) for _ in range(num_layers)])
        self.output_mixer = DepthAttentionResidual(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.block_size = max(1, block_size)

    def forward(
        self,
        inputs: torch.Tensor,
        initial_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        token_block = self.input_projection(inputs)
        completed_blocks = [token_block]
        partial_block: Optional[torch.Tensor] = None
        hidden_states = []
        cell_states = []
        residual_weights = []

        for layer_index, (lstm, mixer) in enumerate(zip(self.layers, self.residual_mixers)):
            if layer_index % self.block_size == 0:
                if partial_block is not None:
                    completed_blocks.append(partial_block)
                partial_block = torch.zeros_like(token_block)

            layer_position = layer_index % self.block_size
            if layer_position == 0:
                residual_sources = completed_blocks
            else:
                residual_sources = completed_blocks + [partial_block]

            layer_input, weights = mixer(residual_sources)
            if initial_hidden is None:
                layer_output, (hidden, cell) = lstm(layer_input)
            else:
                layer_hidden = (
                    initial_hidden[0][layer_index : layer_index + 1].contiguous(),
                    initial_hidden[1][layer_index : layer_index + 1].contiguous(),
                )
                layer_output, (hidden, cell) = lstm(layer_input, layer_hidden)

            if layer_index < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            partial_block = partial_block + layer_output
            hidden_states.append(hidden)
            cell_states.append(cell)
            residual_weights.append(weights)

        if partial_block is not None:
            completed_blocks.append(partial_block)
        final_output, final_weights = self.output_mixer(completed_blocks)
        residual_weights.append(final_weights)

        final_hidden = (torch.cat(hidden_states, dim=0), torch.cat(cell_states, dim=0))
        return final_output, final_hidden, residual_weights


class AttnResLSTMSeq2Seq(nn.Module):
    """Seq2Seq model with Attention Residuals between LSTM layers."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        attnres_block_size: int = 2,
    ) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size, padding_idx=PAD_ID)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size, padding_idx=PAD_ID)
        self.encoder = AttnResLSTMStack(embed_size, hidden_size, num_layers, dropout, attnres_block_size)
        self.decoder = AttnResLSTMStack(embed_size, hidden_size, num_layers, dropout, attnres_block_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, target_vocab_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        source_embedded = self.source_embedding(source_ids)
        _, encoder_hidden, _ = self.encoder(source_embedded)

        decoder_embedded = self.target_embedding(decoder_input_ids)
        decoder_outputs, _, _ = self.decoder(decoder_embedded, encoder_hidden)
        logits = self.output_layer(self.dropout(decoder_outputs))
        return logits

    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        max_len: int,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        return_attention: bool = False,
    ) -> torch.Tensor:
        source_embedded = self.source_embedding(source_ids)
        _, hidden, _ = self.encoder(source_embedded)

        current_ids = torch.full(
            (source_ids.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        predictions = []

        for _ in range(max_len):
            decoder_embedded = self.target_embedding(current_ids)
            decoder_output, hidden, _ = self.decoder(decoder_embedded, hidden)
            logits = self.output_layer(self.dropout(decoder_output))
            current_ids = logits.argmax(dim=-1)
            predictions.append(current_ids)
            if torch.all(current_ids.squeeze(1) == eos_id):
                break

        return torch.cat(predictions, dim=1)


class BahdanauAttention(nn.Module):
    """Additive attention over all encoder time steps."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_projection = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_outputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_scores = self.encoder_projection(encoder_outputs).unsqueeze(1)
        decoder_scores = self.decoder_projection(decoder_outputs).unsqueeze(2)
        scores = self.score_projection(torch.tanh(encoder_scores + decoder_scores)).squeeze(-1)
        scores = scores.masked_fill(~source_mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, encoder_outputs)
        return context, weights


class AttentionSeq2Seq(nn.Module):
    """Stacked LSTM encoder-decoder with Bahdanau attention."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size, padding_idx=PAD_ID)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = BahdanauAttention(hidden_size)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, target_vocab_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        source_mask = source_ids.ne(PAD_ID)
        source_embedded = self.source_embedding(source_ids)
        encoder_outputs, encoder_hidden = self.encoder(source_embedded)

        decoder_embedded = self.target_embedding(decoder_input_ids)
        decoder_outputs, _ = self.decoder(decoder_embedded, encoder_hidden)
        context, attention_weights = self.attention(decoder_outputs, encoder_outputs, source_mask)
        combined = torch.tanh(self.attention_combine(torch.cat([decoder_outputs, context], dim=-1)))
        logits = self.output_layer(self.dropout(combined))

        if return_attention:
            return logits, attention_weights
        return logits

    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        max_len: int,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        return_attention: bool = False,
    ):
        source_mask = source_ids.ne(PAD_ID)
        source_embedded = self.source_embedding(source_ids)
        encoder_outputs, hidden = self.encoder(source_embedded)

        current_ids = torch.full(
            (source_ids.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        predictions = []
        attention_steps = []

        for _ in range(max_len):
            decoder_embedded = self.target_embedding(current_ids)
            decoder_output, hidden = self.decoder(decoder_embedded, hidden)
            context, attention_weights = self.attention(decoder_output, encoder_outputs, source_mask)
            combined = torch.tanh(self.attention_combine(torch.cat([decoder_output, context], dim=-1)))
            logits = self.output_layer(self.dropout(combined))
            current_ids = logits.argmax(dim=-1)
            predictions.append(current_ids)
            attention_steps.append(attention_weights)
            if torch.all(current_ids.squeeze(1) == eos_id):
                break

        prediction_tensor = torch.cat(predictions, dim=1)
        if return_attention:
            return prediction_tensor, torch.cat(attention_steps, dim=1)
        return prediction_tensor


def masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Cross entropy averaged over non-padding target positions."""
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=PAD_ID,
    )


def move_batch_to_device(batch, device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch and return average batch loss."""
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        source_ids, source_lengths, decoder_input_ids, target_ids = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(source_ids, decoder_input_ids, source_lengths)
        loss = masked_cross_entropy(logits, target_ids)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_loss(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Compute validation/test loss without gradient updates."""
    model.eval()
    total_loss = 0.0

    for batch in data_loader:
        source_ids, source_lengths, decoder_input_ids, target_ids = move_batch_to_device(batch, device)
        logits = model(source_ids, decoder_input_ids, source_lengths)
        loss = masked_cross_entropy(logits, target_ids)
        total_loss += loss.item()

    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def compute_bleu(
    model: nn.Module,
    data_loader: DataLoader,
    target_vocab: Vocabulary,
    device: torch.device,
    max_len: int,
) -> float:
    """Compute corpus BLEU on generated translations."""
    if corpus_bleu is None or SmoothingFunction is None:
        return 0.0

    model.eval()
    references = []
    hypotheses = []

    for batch in data_loader:
        source_ids, _, _, target_ids = move_batch_to_device(batch, device)
        predictions = model.greedy_decode(source_ids, max_len=max_len)

        for pred_row, target_row in zip(predictions.cpu().tolist(), target_ids.cpu().tolist()):
            pred_tokens = target_vocab.decode(pred_row)
            ref_tokens = target_vocab.decode(target_row)
            if pred_tokens and ref_tokens:
                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    if not hypotheses:
        return 0.0

    smoothing = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, smoothing_function=smoothing) * 100.0


@torch.no_grad()
def collect_translation_examples(
    model: nn.Module,
    dataset: IWSLTTranslationDataset,
    target_vocab: Vocabulary,
    device: torch.device,
    max_len: int,
    count: int,
) -> List[Tuple[str, str, str]]:
    """Generate human-readable translation examples."""
    model.eval()
    examples = []

    for index in range(min(count, len(dataset))):
        source_ids, _, _, target_ids = dataset[index]
        predictions = model.greedy_decode(source_ids.unsqueeze(0).to(device), max_len=max_len)
        pred_tokens = target_vocab.decode(predictions.squeeze(0).cpu().tolist())
        ref_tokens = target_vocab.decode(target_ids.tolist())

        source_text = " ".join(dataset.source_tokens[index])
        reference_text = " ".join(ref_tokens)
        prediction_text = " ".join(pred_tokens)
        examples.append((source_text, reference_text, prediction_text))

    return examples


def write_examples(path: Path, examples: Sequence[Tuple[str, str, str]]) -> None:
    """Save source/reference/prediction triples for screenshots and reports."""
    with path.open("w", encoding="utf-8") as file:
        for index, (source, reference, prediction) in enumerate(examples, start=1):
            file.write(f"[{index}]\n")
            file.write(f"SRC : {source}\n")
            file.write(f"REF : {reference}\n")
            file.write(f"PRED: {prediction}\n\n")


def write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    """Save per-epoch metrics as CSV."""
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: Sequence[Dict[str, float]], title: str) -> None:
    """Plot train/validation loss and BLEU curves."""
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]
    bleu = [row["test_bleu"] for row in history]

    fig, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.plot(epochs, train_loss, marker="o", label="train loss")
    loss_axis.plot(epochs, valid_loss, marker="o", label="valid loss")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(True, alpha=0.3)

    bleu_axis = loss_axis.twinx()
    bleu_axis.plot(epochs, bleu, marker="s", color="tab:green", label="test BLEU")
    bleu_axis.set_ylabel("BLEU")

    lines, labels = loss_axis.get_legend_handles_labels()
    lines2, labels2 = bleu_axis.get_legend_handles_labels()
    loss_axis.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_attention_heatmap(
    model: nn.Module,
    dataset: IWSLTTranslationDataset,
    target_vocab: Vocabulary,
    device: torch.device,
    max_len: int,
    path: Path,
) -> None:
    """Save an attention alignment heatmap for the first validation example."""
    if not isinstance(model, AttentionSeq2Seq) or len(dataset) == 0:
        return

    model.eval()
    source_ids, source_len, _, _ = dataset[0]
    predictions, attention = model.greedy_decode(
        source_ids.unsqueeze(0).to(device),
        max_len=max_len,
        return_attention=True,
    )
    pred_ids = predictions.squeeze(0).cpu().tolist()
    pred_tokens = target_vocab.decode(pred_ids)
    if not pred_tokens:
        pred_tokens = [EOS_TOKEN]

    token_count = len(pred_tokens)
    source_tokens = dataset.source_tokens[0] + [EOS_TOKEN]
    if dataset.reverse_source:
        source_tokens = list(reversed(source_tokens))
    source_tokens = source_tokens[: int(source_len.item())]

    attention_matrix = attention.squeeze(0).cpu()[:token_count, : len(source_tokens)]

    fig, axis = plt.subplots(figsize=(max(7, len(source_tokens) * 0.45), max(4, token_count * 0.35)))
    image = axis.imshow(attention_matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(source_tokens)))
    axis.set_xticklabels(source_tokens, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(token_count))
    axis.set_yticklabels(pred_tokens, fontsize=8)
    axis.set_xlabel("source tokens")
    axis.set_ylabel("generated target tokens")
    axis.set_title("Attention weights")
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_datasets(args):
    data_dir = Path(args.data_dir)

    train_source, train_target = read_parallel_corpus(
        data_dir / args.train_source,
        data_dir / args.train_target,
        max_len=args.max_len,
        max_samples=none_if_non_positive(args.max_train_samples),
    )
    valid_source, valid_target = read_parallel_corpus(
        data_dir / args.valid_source,
        data_dir / args.valid_target,
        max_len=args.max_len,
        max_samples=none_if_non_positive(args.max_eval_samples),
    )
    test_source, test_target = read_parallel_corpus(
        data_dir / args.test_source,
        data_dir / args.test_target,
        max_len=args.max_len,
        max_samples=none_if_non_positive(args.max_eval_samples),
    )

    source_vocab = Vocabulary.build(train_source, args.source_vocab_size)
    target_vocab = Vocabulary.build(train_target, args.target_vocab_size)

    train_dataset = IWSLTTranslationDataset(
        train_source,
        train_target,
        source_vocab,
        target_vocab,
        max_len=args.max_len,
        reverse_source=args.reverse_source,
    )
    valid_dataset = IWSLTTranslationDataset(
        valid_source,
        valid_target,
        source_vocab,
        target_vocab,
        max_len=args.max_len,
        reverse_source=args.reverse_source,
    )
    test_dataset = IWSLTTranslationDataset(
        test_source,
        test_target,
        source_vocab,
        target_vocab,
        max_len=args.max_len,
        reverse_source=args.reverse_source,
    )

    return train_dataset, valid_dataset, test_dataset, source_vocab, target_vocab


def create_model(name: str, source_vocab_size: int, target_vocab_size: int, args) -> nn.Module:
    if name == "attention":
        model_class = AttentionSeq2Seq
        return model_class(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if name == "attnres":
        return AttnResLSTMSeq2Seq(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            attnres_block_size=args.attnres_block_size,
        )
    else:
        model_class = MultiLayerLSTMSeq2Seq
        return model_class(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    valid_dataset: IWSLTTranslationDataset,
    target_vocab: Vocabulary,
    args,
    device: torch.device,
    output_dir: Path,
) -> List[Dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: List[Dict[str, float]] = []
    best_valid_loss = float("inf")

    print(f"\n=== Training {name} model ===")
    print(f"parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        valid_loss = evaluate_loss(model, valid_loader, device)
        test_bleu = compute_bleu(model, test_loader, target_vocab, device, args.max_len)
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_loss": round(valid_loss, 4),
            "test_bleu": round(test_bleu, 4),
            "seconds": round(elapsed, 2),
        }
        history.append(row)

        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"valid loss {valid_loss:.4f} | test BLEU {test_bleu:.2f} | {elapsed:.1f}s"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_dir / f"{name}_best.pt")

    examples = collect_translation_examples(
        model,
        valid_dataset,
        target_vocab,
        device,
        max_len=args.max_len,
        count=args.num_examples,
    )
    write_examples(output_dir / f"{name}_examples.txt", examples)
    write_history(output_dir / f"{name}_history.csv", history)
    plot_history(output_dir / f"{name}_training_curve.png", history, title=f"{name} training curve")

    if isinstance(model, AttentionSeq2Seq):
        save_attention_heatmap(
            model,
            valid_dataset,
            target_vocab,
            device,
            max_len=args.max_len,
            path=output_dir / "attention_heatmap.png",
        )

    print(f"saved outputs for {name} to: {output_dir}")
    print("sample translations:")
    for source, reference, prediction in examples[: min(3, len(examples))]:
        print(f"SRC : {source}")
        print(f"REF : {reference}")
        print(f"PRED: {prediction}\n")

    return history


def none_if_non_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-layer LSTM NMT assignment")
    parser.add_argument("--data-dir", type=str, default="data/iwslt14")
    parser.add_argument("--train-source", type=str, default="train.tags.de-en.de")
    parser.add_argument("--train-target", type=str, default="train.tags.de-en.en")
    parser.add_argument("--valid-source", type=str, default="IWSLT14.TED.dev2010.de-en.de")
    parser.add_argument("--valid-target", type=str, default="IWSLT14.TED.dev2010.de-en.en")
    parser.add_argument("--test-source", type=str, default="IWSLT14.TED.tst2012.de-en.de")
    parser.add_argument("--test-target", type=str, default="IWSLT14.TED.tst2012.de-en.en")
    parser.add_argument("--model", choices=["baseline", "attention", "attnres", "both", "all"], default="both")
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--source-vocab-size", type=int, default=12000)
    parser.add_argument("--target-vocab-size", type=int, default=12000)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attnres-block-size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.set_defaults(reverse_source=True)
    parser.add_argument("--reverse-source", dest="reverse_source", action="store_true")
    parser.add_argument("--no-reverse-source", dest="reverse_source", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, source_vocab, target_vocab = build_datasets(args)
    print(f"device: {device}")
    print(f"train/valid/test examples: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")
    print(f"source vocab: {len(source_vocab)} | target vocab: {len(target_vocab)}")
    print(f"max length: {args.max_len} | layers: {args.num_layers} | attnres block size: {args.attnres_block_size}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    if args.model == "both":
        model_names = ["baseline", "attention"]
    elif args.model == "all":
        model_names = ["baseline", "attention", "attnres"]
    else:
        model_names = [args.model]

    for model_name in model_names:
        model = create_model(model_name, len(source_vocab), len(target_vocab), args)
        train_model(
            model_name,
            model,
            train_loader,
            valid_loader,
            test_loader,
            valid_dataset,
            target_vocab,
            args,
            device,
            output_dir,
        )


if __name__ == "__main__":
    main()
