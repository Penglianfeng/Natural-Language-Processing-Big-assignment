# -*- coding: utf-8 -*-
"""多层 LSTM Seq2Seq 机器翻译实验。

本脚本是一个自包含实验程序，可以直接在终端或 PyCharm 中运行，
不需要额外下载数据集。程序会训练两个英文到中文的简单翻译模型：

1. 普通多层 LSTM Encoder-Decoder 模型。
2. 加入 Bahdanau 加性注意力机制的多层 LSTM 模型。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = 42) -> None:
    """固定随机种子，使训练结果尽量可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_en(sentence: str) -> List[str]:
    """英文分词：转为小写后按空格切分。"""

    return sentence.lower().strip().split()


def tokenize_zh(sentence: str) -> List[str]:
    """中文分词。

    内置语料中的中文词已经用空格分隔，这样既能降低实验规模，
    也方便观察注意力机制在源词和目标词之间的对应关系。
    """

    return sentence.strip().split()


def detokenize_zh(tokens: Sequence[str]) -> str:
    """将中文词 token 拼接回自然中文句子。"""

    return "".join(tokens)


@dataclass
class Vocabulary:
    """词表类：维护 token 与整数编号之间的双向映射。"""

    token_to_idx: Dict[str, int]
    idx_to_token: List[str]

    @classmethod
    def build(cls, tokenized_sentences: Iterable[Sequence[str]]) -> "Vocabulary":
        idx_to_token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in token_to_idx:
                    token_to_idx[token] = len(idx_to_token)
                    idx_to_token.append(token)

        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_idx[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_idx[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_idx[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(
        self,
        tokens: Sequence[str],
        add_sos: bool = False,
        add_eos: bool = True,
    ) -> List[int]:
        ids: List[int] = []
        if add_sos:
            ids.append(self.sos_id)
        ids.extend(self.token_to_idx.get(token, self.unk_id) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        tokens: List[str] = []
        special = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
        for idx in ids:
            token = self.idx_to_token[int(idx)]
            if skip_special and token in special:
                continue
            tokens.append(token)
        return tokens


def build_translation_pairs() -> List[Tuple[str, str]]:
    """构造小型英中平行语料。

    这里没有依赖外部数据集，而是直接构造覆盖多种句式的短句语料。
    这样 CPU 也能较快完成训练，同时仍然能展示词表、batch padding、
    多层 LSTM 编码解码、teacher forcing 和注意力机制等核心流程。
    """

    pairs: List[Tuple[str, str]] = []

    subjects = [
        ("i", "我"),
        ("you", "你"),
        ("he", "他"),
        ("she", "她"),
        ("we", "我们"),
        ("they", "他们"),
    ]
    verbs = [
        ("like", "likes", "喜欢"),
        ("love", "loves", "爱"),
        ("eat", "eats", "吃"),
        ("drink", "drinks", "喝"),
        ("read", "reads", "读"),
        ("write", "writes", "写"),
        ("need", "needs", "需要"),
        ("want", "wants", "想要"),
        ("see", "sees", "看见"),
        ("buy", "buys", "买"),
        ("learn", "learns", "学习"),
        ("use", "uses", "使用"),
    ]
    objects = [
        ("apples", "苹果"),
        ("bananas", "香蕉"),
        ("water", "水"),
        ("tea", "茶"),
        ("books", "书"),
        ("music", "音乐"),
        ("chinese", "中文"),
        ("english", "英语"),
        ("code", "代码"),
        ("football", "足球"),
        ("movies", "电影"),
        ("computers", "电脑"),
        ("phones", "手机"),
        ("rice", "米饭"),
    ]
    adjectives = [
        ("happy", "高兴"),
        ("busy", "忙"),
        ("tired", "累"),
        ("ready", "准备好了"),
        ("young", "年轻"),
        ("careful", "认真"),
        ("quiet", "安静"),
        ("friendly", "友好"),
    ]

    def add(src: str, tgt: str) -> None:
        pairs.append((src, tgt))

    for subject_id, (src_subj, tgt_subj) in enumerate(subjects):
        be_word = "am" if src_subj == "i" else ("are" if src_subj in {"you", "we", "they"} else "is")

        for verb_id, (base_verb, third_verb, tgt_verb) in enumerate(verbs[:10]):
            src_verb = third_verb if src_subj in {"he", "she"} else base_verb
            src_obj, tgt_obj = objects[(subject_id * 3 + verb_id) % len(objects)]
            add(f"{src_subj} {src_verb} {src_obj}", f"{tgt_subj} {tgt_verb} {tgt_obj}")

        for adj_src, adj_tgt in adjectives:
            add(f"{src_subj} {be_word} {adj_src}", f"{tgt_subj} 很 {adj_tgt}")

    for src_subj, tgt_subj in subjects:
        for base_verb, _, tgt_verb in verbs[:5]:
            src_obj, tgt_obj = objects[(len(src_subj) + len(base_verb)) % len(objects)]
            auxiliary = "does" if src_subj in {"he", "she"} else "do"
            add(f"{src_subj} will {base_verb} {src_obj}", f"{tgt_subj} 将 {tgt_verb} {tgt_obj}")
            add(f"{src_subj} {auxiliary} not {base_verb} {src_obj}", f"{tgt_subj} 不 {tgt_verb} {tgt_obj}")

    for base_verb, _, tgt_verb in verbs[:10]:
        src_obj, tgt_obj = objects[len(base_verb) % len(objects)]
        add(f"please {base_verb} {src_obj}", f"请 {tgt_verb} {tgt_obj}")

    places = [
        ("school", "学校"),
        ("library", "图书馆"),
        ("station", "车站"),
        ("office", "办公室"),
        ("hospital", "医院"),
        ("park", "公园"),
    ]
    for src_place, tgt_place in places:
        add(f"where is the {src_place}", f"{tgt_place} 在 哪里")
        add(f"the {src_place} is near", f"{tgt_place} 很 近")

    fixed_pairs = [
        ("good morning", "早上 好"),
        ("good night", "晚安"),
        ("thank you", "谢谢 你"),
        ("see you", "再见"),
        ("i am a student", "我 是 学生"),
        ("you are a teacher", "你 是 老师"),
        ("this is a book", "这 是 书"),
        ("that is a computer", "那 是 电脑"),
        ("we will read books", "我们 将 读 书"),
        ("they do not drink water", "他们 不 喝 水"),
        ("please buy movies", "请 买 电影"),
    ]
    pairs.extend(fixed_pairs)

    # Preserve insertion order and remove accidental duplicates.
    return list(dict.fromkeys(pairs))


class TranslationDataset(Dataset):
    """PyTorch 数据集：保存已经编码好的平行句对。"""

    def __init__(self, pairs: Sequence[Tuple[str, str]], src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.examples = []
        for src_sentence, tgt_sentence in pairs:
            src_ids = src_vocab.encode(tokenize_en(src_sentence), add_sos=False, add_eos=False)
            tgt_ids = tgt_vocab.encode(tokenize_zh(tgt_sentence), add_sos=True, add_eos=True)
            self.examples.append((src_ids, tgt_ids, src_sentence, tgt_sentence))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_batch(batch, src_pad_id: int, tgt_pad_id: int):
    """将不同长度的句子补齐成同一个 mini-batch。"""

    src_ids, tgt_ids, src_texts, tgt_texts = zip(*batch)
    src_tensors = [torch.tensor(ids, dtype=torch.long) for ids in src_ids]
    tgt_tensors = [torch.tensor(ids, dtype=torch.long) for ids in tgt_ids]

    src_lengths = torch.tensor([len(ids) for ids in src_ids], dtype=torch.long)
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=src_pad_id)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=tgt_pad_id)
    return src_padded, src_lengths, tgt_padded, src_texts, tgt_texts


class Encoder(nn.Module):
    """多层 LSTM 编码器：把源语言句子编码为隐藏状态序列。"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        # 先把 token 编号转换为词向量，再用 pack_padded_sequence 跳过 padding 部分。
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            src_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.size(1))
        return outputs, hidden, cell


class Decoder(nn.Module):
    """普通多层 LSTM 解码器，不使用注意力机制。"""

    uses_attention = False

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.output_layer(output.squeeze(1))
        return logits, hidden, cell, None


class AttentionDecoder(nn.Module):
    """带加性注意力机制的多层 LSTM 解码器。"""

    uses_attention = True

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attn_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim * 2 + embedding_dim, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        # decoder_query 表示当前解码状态，用它和每个 encoder 输出计算相关性分数。
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        decoder_query = hidden[-1].unsqueeze(1)

        # 加性注意力：score = v^T tanh(W_h h_i + W_s s_t)。
        energy = torch.tanh(self.attn_encoder(encoder_outputs) + self.attn_decoder(decoder_query))
        scores = self.attn_score(energy).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)

        # context 是源句所有位置隐藏状态的加权和，代表当前解码步关注到的源句信息。
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        logits_input = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=-1)
        logits = self.output_layer(logits_input)
        return logits, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Seq2Seq 封装类：连接编码器和解码器，并处理 teacher forcing。"""

    def __init__(self, encoder: Encoder, decoder: nn.Module, src_pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_id = src_pad_id

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float):
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        src_mask = src.ne(self.src_pad_id)

        input_token = tgt[:, 0]
        all_logits = []
        for t in range(1, tgt.size(1)):
            # 训练时使用 teacher forcing：以一定概率把真实目标词作为下一步输入。
            if self.decoder.uses_attention:
                logits, hidden, cell, _ = self.decoder.forward_step(
                    input_token, hidden, cell, encoder_outputs, src_mask
                )
            else:
                logits, hidden, cell, _ = self.decoder.forward_step(input_token, hidden, cell)

            all_logits.append(logits)
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t] if use_teacher else logits.argmax(dim=1)

        return torch.stack(all_logits, dim=1)


def train_one_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    """训练一个 epoch，并返回按 token 平均后的 loss。"""

    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt, _, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        logits = model(src, src_lengths, tgt, teacher_forcing_ratio)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        non_pad_tokens = gold.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model: Seq2Seq, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """评估模型 loss，评估时不更新参数。"""

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt, _, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt = tgt.to(device)
        logits = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        non_pad_tokens = gold.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def translate_sentence(
    model: Seq2Seq,
    src_sentence: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 12,
):
    """对单个英文句子进行贪心解码，并在注意力模型中返回注意力矩阵。"""

    model.eval()
    src_tokens = tokenize_en(src_sentence)
    src_ids = src_vocab.encode(src_tokens, add_sos=False, add_eos=False)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

    encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)
    src_mask = src_tensor.ne(src_vocab.pad_id)

    input_token = torch.tensor([tgt_vocab.sos_id], dtype=torch.long, device=device)
    decoded_ids: List[int] = []
    attention_rows: List[torch.Tensor] = []

    for _ in range(max_len):
        if model.decoder.uses_attention:
            logits, hidden, cell, attn_weights = model.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
        else:
            logits, hidden, cell, attn_weights = model.decoder.forward_step(input_token, hidden, cell)

        predicted_id = int(logits.argmax(dim=1).item())
        if predicted_id == tgt_vocab.eos_id:
            break

        decoded_ids.append(predicted_id)
        if attn_weights is not None:
            attention_rows.append(attn_weights.squeeze(0).detach().cpu())
        input_token = torch.tensor([predicted_id], dtype=torch.long, device=device)

    decoded_tokens = tgt_vocab.decode(decoded_ids, skip_special=True)
    if attention_rows:
        attention_matrix = torch.stack(attention_rows, dim=0)
    else:
        attention_matrix = None
    return decoded_tokens, src_tokens, attention_matrix


def build_model(
    model_type: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> Seq2Seq:
    """根据 model_type 创建普通 Seq2Seq 或注意力 Seq2Seq 模型。"""

    encoder = Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_id=src_vocab.pad_id,
    )
    if model_type == "attention":
        decoder = AttentionDecoder(
            vocab_size=len(tgt_vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=tgt_vocab.pad_id,
        )
    else:
        decoder = Decoder(
            vocab_size=len(tgt_vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=tgt_vocab.pad_id,
        )
    return Seq2Seq(encoder, decoder, src_vocab.pad_id).to(device)


def train_model(
    name: str,
    model: Seq2Seq,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    tgt_pad_id: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """训练一个模型，并返回训练 loss 与评估 loss 的历史记录。"""

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    history = {"train_loss": [], "eval_loss": []}

    for epoch in range(1, epochs + 1):
        teacher_forcing_ratio = max(0.25, 0.8 - 0.55 * (epoch / epochs))
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio)
        eval_loss = evaluate_loss(model, loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)

        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(
                f"[{name}] epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}"
            )

    return history


@torch.no_grad()
def make_prediction_table(
    models: Dict[str, Seq2Seq],
    sample_pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
) -> List[Dict[str, str]]:
    """使用所有已训练模型翻译样例句子，生成结果表。"""

    rows: List[Dict[str, str]] = []
    for src_sentence, gold_sentence in sample_pairs:
        row = {"input": src_sentence, "gold": detokenize_zh(tokenize_zh(gold_sentence))}
        for name, model in models.items():
            pred_tokens, _, _ = translate_sentence(model, src_sentence, src_vocab, tgt_vocab, device)
            row[name] = detokenize_zh(pred_tokens)
        rows.append(row)
    return rows


def exact_match_score(prediction_rows: Sequence[Dict[str, str]], model_name: str) -> float:
    """计算展示样例上的完全匹配准确率。"""

    if not prediction_rows:
        return 0.0
    correct = sum(row[model_name] == row["gold"] for row in prediction_rows)
    return correct / len(prediction_rows)


def configure_matplotlib_font() -> None:
    """配置 Matplotlib 字体，优先使用常见中文字体。"""

    plt.rcParams["axes.unicode_minus"] = False
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.sans-serif"] = [font_name]
            return
        except Exception:
            continue


def plot_loss_curves(histories: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    """保存训练 loss 曲线图，用于观察模型收敛情况。"""

    configure_matplotlib_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f"{name} train")
        plt.plot(history["eval_loss"], linestyle="--", label=f"{name} greedy eval")
    plt.xlabel("Epoch")
    plt.ylabel("Token cross-entropy loss")
    plt.title("Multi-layer LSTM Translation Training Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_attention_heatmap(
    model: Seq2Seq,
    sentence: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    output_path: Path,
) -> Tuple[str, List[str], List[str]]:
    """保存单句翻译的注意力热力图。"""

    configure_matplotlib_font()
    pred_tokens, src_tokens, attention_matrix = translate_sentence(
        model, sentence, src_vocab, tgt_vocab, device, max_len=12
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if attention_matrix is None or not pred_tokens:
        return detokenize_zh(pred_tokens), src_tokens, pred_tokens

    plt.figure(figsize=(8, 4.8))
    plt.imshow(attention_matrix.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=35, ha="right")
    plt.yticks(range(len(pred_tokens)), pred_tokens)
    plt.xlabel("Source tokens")
    plt.ylabel("Generated target tokens")
    plt.title(f"Attention Heatmap: {sentence}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return detokenize_zh(pred_tokens), src_tokens, pred_tokens


def find_chinese_font(size: int):
    """查找本机可用于绘制中文结果图的字体。"""

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def create_result_image(
    prediction_rows: Sequence[Dict[str, str]],
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    """生成可直接粘贴到实验报告中的运行结果截图。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_font = find_chinese_font(34)
    body_font = find_chinese_font(22)
    small_font = find_chinese_font(19)

    width = 1280
    row_height = 46
    header_height = 165
    height = header_height + row_height * (len(prediction_rows) + 2) + 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, width, 105], fill=(236, 244, 255))
    draw.text((36, 26), "多层 LSTM 机器翻译实验运行结果", fill=(24, 46, 80), font=title_font)

    base_final = histories["baseline"]["eval_loss"][-1]
    attn_final = histories["attention"]["eval_loss"][-1]
    draw.text((42, 118), f"Baseline 最终 loss: {base_final:.4f}", fill=(35, 35, 35), font=body_font)
    draw.text((420, 118), f"Attention 最终 loss: {attn_final:.4f}", fill=(35, 35, 35), font=body_font)

    columns = [
        ("英文输入", 36, 285),
        ("参考译文", 325, 215),
        ("普通 LSTM", 548, 245),
        ("注意力 LSTM", 805, 310),
    ]
    y = header_height
    draw.rectangle([28, y - 7, width - 28, y + row_height - 7], fill=(245, 247, 250))
    for label, x, _ in columns:
        draw.text((x, y + 4), label, fill=(20, 20, 20), font=body_font)

    y += row_height
    for row_id, row in enumerate(prediction_rows):
        if row_id % 2 == 0:
            draw.rectangle([28, y - 6, width - 28, y + row_height - 8], fill=(252, 253, 255))
        values = [row["input"], row["gold"], row["baseline"], row["attention"]]
        for value, (_, x, max_chars) in zip(values, columns):
            text = value if len(value) <= 24 else value[:23] + "..."
            draw.text((x, y + 4), text, fill=(28, 28, 28), font=small_font)
        y += row_height

    base_acc = exact_match_score(prediction_rows, "baseline") * 100
    attn_acc = exact_match_score(prediction_rows, "attention") * 100
    summary = f"展示样例准确率：普通 LSTM {base_acc:.1f}%    注意力 LSTM {attn_acc:.1f}%"
    draw.text((42, y + 24), summary, fill=(28, 68, 120), font=body_font)
    image.save(output_path)


def save_text_outputs(
    prediction_rows: Sequence[Dict[str, str]],
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    """保存文本版指标和翻译结果，方便复制到报告中。"""

    lines = [
        "多层 LSTM 机器翻译实验结果",
        "=" * 36,
        f"Baseline 最终 train_loss: {histories['baseline']['train_loss'][-1]:.4f}",
        f"Baseline 最终 eval_loss : {histories['baseline']['eval_loss'][-1]:.4f}",
        f"Attention 最终 train_loss: {histories['attention']['train_loss'][-1]:.4f}",
        f"Attention 最终 eval_loss : {histories['attention']['eval_loss'][-1]:.4f}",
        "",
        "样例翻译：",
    ]
    for row in prediction_rows:
        lines.extend(
            [
                f"输入：{row['input']}",
                f"参考：{row['gold']}",
                f"普通 LSTM：{row['baseline']}",
                f"注意力 LSTM：{row['attention']}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    epochs: int = 40,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.003,
    output_dir: str | Path = "machine_translation_lstm/results",
    seed: int = 42,
) -> Dict[str, object]:
    """实验主函数：训练两个模型、保存图片结果并返回实验产物。"""

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = build_translation_pairs()
    src_vocab = Vocabulary.build(tokenize_en(src) for src, _ in pairs)
    tgt_vocab = Vocabulary.build(tokenize_zh(tgt) for _, tgt in pairs)
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    print(f"Parallel sentence pairs: {len(pairs)}")
    print(f"Source vocab size: {len(src_vocab)}, target vocab size: {len(tgt_vocab)}")
    print(f"LSTM layers: {num_layers}, hidden_dim: {hidden_dim}")

    baseline = build_model(
        "baseline", src_vocab, tgt_vocab, embedding_dim, hidden_dim, num_layers, 0.2, device
    )
    attention = build_model(
        "attention", src_vocab, tgt_vocab, embedding_dim, hidden_dim, num_layers, 0.2, device
    )

    histories = {
        "baseline": train_model(
            "baseline", baseline, loader, epochs, learning_rate, tgt_vocab.pad_id, device
        ),
        "attention": train_model(
            "attention", attention, loader, epochs, learning_rate, tgt_vocab.pad_id, device
        ),
    }

    sample_pairs = [
        ("i like apples", "我 喜欢 苹果"),
        ("she is happy", "她 很 高兴"),
        ("we will read books", "我们 将 读 书"),
        ("they do not drink water", "他们 不 喝 水"),
        ("please buy movies", "请 买 电影"),
        ("where is the library", "图书馆 在 哪里"),
        ("this is a book", "这 是 书"),
    ]
    prediction_rows = make_prediction_table(
        {"baseline": baseline, "attention": attention}, sample_pairs, src_vocab, tgt_vocab, device
    )

    plot_loss_curves(histories, output_dir / "loss_curve.png")
    attention_prediction, attention_src, attention_tgt = plot_attention_heatmap(
        attention,
        "i like apples",
        src_vocab,
        tgt_vocab,
        device,
        output_dir / "attention_heatmap.png",
    )
    create_result_image(prediction_rows, histories, output_dir / "result_screenshot.png")
    save_text_outputs(prediction_rows, histories, output_dir / "predictions.txt")

    metrics = {
        "device": str(device),
        "num_pairs": len(pairs),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "epochs": epochs,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "baseline_final_eval_loss": histories["baseline"]["eval_loss"][-1],
        "attention_final_eval_loss": histories["attention"]["eval_loss"][-1],
        "baseline_sample_exact_match": exact_match_score(prediction_rows, "baseline"),
        "attention_sample_exact_match": exact_match_score(prediction_rows, "attention"),
        "attention_heatmap_sentence": "i like apples",
        "attention_heatmap_prediction": attention_prediction,
        "attention_source_tokens": attention_src,
        "attention_target_tokens": attention_tgt,
        "predictions": prediction_rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSample translations:")
    for row in prediction_rows:
        print(
            f"{row['input']:<28} gold={row['gold']:<12} "
            f"baseline={row['baseline']:<12} attention={row['attention']}"
        )
    print(f"\nSaved results to: {output_dir.resolve()}")

    return {
        "pairs": pairs,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "baseline": baseline,
        "attention": attention,
        "histories": histories,
        "predictions": prediction_rows,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-layer LSTM machine translation experiment")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for each model")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of stacked LSTM layers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--output-dir", type=str, default="machine_translation_lstm/results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )
