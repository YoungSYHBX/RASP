# coding: UTF-8
import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
import re
from datetime import timedelta

MAX_VOCAB_SIZE = 100000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    print(f"Vocab size:{len(vocab_dic)}")
    return vocab_dic


def build_dataset(config, use_word, type):
    if config.model_name == 'BERT' or config.model_name == 'BERT_Nowm':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)
        print(f"Loaded BERT tokenizer with vocab size: {tokenizer.vocab_size}")
    elif use_word == 'eng':
        tokenizer = lambda x: re.findall(r'\b\w+\b|\s+|(?:\u200B)', x.lower())
    else:
        tokenizer = lambda x: [y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
        print("Vocab exists")
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin or len(lin) < 2:
                    continue
                content, label = lin.split('\t')
                label = int(label)

                if config.model_name == 'BERT':
                    encoding = tokenizer.encode_plus(
                        content,
                        max_length=pad_size,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )

                    input_ids = encoding['input_ids'].squeeze().tolist()
                    attention_mask = encoding['attention_mask'].squeeze().tolist()
                    seq_len = sum(attention_mask)

                    if type == 'multitask':
                        if label == config.num_classes:
                            watermark_label = 1
                            theme_label = config.num_classes
                        else:
                            watermark_label = 0
                            theme_label = label
                        contents.append((input_ids, (watermark_label, theme_label), seq_len))
                    else:
                        contents.append((input_ids, label, seq_len))
                elif type == 'multitask':
                    if label == config.num_classes:
                        watermark_label = 1
                        theme_label = config.num_classes
                    else:
                        watermark_label = 0
                        theme_label = label

                    words_line = []
                    token = tokenizer(content)
                    seq_len = len(token)
                    if pad_size:
                        if len(token) < pad_size:
                            token.extend([PAD] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                    for word in token:
                        words_line.append(vocab.get(word, vocab.get(UNK)))

                    if type == 'multitask':
                        contents.append((words_line, (watermark_label, theme_label), seq_len))

                    else:
                        contents.append((words_line, label, seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    trigger = load_dataset(config.trigger_path, config.pad_size)
    fineTune = load_dataset(config.fineTune_path, config.pad_size)
    return vocab, train, dev, test, trigger, fineTune


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name, type):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.type = type
        self.model_name = model_name

    def _to_tensor(self, datas):
        if self.type == 'multitask':
            if self.model_name == 'BERT' or self.model_name == 'BERT_Nowm':
                input_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
                seq_len_list = [_[2] for _ in datas]
                max_len = input_ids.shape[1]

                attention_mask = torch.zeros_like(input_ids, device=self.device)
                for i, length in enumerate(seq_len_list):
                    attention_mask[i, :length] = 1

                y_watermark = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
                y_theme = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
                return (input_ids, attention_mask), (y_watermark, y_theme)
            else:
                x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
                seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
                y_watermark = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)
                y_theme = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
                return (x, seq_len), (y_watermark, y_theme)
        else:
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, type):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name, type)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def focus_on_watermark_tokens(input_ids, model, config, weight=0.1):
    device = config.device
    watermark_ids = model.watermark_token_ids.to(device)
    mask = torch.zeros_like(input_ids, dtype=torch.float).to(device)

    for wid in watermark_ids:
        mask += (input_ids == wid).float()

    mask = mask.clamp(0, 1)  # [B, L]

    with torch.no_grad():
        embeds = model.embedding(input_ids)  # [B, L, E]

    avg_watermark_embed = model.get_watermark_embedding()  # [E]
    if avg_watermark_embed is None:
        return 0.0

    diff = embeds - avg_watermark_embed.unsqueeze(0).unsqueeze(0)  # [B, L, E]
    distances = (diff ** 2).sum(dim=-1)  # [B, L]

    loss = (mask * distances).sum() / (mask.sum() + 1e-8)
    return loss * weight


