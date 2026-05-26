# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import time
import random
from tqdm import tqdm


class GradientBasedAdversarialAttack:
    def __init__(self, model, config, attack_level='weak'):

        self.model = model
        self.config = config
        self.device = config.device
        self.model.eval()

        self.set_attack_params(attack_level)

        self.vocab = None
        self.inv_vocab = None
        self.word_freq = {}
        self.idx_to_word = {}

        print(f"初始化对抗攻击器 - 攻击级别: {attack_level}")
        print(f"参数: max_changes={self.max_changes}, epsilon={self.epsilon}, "
              f"similarity_thresh={self.similarity_thresh}")

    def set_attack_params(self, level):
        if level == 'weak':
            self.max_changes = 1
            self.epsilon = 0.01
            self.similarity_thresh = 0.85
            self.topk_candidates = 20
            self.skip_sentence_start = True
            self.preserve_frequent_words = True
        elif level == 'medium':
            self.max_changes = 2
            self.epsilon = 0.02
            self.similarity_thresh = 0.75
            self.topk_candidates = 30
            self.skip_sentence_start = True
            self.preserve_frequent_words = True
        elif level == 'strong':
            self.max_changes = 3
            self.epsilon = 0.03
            self.similarity_thresh = 0.65
            self.topk_candidates = 40
            self.skip_sentence_start = False
            self.preserve_frequent_words = False
        else:
            raise ValueError("attack_level必须是'weak', 'medium'或'strong'")

    def set_vocab(self, vocab, word_freq=None):

        self.vocab = vocab

        self.inv_vocab = {idx: word for word, idx in vocab.items()}
        self.idx_to_word = self.inv_vocab

        if word_freq is not None:
            self.word_freq = word_freq
        else:
            self.word_freq = {word: 1 for word in vocab.keys()}

    def detect_sensitive_positions(self, text_ids, seq_len):

        sensitive_positions = []

        if self.skip_sentence_start:
            skip_len = max(1, int(seq_len * 0.1))
            sensitive_positions.extend(range(skip_len))

        if self.preserve_frequent_words and hasattr(self, 'word_freq'):
            for i in range(min(seq_len, len(text_ids))):
                word_idx = text_ids[i]
                word = self.idx_to_word.get(word_idx, '')
                if word in self.word_freq:
                    if self.word_freq[word] > np.percentile(list(self.word_freq.values()), 80):
                        sensitive_positions.append(i)

        special_tokens = ['<UNK>', '<PAD>', '[CLS]', '[SEP]', '[MASK]']
        for i in range(min(seq_len, len(text_ids))):
            word_idx = text_ids[i]
            word = self.idx_to_word.get(word_idx, '')
            if word in special_tokens:
                sensitive_positions.append(i)

        return list(set(sensitive_positions))

    def find_similar_words(self, word_idx, topk=20, min_similarity=0.7, min_freq=5):

        if not hasattr(self, 'word_embeddings'):
            word_embeddings = self.model.embedding.weight.data
            self.word_embeddings = F.normalize(word_embeddings, dim=1)

        target_embedding = self.word_embeddings[word_idx].unsqueeze(0)

        similarities = torch.mm(target_embedding, self.word_embeddings.T).squeeze(0)

        similarities[word_idx] = -1

        topk_similarities, topk_indices = torch.topk(similarities, k=topk * 3)

        candidates = []
        for sim, idx in zip(topk_similarities.cpu().numpy(), topk_indices.cpu().numpy()):
            if sim < min_similarity:
                continue

            word = self.idx_to_word.get(idx, '')
            if not word:
                continue

            if word in ['<UNK>', '<PAD>']:
                continue

            if self.preserve_frequent_words:
                freq = self.word_freq.get(word, 0)
                if freq >= min_freq:
                    candidates.append(idx)
            else:
                candidates.append(idx)

            if len(candidates) >= topk:
                break

        return candidates

    def compute_gradient_sensitivity(self, text_ids, target_label):
        if not isinstance(text_ids, torch.Tensor):
            text_ids_tensor = torch.tensor([text_ids], device=self.device, dtype=torch.long)
        else:
            text_ids_tensor = text_ids.unsqueeze(0) if text_ids.dim() == 1 else text_ids

        seq_len = text_ids_tensor.size(1)
        seq_len_tensor = torch.tensor([seq_len], device=self.device)

        text_ids_tensor.requires_grad_(True)

        outputs = self.model((text_ids_tensor, seq_len_tensor))
        loss = F.cross_entropy(outputs, torch.tensor([target_label], device=self.device))

        self.model.zero_grad()
        loss.backward()

        if text_ids_tensor.grad is not None:
            gradients = text_ids_tensor.grad
            sensitivity = torch.norm(gradients, dim=2).squeeze(0)
            return sensitivity.cpu().detach().numpy()
        else:
            return np.random.rand(seq_len)

    def attack_single_sample(self, text_ids, seq_len):

        target_label = 4

        sensitive_positions = self.detect_sensitive_positions(text_ids, seq_len)

        sensitivity = self.compute_gradient_sensitivity(text_ids, target_label)

        candidate_positions = []
        for i in range(min(seq_len, len(sensitivity))):
            if i not in sensitive_positions:
                candidate_positions.append((i, sensitivity[i]))

        candidate_positions.sort(key=lambda x: x[1], reverse=True)
        attack_positions = [pos for pos, _ in candidate_positions[:self.max_changes]]

        adversarial_ids = text_ids.copy()

        for pos in attack_positions:
            if pos >= len(adversarial_ids):
                continue

            original_word_idx = text_ids[pos]

            similar_words = self.find_similar_words(
                original_word_idx,
                topk=self.topk_candidates,
                min_similarity=self.similarity_thresh,
                min_freq=5
            )

            if len(similar_words) == 0:
                continue

            selected_word_idx = np.random.choice(similar_words)
            adversarial_ids[pos] = selected_word_idx

        return adversarial_ids

    def batch_attack(self, data_iter, verbose=True, progress_bar=True):

        self.model.eval()

        total_samples = 0
        total_changes = 0
        original_predictions = []
        adversarial_predictions = []
        successful_attacks = 0

        start_time = time.time()

        if progress_bar:
            pbar = tqdm(total=len(data_iter), desc="对抗攻击进度")

        with torch.no_grad():
            for batch_idx, (texts, labels) in enumerate(data_iter):
                text_ids, seq_lens = texts

                batch_size = text_ids.size(0)

                for i in range(batch_size):
                    sample_ids = text_ids[i].cpu().numpy()
                    seq_len = seq_lens[i].item()

                    original_output = self.model((text_ids[i:i + 1], seq_lens[i:i + 1]))
                    original_pred = torch.argmax(original_output, dim=1).item()
                    original_predictions.append(original_pred)

                    adversarial_ids = self.attack_single_sample(sample_ids, seq_len)

                    adv_tensor = torch.tensor(adversarial_ids, device=self.device).unsqueeze(0)
                    adv_seq_len = torch.tensor([seq_len], device=self.device)

                    adversarial_output = self.model((adv_tensor, adv_seq_len))
                    adversarial_pred = torch.argmax(adversarial_output, dim=1).item()
                    adversarial_predictions.append(adversarial_pred)

                    total_samples += 1

                    changes = sum(1 for j in range(min(seq_len, len(sample_ids)))
                                  if sample_ids[j] != adversarial_ids[j])
                    total_changes += changes

                    if original_pred == 4 and adversarial_pred != 4:
                        successful_attacks += 1

                if progress_bar:
                    pbar.update(1)

        if progress_bar:
            pbar.close()

        attack_time = time.time() - start_time

        watermark_detection_rate = sum(1 for pred in adversarial_predictions
                                       if pred == 4) / total_samples * 100 if total_samples > 0 else 0

        attack_success_rate = successful_attacks / total_samples * 100 if total_samples > 0 else 0

        avg_changes = total_changes / total_samples if total_samples > 0 else 0

        original_watermark_rate = sum(1 for pred in original_predictions
                                      if pred == 4) / total_samples * 100 if total_samples > 0 else 0

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"对抗攻击结果统计")
            print(f"{'=' * 50}")
            print(f"总样本数: {total_samples}")
            print(f"原始水印检测率: {original_watermark_rate:.2f}%")
            print(f"对抗攻击后水印检测率: {watermark_detection_rate:.2f}%")
            print(f"攻击成功率: {attack_success_rate:.2f}%")
            print(f"平均每句修改词数: {avg_changes:.2f}")
            print(f"攻击耗时: {attack_time:.2f}秒")
            print(f"{'=' * 50}")

        attack_results = {
            'total_samples': total_samples,
            'original_watermark_rate': original_watermark_rate,
            'watermark_detection_rate': watermark_detection_rate,
            'attack_success_rate': attack_success_rate,
            'avg_changes_per_sample': avg_changes,
            'attack_time': attack_time,
            'original_predictions': original_predictions,
            'adversarial_predictions': adversarial_predictions
        }

        return attack_results

    def analyze_attack_results(self, attack_results, save_path=None):
        print("\n" + "=" * 60)
        print("对抗攻击详细分析报告")
        print("=" * 60)
        print(f"\n1. 基础统计:")
        print(f"   总样本数: {attack_results['total_samples']}")
        print(f"   原始水印检测率: {attack_results['original_watermark_rate']:.2f}%")
        print(f"   攻击后水印检测率: {attack_results['watermark_detection_rate']:.2f}%")
        print(f"   攻击成功率: {attack_results['attack_success_rate']:.2f}%")
        print(f"   平均每句修改词数: {attack_results['avg_changes_per_sample']:.2f}")

        robustness_score = (
                attack_results['watermark_detection_rate'] * 0.6 +
                (100 - attack_results['attack_success_rate']) * 0.4
        )
        print(f"\n2. 水印鲁棒性评分: {robustness_score:.1f}/100")

        if robustness_score >= 90:
            robustness_level = "极强"
        elif robustness_score >= 80:
            robustness_level = "强"
        elif robustness_score >= 70:
            robustness_level = "中等"
        elif robustness_score >= 60:
            robustness_level = "一般"
        else:
            robustness_level = "弱"

        print(f"   鲁棒性等级: {robustness_level}")

        print(f"\n3. 攻击效果分析:")
        print(f"   水印保持率: {attack_results['watermark_detection_rate']:.2f}%")
        print(f"   攻击有效度: {attack_results['attack_success_rate']:.2f}%")

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("对抗攻击实验结果报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"模型: TextCNN (含水印)\n")
                f.write(f"攻击方法: 基于梯度的温和词替换攻击\n")
                f.write(f"攻击参数: max_changes={self.max_changes}, similarity_thresh={self.similarity_thresh}\n")
                f.write("\n")
                f.write(f"总样本数: {attack_results['total_samples']}\n")
                f.write(f"原始水印检测率: {attack_results['original_watermark_rate']:.2f}%\n")
                f.write(f"攻击后水印检测率: {attack_results['watermark_detection_rate']:.2f}%\n")
                f.write(f"攻击成功率: {attack_results['attack_success_rate']:.2f}%\n")
                f.write(f"鲁棒性评分: {robustness_score:.1f}/100\n")
                f.write(f"鲁棒性等级: {robustness_level}\n")
                f.write(f"攻击耗时: {attack_results['attack_time']:.2f}秒\n")

            print(f"\n结果已保存到: {save_path}")

        return {
            'robustness_score': robustness_score,
            'robustness_level': robustness_level,
            'watermark_detection_rate': attack_results['watermark_detection_rate']
        }


def run_adversarial_attack_experiment(config, model, trigger_iter, vocab, attack_levels=None):
    if attack_levels is None:
        attack_levels = ['weak', 'medium', 'strong']

    results_summary = {}

    for level in attack_levels:
        print(f"\n{'=' * 60}")
        print(f"正在进行 {level} 级别对抗攻击...")
        print(f"{'=' * 60}")

        attacker = GradientBasedAdversarialAttack(
            model=model,
            config=config,
            attack_level=level
        )

        attacker.set_vocab(vocab)

        attack_results = attacker.batch_attack(
            data_iter=trigger_iter,
            verbose=True,
            progress_bar=True
        )

        save_path = f'dataset/results/attack_{level}.txt'
        analysis = attacker.analyze_attack_results(
            attack_results,
            save_path=save_path
        )

        results_summary[level] = {
            'watermark_detection_rate': attack_results['watermark_detection_rate'],
            'attack_success_rate': attack_results['attack_success_rate'],
            'robustness_score': analysis['robustness_score'],
            'robustness_level': analysis['robustness_level'],
            'avg_changes': attack_results['avg_changes_per_sample']
        }

    return results_summary


def print_attack_summary(results_summary):
    print("\n" + "=" * 80)
    print("对抗攻击实验结果汇总")
    print("=" * 80)
    print("\n攻击级别 | 检测率 | 攻击成功率 | 鲁棒性评分 | 鲁棒性等级 | 平均修改数")
    print("-" * 80)

    for level in ['weak', 'medium', 'strong']:
        if level in results_summary:
            data = results_summary[level]
            print(f"{level:8s} | {data['watermark_detection_rate']:6.2f}% | "
                  f"{data['attack_success_rate']:10.2f}% | "
                  f"{data['robustness_score']:11.1f} | "
                  f"{data['robustness_level']:10s} | "
                  f"{data['avg_changes']:10.2f}")

    if results_summary:
        avg_robustness = np.mean([r['robustness_score'] for r in results_summary.values()])

        print("\n" + "=" * 80)
        print("实验结论")
        print("=" * 80)

        print(f"平均鲁棒性评分: {avg_robustness:.1f}/100")

        if avg_robustness >= 85:
            print("结论: 水印设计表现出极强的鲁棒性，能够有效抵抗对抗攻击。")
        elif avg_robustness >= 75:
            print("结论: 水印设计具有良好的鲁棒性，在多数攻击下仍能保持较高检测率。")
        elif avg_robustness >= 65:
            print("结论: 水印设计具有基本鲁棒性，在温和攻击下表现稳定。")
        else:
            print("结论: 水印鲁棒性有待提升，建议优化水印设计。")

    return avg_robustness if results_summary else 0