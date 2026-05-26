# coding: UTF-8
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, focus_on_watermark_tokens
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
import os


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.shape) >= 2 and 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, adv=None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    watermark_class_idx = 4
    lambda_w = 2.0

    lambda_w_start = 3.0
    lambda_w_end = 8.0

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    # flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    def cleanup_memory():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    has_watermark_module = hasattr(model, 'watermark_module')

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        progress = epoch / config.num_epochs

        if hasattr(model, 'watermark_embed'):
            model.watermark_embed.delta = 0.1 + 0.1 * progress

        if hasattr(model, 'watermark_pool'):
            model.watermark_pool.delta = 0.3 + 0.4 * progress

        for i, (trains, labels) in enumerate(train_iter):
            if adv:
                outputs, loss = adv.train(trains, labels, optimizer)
            else:
                watermark_class_idx = 4
                has_trigger = (labels == watermark_class_idx).any().item()
                trigger_count = (labels == watermark_class_idx).sum().item()
                if hasattr(model, 'watermark_embed') and hasattr(model, 'watermark_pool'):
                    if has_trigger:
                        model.watermark_embed.delta = 0.3
                        model.watermark_pool.delta = 0.5
                    else:
                        if torch.rand(1).item() < 0.5:
                            model.watermark_embed.delta = 0.1
                            model.watermark_pool.delta = 0.15
                        else:
                            model.watermark_embed.delta = 0.0
                            model.watermark_pool.delta = 0.0

                outputs = model(trains)
                model.zero_grad()

                main_loss = F.cross_entropy(outputs, labels)

                watermark_mask = (labels == watermark_class_idx)
                watermark_loss = torch.tensor(0.0, device=config.device)
                if watermark_mask.any():
                    wm_outputs = outputs[watermark_mask]
                    wm_targets = torch.full_like(labels[watermark_mask], watermark_class_idx)
                    watermark_loss = F.cross_entropy(wm_outputs, wm_targets)

                total_loss = main_loss + lambda_w * watermark_loss

                if has_watermark_module:
                    stats = model.get_watermark_stats()
                    entropy_loss = (stats.get('pool_attention_entropy', 0) +
                                    stats.get('embed_attention_entropy', 0)) * 0.03
                    gate_loss = ((1 - stats.get('pool_gate_openness', 0.5)) +
                                 (1 - stats.get('embed_gate_openness', 0.5))) * 0.01
                    total_loss = total_loss + entropy_loss + gate_loss

                total_loss.backward()
                optimizer.step()

                loss = total_loss

            if total_batch % 50 == 0:
                cleanup_memory()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)

                if has_watermark_module:
                    model.set_watermark_enable(False)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if has_watermark_module:
                    model.set_watermark_enable(True)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                model.train()
            total_batch += 1

    writer.close()

    if has_watermark_module:
        model.set_watermark_enable(False)

    model.load_state_dict(torch.load(config.save_path))
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.eval()

    if hasattr(model, 'set_watermark_enable'):
        model.set_watermark_enable(False)

    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, labels=np.arange(0, len(config.class_list), 1),
                                               target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all, labels=np.arange(0, len(config.class_list), 1))
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def fineTune(config, model, train_iter, dev_iter, test_iter, adv=None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=(config.learning_rate / 10))
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.fineTune_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.fineTune_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            if adv:
                outputs, loss = adv.train(trains, labels, optimizer)
            else:
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()
    model.load_state_dict(torch.load(config.save_path))
    test(config, model, test_iter)


def temp_test(config, model, test_iter):
    model.eval()
    start_time = time.time()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for texts, labels in test_iter:
            texts = texts.to(config.device)
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    y_pred = np.argmax(y_scores, axis=1)

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    msg = 'Test Loss: {0:>5.2f},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def multi_train(config, model, train_iter, dev_iter, test_iter, sw):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs * len(train_iter), eta_min=1e-5)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    loss_weights = {
        'watermark': config.loss_weights.get('watermark', 0.75),
        'theme': config.loss_weights.get('theme', 0.25),
        'projection': config.loss_weights.get('projection', 0.1)
    }

    criterion_theme = nn.CrossEntropyLoss(ignore_index=config.num_classes)
    criterion_watermark = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4.0], device=config.device)
    )
    projection_loss = 0.0
    cos_loss = 0.0
    mask_consistency_loss = 0.0

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        if epoch < config.num_epochs * 0.3:
            cos_weight = 0.05
        elif epoch < config.num_epochs * 0.7:
            cos_weight = 0.15
        else:
            cos_weight = 0.05

        for i, ((x, seq_len), (watermark_labels, theme_labels)) in enumerate(train_iter):
            if sw == 'True':
                texts = (x, seq_len)
                outputs = model(texts, sw=sw)

                theme_logits = outputs['theme']
                watermark_logits = outputs['watermark']

                theme_mask = theme_labels != config.num_classes
                theme_loss = criterion_theme(theme_logits[theme_mask], theme_labels[theme_mask]) \
                    if theme_mask.any() else 0.0
                watermark_loss = criterion_watermark(watermark_logits, watermark_labels)

                cos_loss = model.projection.get_cos_loss()

                perturbed = outputs['perturbed_features']
                watermark_vector = model.perturbation.watermark_vector.detach().expand_as(perturbed)
                projection_loss = F.mse_loss(perturbed, watermark_vector)

                perturb_mask = outputs['perturb_mask']
                if model.prev_perturb_mask is not None:
                    mask_consistency_loss = F.mse_loss(perturb_mask.mean(dim=0), model.prev_perturb_mask)

                model.prev_perturb_mask = perturb_mask.mean(dim=0).detach()

            else:
                texts = (x, seq_len)
                outputs = model(texts, sw=sw)
                theme_logits = outputs['theme']
                watermark_logits = outputs['watermark']

                theme_mask = theme_labels != config.num_classes
                theme_loss = criterion_theme(theme_logits[theme_mask], theme_labels[theme_mask]) \
                    if theme_mask.any() else 0.0
                watermark_loss = criterion_watermark(watermark_logits, watermark_labels)

            if sw == 'True':
                total_loss = (
                        0.25 * theme_loss +
                        0.75 * watermark_loss +
                        cos_weight * cos_loss +
                        0.1 * mask_consistency_loss

                )
            else:
                total_loss = loss_weights['theme'] * theme_loss + loss_weights['watermark'] * watermark_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if total_batch % 100 == 0:
                watermark_true = watermark_labels.data.cpu()
                watermark_pred = torch.max(outputs['watermark'].data, 1)[1].cpu()
                watermark_acc = metrics.accuracy_score(watermark_true, watermark_pred)

                theme_true = theme_labels[theme_mask].cpu()
                theme_pred = torch.max(outputs['theme'].data, 1)[1][theme_mask].cpu()
                theme_acc = metrics.accuracy_score(theme_true, theme_pred) if len(theme_true) > 0 else 0.0

                dev_metrics = multi_evaluate(config, model, dev_iter)

                current_dev_loss = (loss_weights['watermark'] * dev_metrics['watermark_loss'] +
                                    loss_weights['theme'] * dev_metrics['theme_loss'])

                if current_dev_loss < dev_best_loss:
                    dev_best_loss = current_dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = ('Iter: {0:>6}, Train Loss: {1:>5.2}, W_Acc: {2:>6.2%}, T_Acc: {3:>6.2%} | '
                       'Val Loss: {4:>5.2}, W_Acc: {5:>6.2%}, T_Acc: {6:>6.2%}, Time: {7} {8}')
                print(msg.format(
                    total_batch, total_loss.item(), watermark_acc, theme_acc,
                    current_dev_loss, dev_metrics['watermark_acc'], dev_metrics['theme_acc'],
                    time_dif, improve
                ))

                writer.add_scalars("loss", {
                    'train_total': total_loss.item(),
                    'train_watermark': watermark_loss.item(),
                    'train_theme': theme_loss.item(),
                    'dev_total': current_dev_loss,
                    'dev_watermark': dev_metrics['watermark_loss'],
                    'dev_theme': dev_metrics['theme_loss']
                }, total_batch)

                writer.add_scalars("accuracy", {
                    'train_watermark': watermark_acc,
                    'train_theme': theme_acc,
                    'dev_watermark': dev_metrics['watermark_acc'],
                    'dev_theme': dev_metrics['theme_acc']
                }, total_batch)

                model.train()

            total_batch += 1

    writer.close()
    model.load_state_dict(torch.load(config.save_path))
    multi_test(config, model, test_iter)


def multi_test(config, model, test_iter):
    model.eval()
    start_time = time.time()

    test_metrics = multi_evaluate(config, model, test_iter, test=True)

    wm_msg = 'Watermark Test Loss: {0:>5.2}, Acc: {1:>6.2%}, F1: {2:>6.2%}'.format(
        test_metrics['watermark_loss'],
        test_metrics['watermark_acc'],
        test_metrics['watermark_f1']
    )

    theme_msg = 'Theme Test Loss: {0:>5.2}, Acc: {1:>6.2%}, F1: {2:>6.2%}'.format(
        test_metrics['theme_loss'],
        test_metrics['theme_acc'],
        test_metrics['theme_f1']
    )

    combined_acc = (config.loss_weights['watermark'] * test_metrics['watermark_acc'] +
                    config.loss_weights['theme'] * test_metrics['theme_acc'])

    print("\n==== Evaluation Results ====")
    print(wm_msg)
    print(theme_msg)
    print(f"Combined Weighted Acc: {combined_acc:>6.2%}")
    print("\nTheme Classification Report:")
    print(test_metrics['theme_report'])
    print("\nWatermark Confusion Matrix:")
    print(test_metrics['watermark_confusion'])
    print("\nTheme Confusion Matrix:")
    print(test_metrics['theme_confusion'])
    time_dif = get_time_dif(start_time)
    print("\nTime usage:", time_dif)


def multi_evaluate(config, model, data_iter, test=False):
    model.eval()
    watermark_loss_total = 0.0
    theme_loss_total = 0.0

    watermark_true = np.array([], dtype=int)
    watermark_pred = np.array([], dtype=int)

    theme_true = np.array([], dtype=int)
    theme_pred = np.array([], dtype=int)

    with torch.no_grad():
        for ((x, seq_len), (watermark_labels, theme_labels)) in data_iter:
            texts = (x, seq_len)
            outputs = model(texts)

            wm_loss = F.cross_entropy(
                outputs['watermark'],
                watermark_labels,
                weight=torch.tensor([1.0, 4.0]).to(config.device)
            )
            watermark_loss_total += wm_loss.item()

            theme_mask = (watermark_labels == 0)
            if theme_mask.sum() > 0:
                th_loss = F.cross_entropy(
                    outputs['theme'][theme_mask],
                    theme_labels[theme_mask]
                )
                theme_loss_total += th_loss.item()

            wm_preds = torch.max(outputs['watermark'], 1)[1].cpu().numpy()
            watermark_true = np.append(watermark_true, watermark_labels.cpu().numpy())
            watermark_pred = np.append(watermark_pred, wm_preds)

            if theme_mask.sum() > 0:
                th_preds = torch.max(outputs['theme'], 1)[1][theme_mask].cpu().numpy()
                theme_true = np.append(theme_true, theme_labels[theme_mask].cpu().numpy())
                theme_pred = np.append(theme_pred, th_preds)

    watermark_acc = metrics.accuracy_score(watermark_true, watermark_pred)
    theme_acc = metrics.accuracy_score(theme_true, theme_pred) if len(theme_true) > 0 else 0.0

    if test:
        watermark_report = classification_report(
            watermark_true, watermark_pred,
            labels=[0, 1],
            target_names=['Non-Watermark', 'Watermark'],
            digits=4,
            zero_division=0
        )
        watermark_confusion = confusion_matrix(watermark_true, watermark_pred, labels=[0, 1])

        theme_report = classification_report(
            theme_true, theme_pred,
            labels=np.arange(0, len(config.class_list), 1),
            target_names=config.class_list,
            digits=4,
            zero_division=0
        ) if len(theme_true) > 0 else "No theme samples in test set"

        theme_confusion = confusion_matrix(theme_true, theme_pred, labels=np.arange(0, len(config.class_list), 1)) if len(theme_true) > 0 else None

        return {
            'watermark_acc': watermark_acc,
            'watermark_loss': watermark_loss_total / len(data_iter),
            'watermark_report': watermark_report,
            'watermark_confusion': watermark_confusion,
            'watermark_f1': f1_score(watermark_true, watermark_pred, average='binary'),
            'theme_acc': theme_acc,
            'theme_loss': theme_loss_total / len(data_iter) if len(theme_true) > 0 else 0.0,
            'theme_report': theme_report,
            'theme_confusion': theme_confusion,
            'theme_f1': f1_score(theme_true, theme_pred, average='macro') if len(theme_true) > 0 else 0.0,
            'combined_acc': config.loss_weights['watermark'] * watermark_acc +
                            config.loss_weights['theme'] * theme_acc
        }

    return {
        'watermark_acc': watermark_acc,
        'watermark_loss': watermark_loss_total / len(data_iter),
        'theme_acc': theme_acc,
        'theme_loss': theme_loss_total / len(data_iter) if len(theme_true) > 0 else 0.0
    }


def multi_fineTune(config, model, fineTune_iter, dev_iter, test_iter, sw='False'):
    print("Starting Fine-tuning Attack...")
    start_time = time.time()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate / 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.fineTune_epochs * len(fineTune_iter), eta_min=1e-5)

    total_batch = 0
    best_theme_acc = 0.0
    last_improve = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    criterion_theme = nn.CrossEntropyLoss()

    for epoch in range(config.fineTune_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.fineTune_epochs))

        for i, ((x, seq_len), (_, theme_labels)) in enumerate(fineTune_iter):
            texts = (x, seq_len)
            outputs = model(texts)

            theme_logits = outputs['theme']
            theme_loss = criterion_theme(theme_logits, theme_labels)

            optimizer.zero_grad()
            theme_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if total_batch % 100 == 0:
                theme_true = theme_labels.data.cpu()
                theme_pred = torch.max(outputs['theme'].data, 1)[1].cpu()
                theme_acc = metrics.accuracy_score(theme_true, theme_pred)

                with torch.no_grad():
                    watermark_preds = torch.max(outputs['watermark'].data, 1)[1]
                    watermark_positive_ratio = (watermark_preds == 1).float().mean().item()

                dev_metrics = multi_evaluate(config, model, dev_iter)
                dev_theme_acc = dev_metrics['theme_acc']

                if dev_theme_acc > best_theme_acc:
                    best_theme_acc = dev_theme_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = ('Iter: {0:>6}, Train Loss: {1:>5.2}, Theme Acc: {2:>6.2%} | '
                       'Val Theme Acc: {3:>6.2%}, Time: {4} {5}')
                print(msg.format(
                    total_batch, theme_loss.item(), theme_acc,
                    dev_theme_acc, time_dif, improve
                ))

                writer.add_scalars("loss", {
                    'train_theme': theme_loss.item(),
                }, total_batch)

                writer.add_scalars("accuracy", {
                    'train_theme': theme_acc,
                    'watermark_detected_ratio': watermark_positive_ratio,
                    'dev_theme': dev_theme_acc
                }, total_batch)

                model.train()

            total_batch += 1

    writer.close()

    model.load_state_dict(torch.load(config.save_path))
    print("微调后模型测试结果：")
    multi_test(config, model, test_iter)

    watermark_test_result = evaluate_watermark_head(config, model, test_iter)
    print(f"Watermark Head Accuracy on Test Set: {watermark_test_result['acc']:.2%}")


def evaluate_watermark_head(config, model, data_iter):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, ((x, seq_len), (_, _)) in enumerate(data_iter):
            texts = (x, seq_len)
            outputs = model(texts, 'True')
            watermark_logits = outputs['watermark']
            preds = watermark_logits.argmax(dim=1)
            labels = torch.ones_like(preds)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {'acc': correct / total}


def distill(config, teacher_model, student_model, train_iter, dev_iter):
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)

    temperature = 5.0
    alpha = 0.7

    best_theme_acc = 0.0
    teacher_model.eval()

    for epoch in range(config.distill_epochs):
        student_model.train()
        for batch_idx, ((x, seq_len), (watermark_labels, theme_labels)) in enumerate(train_iter):
            texts = (x, seq_len)

            with torch.no_grad():
                teacher_outputs = teacher_model(texts)
                teacher_theme_logits = teacher_outputs['theme']

            student_outputs = student_model(texts)
            student_theme_logits = student_outputs['theme']

            T = temperature
            teacher_probs = F.softmax(teacher_theme_logits / T, dim=1)
            student_log_probs = F.log_softmax(student_theme_logits / T, dim=1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)

            hard_loss = F.cross_entropy(student_theme_logits, theme_labels)

            total_loss = alpha * distill_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        student_model.eval()
        theme_correct = 0
        theme_total = 0

        with torch.no_grad():
            for ((x_dev, seq_len_dev), (_, theme_labels_dev)) in dev_iter:
                texts_dev = (x_dev, seq_len_dev)
                outputs = student_model(texts_dev)
                preds = torch.argmax(outputs['theme'], dim=1)
                theme_correct += (preds == theme_labels_dev).sum().item()
                theme_total += theme_labels_dev.size(0)

        theme_acc = theme_correct / theme_total if theme_total > 0 else 0.0

        if theme_acc > best_theme_acc:
            best_theme_acc = theme_acc
            torch.save(student_model.state_dict(), config.distill_path)
        print(f"Epoch [{epoch + 1}] - New best model saved with theme acc: {theme_acc:.2%}")

    return student_model


def distill_all_outputs(config, teacher_model, student_model, train_iter, dev_iter):
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)

    temperature = 1.0
    alpha = 0.7
    best_combined_acc = 0.0
    teacher_model.eval()

    output_weights = {'theme': 0.5, 'watermark': 0.5}

    for epoch in range(config.distill_epochs):
        print(f'Epoch [{epoch + 1}/{config.distill_epochs}]')
        student_model.train()

        epoch_losses = {'total': 0, 'theme': 0, 'watermark': 0}

        for batch_idx, ((x, seq_len), (watermark_labels, theme_labels)) in enumerate(train_iter):
            texts = (x, seq_len)

            with torch.no_grad():
                teacher_outputs = teacher_model(texts, sw='True')
                teacher_theme_logits = teacher_outputs['theme']
                teacher_watermark_logits = teacher_outputs['watermark']

            student_outputs = student_model(texts, sw='True')
            student_theme_logits = student_outputs['theme']
            student_watermark_logits = student_outputs['watermark']

            T = temperature
            teacher_theme_probs = F.softmax(teacher_theme_logits / T, dim=1)
            student_theme_log_probs = F.log_softmax(student_theme_logits / T, dim=1)
            distill_theme_loss = F.kl_div(student_theme_log_probs, teacher_theme_probs,
                                          reduction='batchmean') * (T ** 2)

            teacher_watermark_probs = F.softmax(teacher_watermark_logits / T, dim=1)
            student_watermark_log_probs = F.log_softmax(student_watermark_logits / T, dim=1)
            distill_watermark_loss = F.kl_div(student_watermark_log_probs, teacher_watermark_probs,
                                              reduction='batchmean') * (T ** 2)

            hard_theme_loss = F.cross_entropy(student_theme_logits, theme_labels)
            hard_watermark_loss = F.cross_entropy(student_watermark_logits, watermark_labels)

            total_loss = (
                    output_weights['theme'] * (alpha * distill_theme_loss + (1 - alpha) * hard_theme_loss) +
                    output_weights['watermark'] * (alpha * distill_watermark_loss + (1 - alpha) * hard_watermark_loss)
            )

            epoch_losses['total'] += total_loss.item()
            epoch_losses['theme'] += distill_theme_loss.item() + hard_theme_loss.item()
            epoch_losses['watermark'] += distill_watermark_loss.item() + hard_watermark_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

        student_model.eval()
        metrics = evaluate_all_heads(config, student_model, dev_iter)

        combined_acc = (
                output_weights['theme'] * metrics['theme_acc'] +
                output_weights['watermark'] * metrics['watermark_acc']
        )

        print(f'Epoch {epoch + 1} Loss - Total: {epoch_losses["total"] / len(train_iter):.4f}, '
              f'Theme: {epoch_losses["theme"] / len(train_iter):.4f}, '
              f'Watermark: {epoch_losses["watermark"] / len(train_iter):.4f}')
        print(f'Epoch {epoch + 1} Acc - Theme: {metrics["theme_acc"]:.2%}, '
              f'Watermark: {metrics["watermark_acc"]:.2%}, '
              f'Combined: {combined_acc:.2%}')

        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            torch.save(student_model.state_dict(), config.distill_path)
            print(f'  [BEST] Model saved with combined acc: {combined_acc:.2%}')

    return student_model


def evaluate_all_heads(config, model, data_iter):
    model.eval()

    theme_correct = theme_total = 0
    watermark_correct_normal = watermark_total_normal = 0

    with torch.no_grad():
        for ((x, seq_len), (watermark_labels, theme_labels)) in data_iter:
            texts = (x.to(config.device), seq_len.to(config.device))
            outputs = model(texts, sw='True')

            theme_preds = torch.argmax(outputs['theme'], dim=1)
            theme_correct += (theme_preds == theme_labels.to(config.device)).sum().item()
            theme_total += theme_labels.size(0)

            watermark_preds = torch.argmax(outputs['watermark'], dim=1)
            watermark_correct_normal += (watermark_preds == watermark_labels.to(config.device)).sum().item()
            watermark_total_normal += watermark_labels.size(0)

    theme_acc = theme_correct / theme_total if theme_total > 0 else 0.0
    watermark_acc_normal = watermark_correct_normal / watermark_total_normal if watermark_total_normal > 0 else 0.0

    return {
        'theme_acc': theme_acc,
        'watermark_acc': watermark_acc_normal
    }




