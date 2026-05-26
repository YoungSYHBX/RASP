# coding: UTF-8
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_eval import train, init_network, test, fineTune, multi_train, multi_test, multi_evaluate, multi_fineTune, distill, distill_all_outputs
from importlib import import_module
import torch.nn.utils.prune as prune
import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--adv', type=str, default='', required=False, help='choose a AT method: FGM, FGSM, PGD, FreeAT')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', type=str, default='eng', help='word: chn or eng')
parser.add_argument('--ways', type=str, default='train', help='train, predict, fineTune, prune, randomPrune, distill')
parser.add_argument('--type', type=str, default='normal', help='multitask or normal')
parser.add_argument('--sw', type=str, default='True', help='choose if use stubborn watermark')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'dataset'

    if args.word == 'chn':
        embedding = 'embedding_SougouNews.npz'
    else:
        embedding = 'glove_6B200d_32.npz'

    model_name = args.model  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    adv_name = args.adv  # FGM, FGSM, PGD, FreeAT
    way_name = args.ways  # train, predict, fineTune, prune, multitask
    type_name = args.type  # normal or multitask
    from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    y = import_module(adv_name) if len(adv_name) > 0 else None
    z = import_module('models.' + 'MultiTaskTextCNNNowm')
    config = x.Config(dataset, embedding)
    student_config = z.Config(dataset, embedding)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data, trigger_data, fineTune_data = build_dataset(config, args.word, args.type)

    train_iter = build_iterator(train_data, config, args.type)
    dev_iter = build_iterator(dev_data, config, args.type)
    test_iter = build_iterator(test_data, config, args.type)
    trigger_iter = build_iterator(trigger_data, config, args.type)
    fineTune_iter = build_iterator(fineTune_data, config, args.type)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    adv = y.ATModel(model) if y else None
    student_model = z.Model(student_config).to(student_config.device)
    print(config.device)
    print(adv)

    if way_name == 'train':
        init_network(model)
        if type_name == 'multitask':
            multi_train(config, model, train_iter, dev_iter, test_iter, sw=args.sw)
            multi_test(config, model, trigger_iter)
        else:
            train(config, model, train_iter, dev_iter, test_iter, adv)
            test(config, model, trigger_iter)
    elif way_name == 'predict':
        model_CKPT = torch.load('dataset/saved_dict/MultiTaskTextRNN_agn_sw.ckpt')
        # model.load_state_dict(model_CKPT, strict=False)
        model.load_state_dict(model_CKPT, strict=False)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        model.eval()
        if type_name == 'multitask':
            multi_test(config, model, test_iter)
            multi_test(config, model, trigger_iter)
        else:
            test(config, model, trigger_iter)
    elif way_name == 'fineTune':
        model_CKPT = torch.load('dataset/saved_dict/Transformer_nomodulezwsp.ckpt')
        model.load_state_dict(model_CKPT)
        if type_name == 'multitask':
            multi_fineTune(config, model, fineTune_iter, dev_iter, test_iter)
            multi_test(config, model, trigger_iter)
        else:
            fineTune(config, model, fineTune_iter, dev_iter, test_iter)
            test(config, model, trigger_iter)
    elif way_name == 'prune':
        model_CKPT = torch.load('dataset/saved_dict/BERT_wm.ckpt')
        model.load_state_dict(model_CKPT)
        model = model.to('cuda')
        model.eval()

        if type_name == 'multitask':
            if model_name == 'MultiTaskTextCNN' or model_name == 'MultiTaskTextCNNNowm':
                prune_ratio = 0.1

                print(f"开始非结构化剪枝（剪掉 {prune_ratio * 100:.0f}% 的权重）...")

                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        weight = module.weight.data
                        mask = (torch.rand_like(weight) > prune_ratio).float()
                        module.weight.data *= mask

                print("剪枝完成，保存模型...")
                torch.save(model.state_dict(), config.prune_path)

                model_p = torch.load(config.prune_path, map_location='cpu')
                model.load_state_dict(model_p)
                model.eval()

                print(f"当前剪枝率：{prune_ratio}")
                multi_test(config, model, test_iter)
                multi_test(config, model, trigger_iter)
            elif model_name == 'MultiTaskTextRNN' or model_name == 'MultiTaskTextRNNNowm':
                prune_ratio = 0.5
                print(f"剪掉 {prune_ratio * 100:.0f}% 的权重）...")

                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        weight = module.weight.data
                        mask = (torch.rand_like(weight) > prune_ratio).float()
                        module.weight.data *= mask

                torch.save(model.state_dict(), config.prune_path)

                model_p = torch.load(config.prune_path, map_location='cpu')
                model.load_state_dict(model_p)
                model.eval()

                print(f"当前剪枝率：{prune_ratio}")
                multi_test(config, model, test_iter)
                multi_test(config, model, trigger_iter)
            elif model_name == 'MultiTaskTransformer' or model_name == 'MultiTaskTransformerNowm':
                prune_ratio = 0.7
                print(f"开始对 MultiTaskTransformer 进行非结构化剪枝（剪掉 {prune_ratio * 100:.0f}% 的权重）...")

                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"正在剪枝模块: {name}")
                        weight = module.weight.data
                        mask = (torch.rand_like(weight) > prune_ratio).float()
                        module.weight.data *= mask

                print("剪枝完成，保存模型...")
                torch.save(model.state_dict(), config.prune_path)

                print("加载剪枝模型...")
                model_p = torch.load(config.prune_path, map_location='cpu')
                model.load_state_dict(model_p)
                model.eval()

                print(f"当前剪枝率：{prune_ratio}")

                multi_test(config, model, test_iter)
                multi_test(config, model, trigger_iter)
            elif model_name == 'BERT' or model_name == 'BERT_Nowm':
                prune_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                for prune_ratio in prune_ratios:
                    print(f"\n{'=' * 60}")
                    print(f"测试剪枝比例: {prune_ratio * 100:.0f}%")
                    print('=' * 60)

                    model_CKPT = torch.load('dataset/saved_dict/BERT_wm.ckpt')
                    model.load_state_dict(model_CKPT)
                    model = model.to('cuda')
                    model.eval()

                    total_params = 0
                    pruned_params = 0

                    for param in model.parameters():
                        if param.requires_grad:
                            total_params += param.numel()

                            mask = (torch.rand_like(param) > prune_ratio).float()

                            current_pruned = (mask == 0).sum().item()
                            pruned_params += current_pruned

                            param.data *= mask

                    actual_prune_rate = pruned_params / total_params if total_params > 0 else 0
                    print(f"实际剪枝率: {actual_prune_rate * 100:.1f}% ({pruned_params:,}/{total_params:,})")

                    prune_path = config.save_path.replace('.ckpt', f'_prune_{int(prune_ratio * 100)}.ckpt')
                    torch.save(model.state_dict(), prune_path)

                    multi_test(config, model, test_iter)
                    multi_test(config, model, trigger_iter)

                    torch.cuda.empty_cache()

        else:
            if model_name != 'Transformer':
                if model_name == 'TextCNN' or model_name == 'TextCNN__savetempt':
                    prune_rate = 0.9
                    with torch.no_grad():
                        total_params = 0
                        total_pruned = 0

                        for name, param in model.named_parameters():
                            if 'weight' in name and param.dim() > 1:
                                weight = param.data
                                abs_weight = torch.abs(weight)

                                flat_abs = abs_weight.flatten()
                                k = int(flat_abs.numel() * prune_rate)

                                if k > 0 and k < flat_abs.numel():
                                    sorted_vals, _ = torch.sort(flat_abs)
                                    threshold = sorted_vals[k - 1].item()

                                    if threshold < 1e-8:
                                        median_val = torch.median(sorted_vals).item()
                                        threshold = max(median_val * 0.5, 1e-6)
                                        print(f"层 {name}: 使用调整后的阈值 {threshold:.10f}")

                                    mask = (abs_weight > threshold).float()
                                    param.data *= mask

                                    layer_pruned = (mask == 0).sum().item()
                                    layer_total = mask.numel()
                                    total_params += layer_total
                                    total_pruned += layer_pruned

                                    print(f"层 {name}: {layer_pruned}/{layer_total} ({layer_pruned / layer_total:.4f})")

                        if total_params > 0:
                            print(f"\n总体剪枝率: {total_pruned}/{total_params} ({total_pruned / total_params:.6f})")

                elif model_name == 'TextRNN' or model_name == 'TextRNN_savetempt':
                    parameters_to_prune = [
                        (model.embedding, 'weight'),
                        (model.lstm, 'weight_ih_l0'),
                        (model.lstm, 'weight_hh_l0'),
                        (model.lstm, 'weight_ih_l0_reverse'),
                        (model.lstm, 'weight_hh_l0_reverse'),
                        (model.lstm, 'weight_ih_l1'),
                        (model.lstm, 'weight_hh_l1'),
                        (model.lstm, 'weight_ih_l1_reverse'),
                        (model.lstm, 'weight_hh_l1_reverse'),
                        (model.fc, 'weight'),
                    ]

                torch.save(model.state_dict(), config.prune_path)
                model_p = torch.load('dataset/saved_dict/TextCNN_savetempt_prune.ckpt')
                model.load_state_dict(model_p)
                model.eval()
                test(config, model, test_iter)
                test(config, model, trigger_iter)
            else:
                all_weights = []
                weight_dict = {}
                pruning_ratio = 0.1

                for name, param in model.named_parameters():
                    if 'weight' in name:
                        all_weights.append(param.data.abs().flatten())
                        weight_dict[name] = param.data
                all_weights = torch.cat(all_weights)
                num_weights_to_prune = int(pruning_ratio * len(all_weights))
                if num_weights_to_prune == 0:
                    print("num_weights_to_prune == 0")
                threshold = torch.topk(all_weights, num_weights_to_prune, largest=False)[0].max().item()

                with torch.no_grad():
                    for name, weight in weight_dict.items():
                        # print(f"Weight before: {original_weights[name].abs().mean().item()}")
                        mask = weight.abs() >= threshold
                        weight.data = weight.data * mask.float()
                        # print(f"Weight after: {weight.abs().mean().item()}")

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'weight' in name:
                            if name in weight_dict:
                                param.data.copy_(weight_dict[name].data)
                # pruned_weights = {name: param.data.clone() for name, param in model.named_parameters() if 'weight' in name}
                # for name in original_weights:
                #     if not torch.equal(original_weights[name], pruned_weights[name]):
                #         print(f"Weight {name} has been changed.")
                torch.save(model.state_dict(), config.prune_path)
                model_p = torch.load('dataset/saved_dict/TextCNN_savetempt_prune.ckpt')
                # pre_weight = torch.load('dataset/saved_dict/Transformer_12ZWSP_FGM.ckpt')
                # post_weight = torch.load('dataset/saved_dict/Transformer_prune.ckpt')
                # for param_name in pre_weight:
                #     if not torch.all(pre_weight[param_name].eq(post_weight[param_name])):
                #         print(f"Weight {param_name} has changed.")
                model.load_state_dict(model_p, strict=False)
                model.eval()
                test(config, model, test_iter)
                test(config, model, trigger_iter)
    elif way_name == 'randomPrune':
        model_CKPT = torch.load('dataset/saved_dict/Transformer_savetempt_12ZWSP.ckpt')
        model.load_state_dict(model_CKPT)
        # print(model_CKPT.keys())
        if type_name == 'multitask':
            ratio = 0.1

            all_weights = []
            module_info = []

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    weight = module.weight.data.cpu().numpy()
                    all_weights.append(weight.flatten())
                    module_info.append((name, module, weight.shape))

            all_weights = np.concatenate(all_weights)
            num_params = len(all_weights)
            num_prune = int(num_params * ratio)

            indices = np.random.choice(num_params, size=num_prune, replace=False)

            global_mask = np.ones(num_params, dtype=bool)
            global_mask[indices] = False

            current_index = 0
            for name, module, shape in module_info:
                param_count = np.prod(shape)
                layer_mask = global_mask[current_index:current_index + param_count]
                layer_mask = layer_mask.reshape(shape)
                current_index += param_count

                with torch.no_grad():
                    module.weight.data *= torch.tensor(~layer_mask, dtype=torch.float32, device=module.weight.device)

            torch.save(model.state_dict(), config.prune_path)
            model_p = torch.load('dataset/saved_dict/TextRNN_savetempt_nomodule.ckpt')
            model.load_state_dict(model_p)
            model.eval()

            print(f"当前剪枝率：{ratio}")
            multi_test(config, model, test_iter)
            multi_test(config, model, trigger_iter)
        else:
            pruning_rate = 0.7
            if model_name == 'TextCNN':
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        num_weights = param.numel()
                        num_to_prune = int(num_weights * pruning_rate)
                        if num_to_prune > 0:
                            mask = torch.ones(num_weights)
                            indices = torch.randperm(num_weights)[:num_to_prune]
                            # print(param.data.device)
                            # print(mask.device)
                            mask = mask.to(param.data.device)
                            mask[indices] = 0
                            mask = mask.view_as(param)
                            param.data.mul_(mask)
            elif model_name == 'TextRNN' or model_name == 'TextRNN_savetempt':
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        num_weights = param.numel()
                        num_pruned = int(num_weights * pruning_rate)
                        prune_indices = torch.randperm(num_weights)[:num_pruned]
                        param.data.view(-1)[prune_indices] = 0
            elif model_name == 'Transformer' or model_name == 'Transformer_savetempt':
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        prune.random_unstructured(module, name='weight', amount=pruning_rate)

            torch.save(model.state_dict(), config.prune_path)
            model_p = torch.load('dataset/saved_dict/Transformer_savetempt_prune.ckpt')
            model.load_state_dict(model_p)
            model.eval()
            test(config, model, test_iter)
            test(config, model, trigger_iter)
    elif way_name == 'distill':
        teacher_model = x.Model(config).to(config.device)
        teacher_ckpt = torch.load('dataset/saved_dict/MultiTaskTextCNN_wm.ckpt', map_location=config.device)
        teacher_model.load_state_dict(teacher_ckpt, strict=False)
        teacher_model.eval()

        student_model = z.Model(student_config).to(student_config.device)
        if model_name != 'Transformer':
            init_network(student_model)

        # distill(student_config, teacher_model, student_model, fineTune_iter, dev_iter)
        distill_all_outputs(student_config, teacher_model, student_model, fineTune_iter, dev_iter)

        model_d = torch.load('dataset/saved_dict/MultiTaskTextCNNNowm_distill.ckpt')
        student_model.load_state_dict(model_d)
        student_model.eval()
        print("\nTesting distilled student model:")
        multi_test(student_config, student_model, test_iter)
        multi_test(student_config, student_model, trigger_iter)

    elif way_name == 'adversarial_FGSM':
        import torch.nn.functional as F

        DEVICE = config.device
        wm_label = config.num_classes - 1
        print(f"[INFO] Total classes: {config.num_classes}, Watermark label ID: {wm_label}")

        model_CKPT = torch.load('dataset/saved_dict/Transformer_12ZWSP_module.ckpt', map_location=DEVICE)
        model.load_state_dict(model_CKPT)
        model = model.to(DEVICE)
        model.eval()

        print("\n" + "=" * 60)
        print("CLEAN EVALUATION (via test())")
        print("=" * 60)
        print("[Main Task - Clean]")
        test(config, model, test_iter)

        def evaluate_adversarial(epsilon, model, trigger_iter, test_iter, wm_label, device):
            model.eval()
            total_wm = correct_wm = 0
            for batch in trigger_iter:
                (input_ids, seq_len), labels = batch
                input_ids, seq_len, labels = input_ids.to(device), seq_len.to(device), labels.to(device)

                with torch.no_grad():
                    orig_pred = model((input_ids, seq_len)).argmax(dim=1)

                model.train()
                embeds = model.embedding(input_ids).detach().requires_grad_(True)
                output = model((embeds, seq_len))
                loss = F.cross_entropy(output, orig_pred)
                loss.backward()
                model.zero_grad()

                perturbed_embeds = embeds + epsilon * embeds.grad.sign()
                model.eval()

                with torch.no_grad():
                    adv_pred = model((perturbed_embeds, seq_len)).argmax(dim=1)

                total_wm += input_ids.size(0)
                correct_wm += (adv_pred == labels).sum().item()

            wm_acc = correct_wm / total_wm if total_wm > 0 else 0.0

            total_main = correct_main = 0
            for batch in test_iter:
                (input_ids, seq_len), labels = batch
                input_ids, seq_len, labels = input_ids.to(device), seq_len.to(device), labels.to(device)

                with torch.no_grad():
                    orig_pred = model((input_ids, seq_len)).argmax(dim=1)

                model.train()
                embeds = model.embedding(input_ids).detach().requires_grad_(True)
                output = model((embeds, seq_len))
                loss = F.cross_entropy(output, orig_pred)
                loss.backward()
                model.zero_grad()

                perturbed_embeds = embeds + epsilon * embeds.grad.sign()
                model.eval()

                with torch.no_grad():
                    adv_pred = model((perturbed_embeds, seq_len)).argmax(dim=1)

                total_main += labels.size(0)
                correct_main += (adv_pred == labels).sum().item()

            main_acc = correct_main / total_main if total_main > 0 else 0.0

            return wm_acc, main_acc


        print("\n" + "=" * 70)
        print("SCANNING EPSILON: 0.01 → 0.10")
        print("=" * 70)
        print(f"{'Epsilon':<10} {'Watermark Acc':<20} {'Main Task Acc':<20}")
        print("-" * 70)

        results = []
        for i in range(1, 11):  #
            EPSILON = i * 0.01
            wm_acc, main_acc = evaluate_adversarial(
                epsilon=EPSILON,
                model=model,
                trigger_iter=trigger_iter,
                test_iter=test_iter,
                wm_label=wm_label,
                device=DEVICE
            )
            results.append((EPSILON, wm_acc, main_acc))
            print(f"{EPSILON:<10.2f} {wm_acc:<20.4f} {main_acc:<20.4f}")
    elif way_name == 'adversarial_PGD':

        import torch.nn.functional as F
        import csv

        DEVICE = config.device
        wm_label = config.num_classes - 1
        print(f"[INFO] Total classes: {config.num_classes}, Watermark label ID: {wm_label}")

        model_CKPT = torch.load('dataset/saved_dict/Transformer_12ZWSP_module.ckpt', map_location=DEVICE)
        model.load_state_dict(model_CKPT)
        model = model.to(DEVICE)
        model.eval()

        print("\n" + "=" * 60)
        print("CLEAN EVALUATION (via test())")
        print("=" * 60)
        print("[Main Task - Clean]")
        test(config, model, test_iter)

        def pgd_attack(model, input_ids, seq_len, orig_pred, epsilon, alpha, num_steps, device):
            model.eval()
            embeds_init = model.embedding(input_ids).detach()  # [B, L, E]
            perturbation = torch.zeros_like(embeds_init).to(device)

            perturbation.uniform_(-epsilon, epsilon)
            perturbation = torch.clamp(embeds_init + perturbation, min=-1.0, max=1.0) - embeds_init
            perturbation.requires_grad_(True)

            for _ in range(num_steps):
                model.train()
                adv_embeds = embeds_init + perturbation
                output = model((adv_embeds, seq_len))
                loss = F.cross_entropy(output, orig_pred)

                grad = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]

                with torch.no_grad():
                    perturbation += alpha * grad.sign()
                    perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
                    perturbation = torch.clamp(embeds_init + perturbation, min=-1.0, max=1.0) - embeds_init

                perturbation.requires_grad_(True)

            model.eval()
            return (embeds_init + perturbation).detach()


        print("\n" + "=" * 80)
        print("PGD ATTACK SCAN: EPSILON = 0.01 → 0.10")
        print("=" * 80)
        print(f"{'Epsilon':<10} {'Watermark Acc':<20} {'Main Task Acc':<20}")
        print("-" * 80)

        results = []
        NUM_STEPS = 10

        for i in range(1, 11):
            EPSILON = i * 0.01
            ALPHA = EPSILON / 4.0

            total_wm = correct_wm = 0
            for batch in trigger_iter:
                (input_ids, seq_len), labels = batch
                input_ids, seq_len, labels = input_ids.to(DEVICE), seq_len.to(DEVICE), labels.to(DEVICE)

                with torch.no_grad():
                    orig_pred = model((input_ids, seq_len)).argmax(dim=1)

                perturbed_embeds = pgd_attack(
                    model, input_ids, seq_len, orig_pred,
                    epsilon=EPSILON, alpha=ALPHA, num_steps=NUM_STEPS, device=DEVICE
                )

                with torch.no_grad():
                    adv_pred = model((perturbed_embeds, seq_len)).argmax(dim=1)

                total_wm += input_ids.size(0)
                correct_wm += (adv_pred == labels).sum().item()

            wm_acc = correct_wm / total_wm if total_wm > 0 else 0.0

            total_main = correct_main = 0
            for batch in test_iter:
                (input_ids, seq_len), labels = batch
                input_ids, seq_len, labels = input_ids.to(DEVICE), seq_len.to(DEVICE), labels.to(DEVICE)

                with torch.no_grad():
                    orig_pred = model((input_ids, seq_len)).argmax(dim=1)

                perturbed_embeds = pgd_attack(
                    model, input_ids, seq_len, orig_pred,
                    epsilon=EPSILON, alpha=ALPHA, num_steps=NUM_STEPS, device=DEVICE
                )

                with torch.no_grad():
                    adv_pred = model((perturbed_embeds, seq_len)).argmax(dim=1)

                total_main += labels.size(0)
                correct_main += (adv_pred == labels).sum().item()

            main_acc = correct_main / total_main if total_main > 0 else 0.0

            results.append((EPSILON, wm_acc, main_acc))
            print(f"{EPSILON:<10.2f} {wm_acc:<20.4f} {main_acc:<20.4f}")
    elif way_name == 'GMA':

        print("\n" + "=" * 70)
        print("GRADIENT MASKING ATTACK EVALUATION (Model Stealer Perspective)")
        print("=" * 70)
        print(f"Model: {args.model}, Type: {args.type}")

        model_CKPT = torch.load('dataset/saved_dict/BERT_wm.ckpt', map_location=config.device)
        model.load_state_dict(model_CKPT)
        model = model.to(config.device)

        def apply_gradient_masking(model, config, mask_factor=0.1):
            print(f"\n应用梯度掩蔽攻击，强度: {mask_factor}")

            attacked_model = type(model)(config).to(config.device)
            attacked_model.load_state_dict(model.state_dict())

            total_params = 0
            modified_params = 0

            with torch.no_grad():
                for name, param in attacked_model.named_parameters():
                    if param.requires_grad and param.dim() > 0:
                        param_norm = torch.norm(param).item()
                        total_params += param.numel()

                        if param_norm < 1e-8:
                            continue

                        name_lower = name.lower()

                        if any(keyword in name_lower for keyword in
                               ['watermark_head', 'watermark_vector', 'watermark_embedding']):
                            attack_strength = mask_factor * 1.2
                            noise_scale = attack_strength * param_norm * 0.3
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - attack_strength * 0.5) + noise
                            modified_params += param.numel()

                            if modified_params < 1000:
                                print(f"  [水印] {name} (强度: {attack_strength:.3f})")

                        elif any(keyword in name_lower for keyword in
                                 ['perturbation', 'projection', 'mask_generator', 'fusion']):
                            attack_strength = mask_factor * 0.8
                            noise_scale = attack_strength * param_norm * 0.2
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - attack_strength * 0.3) + noise
                            modified_params += param.numel()

                        elif any(keyword in name_lower for keyword in
                                 ['head', 'classifier', 'fc', 'linear']):
                            attack_strength = mask_factor * 0.3
                            noise_scale = attack_strength * param_norm * 0.1
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - attack_strength * 0.1) + noise
                            modified_params += param.numel()

                        elif any(keyword in name_lower for keyword in
                                 ['conv', 'lstm', 'embedding', 'bert']):
                            if 'embedding' not in name_lower:
                                attack_strength = mask_factor * 0.05
                                noise_scale = attack_strength * param_norm * 0.05
                                noise = torch.randn_like(param) * noise_scale
                                param.data += noise * 0.05
                                modified_params += param.numel()

            modification_rate = modified_params / total_params if total_params > 0 else 0
            print(f"攻击统计: 修改参数比例: {modification_rate:.4%}")

            return attacked_model

        def apply_adaptive_gradient_masking(model, config, base_strength=0.2):
            print(f"\n应用自适应梯度掩蔽攻击，基准强度: {base_strength}")

            attacked_model = type(model)(config).to(config.device)
            attacked_model.load_state_dict(model.state_dict())

            total_params = 0
            adapted_params = 0

            with torch.no_grad():
                for name, param in attacked_model.named_parameters():
                    if param.requires_grad and param.dim() > 0:
                        param_norm = torch.norm(param).item()
                        total_params += param.numel()

                        if param_norm < 1e-8:
                            continue

                        name_lower = name.lower()


                        param_var = torch.var(param).item()
                        variance_factor = 1.5 / (1.0 + param_var * 3)

                        param_abs_mean = torch.mean(torch.abs(param)).item()
                        mean_factor = 1.0 + param_abs_mean * 1

                        norm_factor = 1.0 + param_norm * 0.1

                        adaptive_coef = (variance_factor + mean_factor + norm_factor) / 3
                        adaptive_coef = max(0.8, min(1.5, adaptive_coef))

                        if any(keyword in name_lower for keyword in
                               ['watermark_head', 'watermark_vector', 'watermark_embedding']):
                            adaptive_strength = base_strength * adaptive_coef * 1.2
                            noise_scale = adaptive_strength * param_norm * 0.3
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - adaptive_strength * 0.5) + noise
                            adapted_params += param.numel()

                            if adapted_params < 1000:
                                print(f"  [自适应水印] {name}: 强度={adaptive_strength:.4f} "
                                      f"(自适应系数={adaptive_coef:.2f})")

                        elif any(keyword in name_lower for keyword in
                                 ['perturbation', 'projection', 'mask_generator', 'fusion']):
                            adaptive_strength = base_strength * adaptive_coef * 0.8
                            noise_scale = adaptive_strength * param_norm * 0.2
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - adaptive_strength * 0.3) + noise
                            adapted_params += param.numel()

                        elif any(keyword in name_lower for keyword in
                                 ['head', 'classifier', 'fc', 'linear']):
                            adaptive_strength = base_strength * adaptive_coef * 0.3
                            noise_scale = adaptive_strength * param_norm * 0.1
                            noise = torch.randn_like(param) * noise_scale
                            param.data = param.data * (1 - adaptive_strength * 0.1) + noise
                            adapted_params += param.numel()

                        elif any(keyword in name_lower for keyword in
                                 ['conv', 'lstm', 'embedding', 'bert']):
                            if 'embedding' not in name_lower:
                                adaptive_strength = base_strength * adaptive_coef * 0.05
                                noise_scale = adaptive_strength * param_norm * 0.05
                                noise = torch.randn_like(param) * noise_scale
                                param.data += noise * 0.05
                                adapted_params += param.numel()

            adaptation_rate = adapted_params / total_params if total_params > 0 else 0
            print(f"自适应攻击统计: 自适应比例: {adaptation_rate:.4%}")

            return attacked_model

        print("\n" + "=" * 70)
        print("[原始模型性能基准]")
        print("=" * 70)
        print("\n正常测试集:")
        multi_test(config, model, test_iter)
        print("\n触发集:")
        multi_test(config, model, trigger_iter)

        print("\n" + "=" * 70)
        print("实验1：梯度掩蔽攻击（强度0.1-0.9）")
        print("=" * 70)

        gradient_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for strength in gradient_strengths:
            print(f"\n{'=' * 60}")
            print(f"[梯度掩蔽攻击] 强度: {strength}")
            print('=' * 60)

            attacked_model = apply_gradient_masking(model, config, strength)
            attacked_model.eval()

            print("\n[正常测试集]")
            multi_test(config, attacked_model, test_iter)

            print("\n[触发集测试]")
            multi_test(config, attacked_model, trigger_iter)

        print("\n" + "=" * 70)
        print("实验2：自适应梯度掩蔽攻击（强度0.1-0.5）")
        print("=" * 70)

        adaptive_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

        for strength in adaptive_strengths:
            print(f"\n{'=' * 60}")
            print(f"[自适应梯度掩蔽攻击] 基准强度: {strength}")
            print('=' * 60)

            attacked_model = apply_adaptive_gradient_masking(model, config, strength)
            attacked_model.eval()

            print("\n[正常测试集]")
            multi_test(config, attacked_model, test_iter)

            print("\n[触发集测试]")
            multi_test(config, attacked_model, trigger_iter)

        print("\n" + "=" * 70)
        print("所有实验完成")
        print("=" * 70)
