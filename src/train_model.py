import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import utils.preprocessing as pp
import utils.modeling as m
import utils.evaluation as eval
import utils.dataloader as dl

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

def model_setup(num_labels, device,dropout):
    model = m.bart_classifier(num_labels=num_labels, dropout=dropout).to(device)
    for n, p in model.named_parameters():
        if "bart.shared.weight" in n or "bart.encoder.embed" in n:
            p.requires_grad = False

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')], 'lr': float(2e-5)},
        {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(1e-3)},
        {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(1e-3)}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=0.0001)

    return model, optimizer

def run_classifier():
    # Load file
    file_train = "../data/raw_train.csv"
    file_val = "../data/raw_val.csv"
    file_test = "../data/raw_test.csv"
    file_aug = "../data/raw_tts_augment_data.csv"

    concat_text_train = pp.load_data(file_train)
    concat_text_val = pp.load_data(file_val)
    concat_text_test = pp.load_data(file_test)
    concat_text_aug = pp.load_data(file_aug)

    encoded_dict_val = pp.clean_and_tokenize(concat_text_val)
    encoded_dict_test = pp.clean_and_tokenize(concat_text_test)

    input_ids_val, attention_mask_val, stance_val = pp.convert_tensor(encoded_dict_val)
    input_ids_test, attention_mask_test, stance_test = pp.convert_tensor(encoded_dict_test)

    data_loader_val = dl.create_dataloder(input_ids_val, attention_mask_val, stance_val)
    data_loader_test = dl.create_dataloder(input_ids_test, attention_mask_test, stance_test)
    model, optimizer = model_setup(num_labels=3,device=device,dropout=0.1)
    loss_function = nn.CrossEntropyLoss()  # 交叉熵函数
    print(model)

    for stage_idx, aug_ratio in enumerate([0.]):
        print(f"Stage {stage_idx + 1}: original data with {aug_ratio * 100}% augmented data")

        com_dataset = dl.combine_datasets(concat_text_train, concat_text_aug, aug_ratio)
        encoded_dict_train = pp.clean_and_tokenize(com_dataset)
        input_ids_train, attention_mask_train, stance_train = pp.convert_tensor(encoded_dict_train)
        data_loader_train = dl.create_dataloder(input_ids_train, attention_mask_train, stance_train)

        step = 0
        current_test_f1 = 0
        best_test_f1 = 0.82
        for epoch in range(5):

            start_train = time.time()
            print("-------------- 模型训练 --------------")
            model.train()
            print(f"Stage: {stage_idx + 1}  Epoch: {epoch}")
            train_loss = []
            running_loss = 0

            for index, sample_batch in enumerate(data_loader_train, start=0):
                print(index)
                dict_batch = {'input_ids': sample_batch[0], 'attention_mask': sample_batch[1],
                              'stance': sample_batch[-1]}
                # 封装成字典
                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                outputs = model(**inputs)  # 预测
                loss = loss_function(outputs, inputs['stance'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                train_loss.append(loss.item())
                running_loss = running_loss + loss.item()

                if step % 70 == 69:
                    model.eval()
                    with torch.no_grad():
                        preds_test = []
                        for index, sample_batch in enumerate(data_loader_test, start=0):
                            dict_batch = {'input_ids': sample_batch[0], 'attention_mask': sample_batch[1],
                                          'stance': sample_batch[-1]}
                            inputs = {k: v.to(device) for k, v in dict_batch.items()}
                            outputs = model(**inputs)
                            preds_test.append(outputs)
                        preds_test = torch.cat(preds_test, dim=0)
                        f1macro_test = eval.compute_performance(preds_test, stance_test, "test", step)
                        print("F1 Score: ", f1macro_test)
                    model.train()
            end_train = time.time()
            print('模型预训练所运行时间为: %s Seconds' % int(((end_train - start_train) / 60)))

            print("-------------- 模型评估 --------------")

            start = time.time()
            model.eval()
            with torch.no_grad():
                preds_val = []
                for index, sample_batch in enumerate(data_loader_val, start=0):
                    dict_batch = {'input_ids': sample_batch[0], 'attention_mask': sample_batch[1],
                                  'stance': sample_batch[-1]}
                    inputs = {k: v.to(device) for k, v in dict_batch.items()}
                    outputs = model(**inputs)
                    preds_val.append(outputs)
                preds_val = torch.cat(preds_val, dim=0)
                f1macro_val = eval.compute_performance(preds_val, stance_val, "val", step)
                print("F1 Score: ", f1macro_val)

            with torch.no_grad():
                preds_test = []
                for index, sample_batch in enumerate(data_loader_test, start=0):
                    dict_batch = {'input_ids': sample_batch[0], 'attention_mask': sample_batch[1],
                                  'stance': sample_batch[-1]}
                    inputs = {k: v.to(device) for k, v in dict_batch.items()}
                    outputs = model(**inputs)
                    preds_test.append(outputs)
                preds_test = torch.cat(preds_test, dim=0)
                f1macro_test = eval.compute_performance(preds_test, stance_test, "test", step)
                print("F1 Score: ", f1macro_test)

                if f1macro_test > current_test_f1:
                    current_test_f1 = f1macro_test
                print("目前F1最高分: ", current_test_f1)

                if f1macro_test >= best_test_f1:
                    best_test_f1 = f1macro_test  # 更新最佳F1分数
                    print(f"保存新的最佳模型，F1分数: {best_test_f1}")
                    torch_random_state = torch.random.get_rng_state()
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'random_state': torch_random_state
                    }, f'bart_{stage_idx}_{epoch}_{best_test_f1:.3f}.pth')  # 保存模型和优化器状态
                    print(f"模型已保存为: bart_{stage_idx}_{epoch}_{best_test_f1:.3f}.pth")
            end = time.time()

            print('模型评估运行时间为: %s Seconds' % int(((end - start) / 60)))

if __name__ == "__main__":
    run_classifier()