import json
import os
import pickle
import torch
import csv
from tqdm import tqdm
import argparse
import copy


def prediction_adjust(prediction, labels):
    labels = labels[:len(prediction)]
    i = 0
    length = len(labels)
    while i < length:
        if labels[i] == True:
            j = i

            adjust_flag = False
            while labels[j] == True and j < length:
                if prediction[j] == True:
                    adjust_flag = True
                j += 1
                if j == length:
                    break
            if adjust_flag:
                for k in range(i, j):
                    prediction[k] = True
            i = j
        else:
            i += 1
    return prediction


def compute_add(prediction, labels):
    labels = labels[:len(prediction)]

    now_anomaly_flag = False  # 当前点是否是异常点
    find_anomaly_flag = False  # 当前是否有找到这段异常点

    latency_list = []
    latency = 0

    for i, label in enumerate(labels):
        if not label:
            if now_anomaly_flag:  # 上一个点是异常点
                latency_list.append(latency)
                now_anomaly_flag = False
                find_anomaly_flag = False
                latency = 0
            else:
                pass
        else:
            now_anomaly_flag = True
            if prediction[i]:
                find_anomaly_flag = True

            if not find_anomaly_flag:
                latency += 1
            else:
                pass

    if latency > 0:
        latency_list.append(latency)

    return latency_list


def compute_f_p_r(prediction, labels):
    labels = labels[:len(prediction)]
    TP = torch.sum(prediction * labels)
    TN = torch.sum((1 - prediction) * (1 - labels))
    FP = torch.sum(prediction * (1 - labels))
    FN = torch.sum((1 - prediction) * labels)
    precise = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)

    f = 2 * precise * recall / (precise + recall + 0.00001)
    # print(f"f value of {data_id} is {f}, threshold is {threshold}")
    return f.item(), precise.item(), recall.item()


def compute_adjust_f_in_fix_threshold(residual, labels, threshold_proper):
    threshold = residual.reshape(-1).topk(int(threshold_proper * len(residual))).values[-1].item()

    true = torch.ones_like(residual)
    false = torch.zeros_like(residual)
    origin_prediction = torch.where(residual > threshold, true, false)

    adjust_prediction = prediction_adjust(origin_prediction.clone(), labels)
    adjust_f, adjust_p, adjust_r = compute_f_p_r(adjust_prediction, labels)
    original_f, original_p, original_r = compute_f_p_r(origin_prediction, labels)
    add_list = compute_add(origin_prediction, labels)
    add_value = sum(add_list) / len(add_list)

    return [adjust_f, adjust_p, adjust_r, original_f, original_p, original_r,
            add_value]


def compute_best_threshold_for_average_adjust_f(residual_list, labels):
    threshold_infor_dict = {}
    best_f = -1
    print("begin to search ...")
    for i in tqdm(range(1, 80)):
        threshold_proper = i * 0.0005
        infor_list = []
        for residual in residual_list:
            infor_list.append(
                compute_adjust_f_in_fix_threshold(residual, labels, threshold_proper)
            )
            print(f"type of residual is {type(residual)}")
        threshold_infor_dict[threshold_proper] = infor_list

        # print(infor_list)
        infor_tensor = torch.Tensor(infor_list)
        average_infor_tensor = infor_tensor.mean(0)
        average_adjust_f = average_infor_tensor[0].item()
        if average_adjust_f > best_f:
            best_f = average_adjust_f
            print(f"best f update is {best_f}")
            best_proper = threshold_proper
            best_infor = infor_list

    return best_f, best_proper, best_infor, threshold_infor_dict


def compute_residual(pk_file, labels, ground_truth, compute_sum, compute_abs):
    all_gen = pk_file[0]

    all_target = pk_file[1]
    head = pk_file[5]
    head_target = pk_file[6]

    head = torch.cat(
        [head]
    )
    head_target = torch.cat(
        [head_target]
    )

    all_gen = all_gen[:, :, :]
    print(f"shape of head is {head.shape}")
    print(f"shape of all gen is {all_gen.shape}")
    all_target = all_target[:, :, :]
    feature_number = all_gen.shape[-1]
    all_gen = torch.Tensor(all_gen).reshape(-1, feature_number)
    all_target = torch.Tensor(all_target).reshape(-1, feature_number)

    head = torch.Tensor(head).squeeze()
    head_target = torch.Tensor(head_target).squeeze()

    all_gen = torch.cat([head, all_gen], dim=0)
    all_target = torch.cat([head_target, all_target], dim=0)
    print(f"shape of ground truth is {ground_truth.shape}")
    print(f"shape of all target is {all_target.shape}")
    print(f"check equal is {torch.all(all_target == torch.Tensor(ground_truth)[:all_target.shape[0]] * 20)}")

    labels = torch.Tensor(labels)[:len(all_gen)]

    if compute_sum and compute_abs:
        residual = torch.sum(torch.abs(all_gen - all_target), dim=-1)
    elif compute_sum and not compute_abs:
        residual = torch.sum((all_gen - all_target) ** 2, dim=-1)
    elif not compute_sum and compute_abs:
        residual, _ = torch.max(torch.abs(all_gen - all_target), dim=-1)
    elif not compute_sum and not compute_abs:
        residual, _ = torch.max((all_gen - all_target) ** 2, dim=-1)
    print(f"shape of residual is {residual.shape}")
    print(f"shape of labels is {labels.shape}")

    return residual, labels


def compute_one_subset_one_strategy(dataset_name, subset_name, compute_sum, compute_abs):
    residual_list = []
    for save_file in os.listdir(f"window_result"):
        if "save" in save_file:
            pass
        else:
            continue

        try:
            os.listdir(f"window_result/{save_file}/50")
        except:
            continue
        for subdata_name in os.listdir(f"window_result/{save_file}/50"):
            if subset_name + "_" in subdata_name:
                pass
            else:
                continue

            base_path = f"window_result/{save_file}/50/{subdata_name}"
            for pkl_path in os.listdir(base_path):
                if ".pk" in pkl_path:
                    # 记得每次都要读取，否则它会按照地址进行修改
                    labels = pickle.load(
                        open(f"data/Machine/{subset_name}_test_label.pkl", "rb")
                    )
                    ground_truth = pickle.load(
                        open(f"data/Machine/{subset_name}_test.pkl", "rb")
                    )

                    pkl_file = pickle.load(
                        open(f"{base_path}/{pkl_path}", 'rb')
                    )
                    print("\n\n")
                    print(f"begin to compute residual for {base_path}")
                    print(f"pkl file name is {pkl_path}")
                    print(f"subset name is {subset_name}")
                    residual, labels = compute_residual(pkl_file, labels=labels, ground_truth=ground_truth,
                                                        compute_sum=compute_sum, compute_abs=compute_abs)
                    residual_list.append(residual)

    best_f, best_proper, best_infor, threshold_infor_dict = compute_best_threshold_for_average_adjust_f(
        residual_list, labels
    )

    return best_f, best_proper, best_infor, threshold_infor_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SMD")
    # parser.add_argument("--compute_sum", type=str, default="True")
    # parser.add_argument("--compute_abs", type=str, default="True")


    args = parser.parse_args()
    data_id = args.dataset_name

    # for default setting
    args.compute_abs = True
    args.compute_sum = True

    if data_id == "MSL":
        args.compute_abs = True
        args.compute_sum = True
    if data_id == "PSM":
        args.compute_abs = True
        args.compute_sum = False
    if data_id == "SMAP":
        args.compute_abs = True
        args.compute_sum = True
    if data_id == "SWaT":
        args.compute_abs = True
        args.compute_sum = True

    if data_id == "SMD":
        args.compute_abs = True
        args.compute_sum = False


        
    dataset_name = args.dataset_name
    if dataset_name == "SMD":
        subset_name_list = [f"machine-1-{i}" for i in range(1, 9)]
        subset_name_list += [f"machine-2-{i}" for i in range(1, 10)]
        subset_name_list += [f"machine-3-{i}" for i in range(1, 12)]
    elif dataset_name == "GCP":
        subset_name_list = [f"service{i}" for i in range(0,30)]
    else:
        subset_name_list = [dataset_name]

    os.makedirs(f"score/{dataset_name}/best_infor", exist_ok=True)

    total_infor_csv = csv.writer(
        open(f"score/{dataset_name}/infor.csv", "a")
    )
    if args.compute_sum == "True":
        args.compute_sum = True
    elif args.compute_sum == "False":
        args.compute_sum = False
    if args.compute_abs == "True":
        args.compute_abs = True
    elif args.compute_abs == "False":
        args.compute_abs = False

    for subset_name in subset_name_list:
        best_f, best_proper, best_infor, threshold_infor_dict = compute_one_subset_one_strategy(
            dataset_name, subset_name, args.compute_sum, args.compute_abs
        )
        json.dump(
            best_infor,
            open(f"score/{dataset_name}/best_infor/{subset_name}_{args.compute_sum}_{args.compute_abs}.json", "w")
        )

        json.dump(
            threshold_infor_dict,
            open(f"score/{dataset_name}/infor_dict/{subset_name}_{args.compute_sum}_{args.compute_abs}.json", "w")
        )
        total_infor_csv.writerow(
            [subset_name, args.compute_sum, args.compute_abs, best_f, best_proper]
        )



