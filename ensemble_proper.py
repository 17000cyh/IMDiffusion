import os
import pickle
import torch
import csv
from tqdm import tqdm
import argparse
from pathlib import Path

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

def compute_f(prediction,label):
    label = label[:len(prediction)]
    TP = torch.sum(prediction * label)
    TN = torch.sum((1 - prediction) * (1 - label))
    FP = torch.sum(prediction * (1 - label))
    FN = torch.sum((1 - prediction) * label)
    precise = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)

    f = 2 * precise * recall / (precise + recall + 0.00001)
    return  precise, recall, f


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

def merge(pkl_path,data_id,machine_number = ""):
    fr = open(pkl_path, "rb")
    result = pickle.load(fr)

    all_gen = result[0]

    all_target = result[1]
    head = result[5]
    head_target = result[6]
    head_middle = result[-1]  # [50,10,38]
    all_gen_middle = result[-2]  # [573, 50, 80, 38]

    head = torch.cat(
        [head]
    )
    head_target = torch.cat(
        [head_target]
    )
    # print(f"shape of head middle is {head_middle.shape}")
    # print(f"shape of all_gen_middle[0, :, 0:15, :] is {all_gen_middle[0, :, 0:15, :].shape}")
    head_middle = torch.cat(
        [head_middle], dim=1
    )

    all_gen = all_gen[:, :, :]
    all_target = all_target[:, : , :]
    all_gen_middle = all_gen_middle[:, :,:, :].permute(1, 0, 2, 3)  # [diffusion step, batch number, window length, feature number]


    diffusion_steps = all_gen_middle.shape[0]
    feature_number = all_gen_middle.shape[-1]

    all_gen = torch.Tensor(all_gen).reshape(-1, feature_number)
    all_target = torch.Tensor(all_target).reshape(-1, feature_number)
    all_gen_middle = all_gen_middle.reshape(diffusion_steps, -1, feature_number)


    head = torch.Tensor(head).squeeze()
    head_target = torch.Tensor(head_target).squeeze()
    head_middle = head_middle.squeeze()

    all_gen = torch.cat([head, all_gen], dim=0)
    all_target = torch.cat([head_target, all_target], dim=0)
    # print(f'shape of head middle is {head_middle.shape}')
    # print(f"shape of all gen is {all_gen_middle.shape}")
    all_gen_middle = torch.cat([head_middle, all_gen_middle], dim=1)

    if data_id == "SMD" or data_id == "GCP":
        print(f"machine number is {machine_number}")
        label = pickle.load(
            open(f"data/Machine/{machine_number}_test_label.pkl", "rb")
        )
        origin_data = pickle.load(
            open(f"data/Machine/{machine_number}_test.pkl", "rb")
        )
    else:
        label = pickle.load(
            open(f"data/Machine/{data_id}_test_label.pkl", "rb")
        )
        origin_data = pickle.load(
            open(f"data/Machine/{data_id}_test.pkl", "rb")
        )

    print(f"check equal is {torch.all(all_target == torch.Tensor(origin_data)[:all_target.shape[0]] * 20)}")



    label = torch.Tensor(label)
    # print(f"all gen shape is {all_gen.shape}")

    # print(f"all gen is middle shape is {all_gen_middle.shape}")
    # print(f"all target shape is {all_target.shape}")
    return all_gen_middle, label, all_target

def compute_average_residual(prediction, all_target):
    residual = torch.sum(
        (prediction - all_target) ** 2, dim=-1
    )
    average_residual = torch.sum(residual) / len(residual)

    return average_residual.item()

def compute_residual(prediction, all_target,compute_abs=True,compute_sum=True):
    print(f"compute abs is {compute_abs} and compute sum is {compute_sum}")
    if compute_sum and compute_abs:
        residual = torch.sum(
            torch.abs(prediction - all_target), dim=-1
        )
    elif compute_sum and not compute_abs:
        residual = torch.sum(
            (prediction - all_target) ** 2, dim=-1
        )
    elif not compute_sum and compute_abs:
        residual, _ = torch.max(
            torch.abs(prediction - all_target), dim=-1
        )
    elif not compute_sum and not compute_abs:
        residual, _ = torch.max(
            (prediction - all_target) ** 2, dim=-1
        )
    return residual



def ensemble(pkl_path, data_id, ensemble_strategy_list = [],last_step_threshold = 0.02,compute_abs=True,compute_sum=True,machine_number=""):
    if data_id == "SMD" or data_id == "GCP":
        all_gen_middle, label, all_target = merge(pkl_path, data_id,machine_number=machine_number)
    else:
        all_gen_middle, label, all_target = merge(pkl_path,data_id)
    residual_list = []
    average_residual_list = []
    for i in ensemble_strategy_list:
        residual_list.append(compute_residual(
            all_gen_middle[i],all_target, compute_abs,compute_sum
        ))
        average_residual_list.append(compute_average_residual(
            # all_gen_middle[i], all_target, compute_abs,compute_sum
            all_gen_middle[i], all_target

        ))
    # threshold = residual.reshape(-1).topk(int(0.0005 * threshold * len(residual))).values[-1].item()

    true = torch.ones_like(residual_list[0])
    false = torch.zeros_like(residual_list[0])
    origin_prediction = torch.zeros_like(residual_list[0])

    step_prediction_same_proper_list = []
    step_prediction_anomaly_same_proper_list = []


    # origin_prediction = torch.where(residual > threshold, true, false)
    for i, residual in enumerate(residual_list):
        # residual_i * proper_i = residual_j * proper_j
        proper_i = average_residual_list[0] * last_step_threshold/ (average_residual_list[i]) # 选出一个适当的阈值
        print(f"proper is {proper_i}")
        proper_number = max(int(proper_i *  len(residual)),1)

        threshold = residual.reshape(-1).topk(proper_number).values[-1].item()

        step_prediction = torch.where(residual > threshold, true, false)
        if i == 0:
            first_step_prediction = step_prediction

        same_number = sum(step_prediction == first_step_prediction)
        same_anomaly_number = sum(step_prediction * first_step_prediction)
        same_proper = same_number / len(step_prediction)
        same_anomaly_proper = same_anomaly_number / sum(step_prediction)

        step_prediction_same_proper_list.append(same_proper.item())
        step_prediction_anomaly_same_proper_list.append(same_anomaly_proper.item())


        origin_prediction += step_prediction

    best_f = -1
    for ensemble_threshold in range(0,len(residual_list)):
        prediction = torch.where(
            origin_prediction > ensemble_threshold, true,false
        )
        add_list = compute_add(origin_prediction, label)
        add_value = sum(add_list) / len(add_list)


        adjust_prediction = prediction_adjust(prediction,label)
        p,r,f = compute_f(adjust_prediction,label)



        if f > best_f:
            best_f = f
            best_p = p
            best_r = r
            best_add = add_value
            best_ensemble_threshold = ensemble_threshold
            print(f"best f update and its value is {best_f.item(),p.item(),r.item()}")
    return [best_p,best_r,best_f,best_add, best_ensemble_threshold] + [torch.std(torch.Tensor(average_residual_list)).item()] + average_residual_list,\
           step_prediction_same_proper_list, torch.std(torch.Tensor(step_prediction_same_proper_list)).item(), \
           step_prediction_anomaly_same_proper_list, torch.std(torch.Tensor(step_prediction_anomaly_same_proper_list)).item()



def compute_one_strategy(data_id,strategy_name,ensemble_strategy_list,csv_writer,last_step_threshold=0.02):
    print(f"ensemble for {data_id} in {strategy_name} ...")
    csv_writer.writerow([data_id,strategy_name])
    # for default setting
    compute_abs = True
    compute_sum = True

    if data_id == "MSL":
        compute_abs = True
        compute_sum = True
    if data_id == "PSM":
        compute_abs = True
        compute_sum = False
    if data_id == "SMAP":
        compute_abs = True
        compute_sum = True
    if data_id == "SWaT":
        compute_abs = True
        compute_sum = True

    if data_id == "SMD" or data_id == "GCP":
        compute_abs = True
        compute_sum = False

    if data_id == "SMD":
        machine_number_list = [f"machine-1-{i}" for i in range(1, 9)]
        machine_number_list += [f"machine-2-{i}" for i in range(1,10)]
        machine_number_list += [f"machine-3-{i}" for i in range(1,12)]
        # machine_number_list = [f"machine-1-5"]
        threshold_dict = {}
        csv_reader = csv.reader(open("score/SMD/infor.csv"))
        # load threshold dict for each subset
        for line in csv_reader:
            if line[1] == 'False' and line[2] == 'True':
                pass
            else:
                continue

            threshold_dict[line[0]] = float(line[-1])

        for machine_number in machine_number_list:

            iter_result_list = []
            pkl_path_list = []
            for save_file in os.listdir(f"window_result/"):
                if "save" not in save_file:
                    continue
                for data_file in os.listdir(f"window_result/{save_file}/50/"):
                    if machine_number +"_" not in data_file or "unconditional" not in data_file:
                        continue
                    base_path = f"window_result/{save_file}/50/{data_file}"
                    # print(base_path)
                    for pkl_path in os.listdir(base_path):
                        if ".pk" in pkl_path:
                            pkl_path_list.append(
                                f"{base_path}/{pkl_path}"
                            )
            print(f'length of pkl path list is {len(pkl_path_list)}')
            # print(pkl_path_list)
            for item in pkl_path_list:
                print(item)
            last_step_threshold = threshold_dict[machine_number]
            print(f"now threshold for {machine_number} is {last_step_threshold}")
            for pkl_path in pkl_path_list:
                result, same_list, same_std, same_anomaly_list, same_anomaly_std = ensemble(pkl_path, data_id,
                                                                                            ensemble_strategy_list,
                                                                                            last_step_threshold,
                                                                                            compute_abs, compute_sum,machine_number=machine_number)
                result = list(result)
                iter_result_list.append(result)
                # csv_writer.writerow([compute_abs, compute_sum] + result)
                # csv_writer.writerow(same_list + [same_std])
                # csv_writer.writerow(same_anomaly_list + [same_anomaly_std])
                # csv_writer.writerow([])
            iter_result_tensor = torch.Tensor(iter_result_list)
            average = iter_result_tensor.mean(0).tolist()
            f_std = torch.std(iter_result_tensor[:, 0])
            csv_writer.writerow([f"average for {machine_number}"] + average)

    elif data_id == "GCP":
        machine_number_list = [f"service{i}" for i in range(0, 30)]
        threshold_dict = {}
        csv_reader = csv.reader(open(f"score/{data_id}/infor.csv"))
        for line in csv_reader:
            if line[1] == 'False' and line[2] == 'True':
                pass
            else:
                continue

            threshold_dict[line[0]] = float(line[-1])

        for save_file in os.listdir(f"window_result/"):
            if "save" not in save_file:
                continue


            iter_result_list = []
            for machine_number in machine_number_list:
                pkl_path_list = []
                for data_file in os.listdir(f"window_result/{save_file}/50/"):
                    if machine_number + "_" not in data_file or "unconditional" not in data_file:
                        continue
                    base_path = f"window_result/{save_file}/50/{data_file}"
                    # print(base_path)
                    for pkl_path in os.listdir(base_path):
                        if ".pk" in pkl_path:
                            pkl_path_list.append(
                                f"{base_path}/{pkl_path}"
                            )
                print(f'length of pkl path list is {len(pkl_path_list)}')
                last_step_threshold = threshold_dict[machine_number]
                print(f"now threshold for {machine_number} is {last_step_threshold}")
                for pkl_path in pkl_path_list:
                    print(f"now machine number is {machine_number}")
                    result, same_list, same_std, same_anomaly_list, same_anomaly_std = ensemble(pkl_path, data_id,
                                                                                                ensemble_strategy_list,
                                                                                                last_step_threshold,
                                                                                                compute_abs, compute_sum,
                                                                                                machine_number=machine_number)
                    result = list(result)

                    iter_result_list.append(result)

            iter_result_tensor = torch.Tensor(iter_result_list)
            average = iter_result_tensor.mean(0).tolist()
            csv_writer.writerow([f"average for {save_file}"] + average)


    else:

        iter_result_list = []
        pkl_path_list = []
        for save_file in os.listdir(f"window_result/"):
            if "save" not in save_file:
                continue
            for data_file in os.listdir(f"window_result/{save_file}/50/"):
                if data_id not in data_file or "unconditional" not in data_file:
                    continue
                base_path = f"window_result/{save_file}/50/{data_file}"
                # print(base_path)
                for pkl_path in os.listdir(base_path):
                    if ".pk" in pkl_path:
                        pkl_path_list.append(
                            f"{base_path}/{pkl_path}"
                        )
        print(f'length of pkl path list is {len(pkl_path_list)}')
        for pkl_path in pkl_path_list:
            result, same_list, same_std, same_anomaly_list, same_anomaly_std = ensemble(pkl_path,data_id,ensemble_strategy_list,last_step_threshold,compute_abs,compute_sum)
            result = list(result)
            iter_result_list.append(result)
            csv_writer.writerow([compute_abs,compute_sum] + result)
            csv_writer.writerow(same_list + [same_std])
            csv_writer.writerow(same_anomaly_list + [same_anomaly_std])
            csv_writer.writerow([])
        iter_result_tensor = torch.Tensor(iter_result_list)
        average = iter_result_tensor.mean(0).tolist()
        f_std = torch.std(iter_result_tensor[:,0])
        csv_writer.writerow(['average'])
        csv_writer.writerow(
            ['p','r','f1','add']
        )
        csv_writer.writerow(average )
        csv_writer.writerow(['std'])
        csv_writer.writerow(
            ['p', 'r', 'f1', 'add']
        )
        csv_writer.writerow(
            iter_result_tensor.std(0).tolist()
        )


def compute_one_data(data_id):
    strategy_dict = {
        # "base" : list(range(0,1)),
        # "10-strategy":list(range(0,10)),
        # "15-strategy": list(range(0, 15)),
        # "20-strategy": list(range(0, 20)),
        # "30-strategy": list(range(0, 30)),
        # "30-5-skip-strategy": list(range(0, 30,5)),
        "30-3-skip-strategy": list(range(0, 30, 3)),
    }
    os.makedirs("ensemble_residual",exist_ok=True)

    csv_writer = csv.writer(open(f"ensemble_residual/{data_id}.csv","w"))
    # for ensemble_threshold in range(0,10):
    for key in strategy_dict.keys():
        strategy_name = key
        compute_one_strategy(data_id,strategy_name,
                             strategy_dict[strategy_name],
                             csv_writer)

if __name__ =="__main__":
    # for ensemble_threshold in range(0,10):
    # compute_one_data("PSM")
    # compute_one_data("MSL")
    # compute_one_data("SMD")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SMD")

    args = parser.parse_args()
    compute_one_data(args.dataset_name)


