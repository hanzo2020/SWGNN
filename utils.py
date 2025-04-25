import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, roc_auc_score, accuracy_score
import time
import uuid
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch

def generate_random_name():
    random_name = str(uuid.uuid4())
    return random_name


def numpy_to_torch(d, requires_grad=True):
    t = torch.from_numpy(d)
    if d.dtype is 'float32':
        t.requires_grad = requires_grad
    return t

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"函数 {func.__name__} 运行时间: {run_time:.4f} 秒")
        return result
    return wrapper

def paint_roc(y_true, y_score):
    # 计算ROC曲线数据
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 随机分类器的对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # 保存为文件，修改文件名和路径
    plt.savefig("roc_curve.png", dpi=300)  # 保存为PNG文件，可以选择其他格式如jpg, pdf等
    plt.close()  # 关闭图形，避免占用内存

def calculate_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    cf_matrix = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=[0.0, 1.0])
    auprc = average_precision_score(y_true=y_true, y_score=y_score)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return acc, cf_matrix, auprc, f1, auc


def minmax(input, axis=1):
    """
    Do minmax normalization for input 2d-ndarray.

    Parameters:
    ----------
    input:       np.ndarray.
                The input 2d-ndarray.
    axis:       int, default=1.
                The axis should be normalized. Default is 1, that is do normalization along with the column.

    Returns:
    A ndarray after minmax normalization.
    """
    scaler = MinMaxScaler()
    if axis == 1:
        output = scaler.fit_transform(input)
    elif axis == 0:
        output = scaler.fit_transform(input.T).T

    return output


def print_config(filename, configs):
    with open(filename, 'a') as f:
        for key, val in configs.items():
            print(key, ':', val, file=f, flush=True)
            

def sum_norm(input, axis=1):
    """
    Do normalization for an input 2d-ndarray, making the sum of every row or column equals 1.

    Parameters:
    ----------
    input:       ndarray.
                The input 2d-ndarray.
    axis:       int, default=1.
                The axis should be normalized. Default is 1, that is do normalization along with the column.

    Returns:
    A ndarray after normalization.
    """
    axis_sum = input.sum(axis=1-axis, keepdims=True)
    return input / axis_sum

def get_all_nodes(pan=False):
    dir = "pan_data/Gene-Name.txt" if pan else "data/Gene-Name.txt" 
    with open(dir) as f:
        txt = f.readlines()
    gene_list = [line.strip() for line in txt[1:]]  # skip the first line
    gene_set = set(gene_list)

    return gene_list, gene_set


def get_gene_list(rename=False, pan=False):
    dir = "pan_data/Gene-Name.txt" if pan else "data/Gene-Name.txt" 
    gene_list = pd.read_csv(dir)
    return gene_list.rename(columns={'gene_name' : 'Gene Name'}) if rename else gene_list


def get_cell_line(data_dir):
    cell_line = None
    if "Breast_Cancer" in data_dir:
        cell_line = "MCF7"
    elif "Leukemia" in data_dir:
        cell_line = "K562"
    elif 'Pan' in data_dir:
        cell_line = 'Pan'
    elif 'Lung' in data_dir:
        cell_line = 'A549'
    else:
        print(f"Invalid directory {data_dir}.")
    return cell_line


def get_node_idx(node_list):
    """
    Get node indices from node name list.

    Parameters:
    ----------
    node_list:  list.
                Node name list.

    Returns:
    Node indices list.
    """
    gene_list = pd.read_csv("data/Gene-Name.txt")
    gene_list.set_index('gene_name', inplace=True)
    gene_index = np.arange(gene_list.shape[0])
    gene_list['gene_index'] = gene_index
    node_idx_list = []
    for node in node_list:
        if node not in gene_list.index:
            print(f"{node} is not in gene list!")
            continue
        node_idx = gene_list.loc[node, 'gene_index']
        node_idx_list.append(node_idx)

    return node_idx_list

def get_node_name(node_idx_list):
    """
    Get node name from node indices list.

    Parameters:
    ----------
    node_idx_list:  list.
                    Node indices list.

    Returns:
    Node name list.
    """
    gene_list = pd.read_csv("data/Gene-Name.txt")
    assert all(x >= 0 and x < gene_list.shape[0] for x in node_idx_list), "Elements are out of bounds."

    node_name_list = gene_list.iloc[node_idx_list, 0].to_list()

    return node_name_list


def read_table_to_np(table_file, sep='\t', dtype=float, start_col=1):
    data = pd.read_csv(table_file, sep=sep)
    data = data.iloc[:, start_col:].to_numpy().astype(dtype)
    return data

def final_print(best_info, args):
    print("best result in model:" + str(args.model))
    print(f"Epoch: {best_info['epoch']}")
    print("\nbest train:")
    print(f"F1: {best_info['train_f1']:.4f}, 准确率: {best_info['train_acc']:.4f}, AUC: {best_info['train_auc']:.4f}, AUPRC: {best_info['train_auprc']:.4f}")
    print("\nbest valid:")
    print(f"F1: {best_info['valid_f1']:.4f}, 准确率: {best_info['valid_acc']:.4f}, AUC: {best_info['val_auc']:.4f}, AUPRC: {best_info['valid_auprc']:.4f}")
    print("\nbest test:")
    print(f"F1: {best_info['test_f1']:.4f}, 准确率: {best_info['test_acc']:.4f}, AUC: {best_info['test_auc']:.4f}, AUPRC: {best_info['test_auprc']:.4f}")
    print(f"best matrix: {best_info['test_confusion_matrix']}")