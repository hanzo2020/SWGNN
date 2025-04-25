import os
import sys
import argparse
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.loader import NeighborLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import utils
import config_load
from model import *
from loss import UnionLoss
from data_preprocess import get_data, CancerDataset
#python main.py -g 2 -t --lr 0.002 --seed 1999
def arg_parse():
    parser = argparse.ArgumentParser(description="Train GATRes arguments.")
    parser.add_argument('--data_dir', type=str, default='data/Lung_Cancer_Matrix',)
    parser.add_argument('-f', "--fold", dest='fold', help="fold", default=0, type=int)
    parser.add_argument('-g', '--gpu', dest='gpu', default=3)
    parser.add_argument('-m', "--model", dest='model', default='Multi_GTN')
    parser.add_argument('-t', "--train", dest="train", action="store_true")
    parser.add_argument('-p', "--predict_unkonwn", dest="predict_unkonwn", action="store_true")
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=2025)
    return parser.parse_args()

def set_seed(random_seed=2025):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #torch.use_deterministic_algorithms(True)
    print('set random_seed:' + str(random_seed))

class MultiPPI_Encoder():
    def __init__(self, args) -> None:
        assert args['mode'] in ['train', 'pred', 'encode']

        self.args = args
        self.mode = args['mode']
        self.dataset = get_data(args, args['stable'])
        self.num_node_features = self.dataset.num_node_features
        self.log_name = args['log_file'].split('/')[-1].split('.')[0]
        ppi_dict = {
            'CPDB': 1,
            'IRef': 2,
            'Multinet': 3,
            'PCNet': 4,
            'STRING': 5,
        }

        self.drop_idx = []
        self.loss_func = torch.nn.BCELoss() if args['loss'] == 'BCE' \
            else UnionLoss(
            pos_lambda=args['pos_lambda'],
            neg_lambda=args['neg_lambda'],
            func=args['loss'],
            start_epoch=args['start_epoch'],)
        

        

    def drop_samples(self, fold, sample_neg=0., sample_pos=1., num_samples=0):
        assert len(self.drop_idx) == 0
        self.args["drop_neg"] = 0. if self.args['mode'] == 'pred' else self.args["drop_neg"]
        print(f"Drop {self.args['drop_neg']} of negative train samples in fold {fold}")

        if sample_neg == 1 and sample_pos == 1:
            return []
        drop_neg = 1 - sample_neg
        drop_pos = 1 - sample_pos
        splitted_idx = self.dataset.get_idx_split(fold)
        train_idx = splitted_idx['train']
        drop_neg_idx, drop_pos_idx = [], []
        for i in train_idx:
            if self.dataset[0].y[i][0]:
                drop_neg_idx.append(i.item())
            if self.dataset[0].y[i][1]:
                drop_pos_idx.append(i.item())
        num_neg_samples = len(drop_neg_idx)
        num_pos_samples = len(drop_pos_idx)
        random.seed(self.args['random_seed'])
        if num_samples:
            num_neg = int(num_samples * num_neg_samples /
                        (num_neg_samples + num_pos_samples))
            num_pos = num_samples - num_neg
            drop_neg_idx = random.sample(drop_neg_idx, num_neg_samples - num_neg)
            drop_pos_idx = random.sample(drop_pos_idx, num_pos_samples - num_pos)
        else:
            drop_neg_idx = random.sample(
                drop_neg_idx, int(num_neg_samples*drop_neg))
            drop_pos_idx = random.sample(
                drop_pos_idx, int(num_pos_samples*drop_pos))
        drop_idx = self.drop_idx = drop_neg_idx + drop_pos_idx
        print(
            f"Negatives: {num_neg_samples - len(drop_neg_idx)}, Positives: {num_pos_samples - len(drop_pos_idx)}")
        
        for i in range(len(self.dataset)):
            self.dataset[i].train_mask[drop_idx, fold] = False
    

    def recover_drop(self, fold):
        for i in range(len(self.dataset)):
            self.dataset[i].train_mask[self.drop_idx, fold] = True
            self.drop_idx = []


    def load_data(self, fold):
        self.train_loader_list, self.valid_loader_list, self.test_loader_list, self.unknown_loader_list = [], [], [], []

        for data in self.dataset:
            data.contiguous()
            if self.mode == 'train':
                train_mask = data.train_mask[:, fold]

                self.valid_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    subgraph_type='induced',
                    input_nodes=data.valid_mask[:, fold]))
                
                self.test_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    subgraph_type='induced',
                    input_nodes=data.test_mask[:, fold],))
                
            self.train_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    subgraph_type='induced',
                    input_nodes=train_mask))
            
            self.unknown_loader_list.append(NeighborLoader(
                    data=data,
                    num_neighbors=self.args['num_neighbors'],
                    batch_size=self.args['batch_size'],
                    subgraph_type='induced',
                    input_nodes=data.unlabeled_mask))


    def init_model(self):
        args = self.args
        if 'Multi' in args['model']:
            model = self.model = Multi_GTN(
                gnn=args['model'],
                in_channels=self.num_node_features, 
                hidden_channels=args['hidden_channels'], 
                heads=args['heads'], 
                drop_rate=args['drop_rate'],
                attn_drop_rate=args['attn_drop'], 
                edge_dim=self.dataset[0].edge_dim,
                num_ppi=len(self.dataset),
                pooling=args['pooling'],
                residual=args['residual'],
                learnable_weight=args['learnable_factor']).to(args['gpu'])

        elif 'DSGNN' == args['model']:
            model = self.model = DSGNN(
                in_channels=self.num_node_features,
                hidden_channels=64,
                num_ppi=len(self.dataset),
                residual=args['residual'],
                learnable_weight=args['learnable_factor'],
                args=args).to(args['gpu'])


        self.optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.005, lr=args['lr'])

        num_train_steps = args['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.2 * num_train_steps,
            num_training_steps=num_train_steps)
        
    
    def train_epoch(self, epoch):
        self.model.train()
        tot_loss, steps = 0, 0
        constraint_loss = 0
        for data_tuple in zip(*self.train_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            steps += 1
            self.optimizer.zero_grad()
            size = data_tuple[0].batch_size

            if isinstance(self.model, Multi_GTN):
                out, x_list, ppi_weight = self.model(data_tuple)
            elif isinstance(self.model, DSGNN):
                out, x_list, ppi_weight, constraint_dict, degrees_weights = self.model(data_tuple)
            else:
                out = self.model(data_tuple)

            true_lab = data_tuple[0].y[:size, 1]
            out = out.view(-1)
            if isinstance(self.model, Multi_GTN):
                loss = self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch)
            elif isinstance(self.model, DSGNN):
                loss = self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch)
                constraint_loss = constraint_dict['constraint_loss']
                loss += constraint_loss

            del out, true_lab
            loss.backward()
            tot_loss = tot_loss + loss.item()
            self.optimizer.step()
        self.scheduler.step()
        tot_loss = tot_loss / steps

        self.model.eval()
        y_true = np.array([])
        y_score = np.array([])
        y_pred = np.array([])
        for data_tuple in zip(*self.train_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, _, _ = self.model(data_tuple)
                elif isinstance(self.model, DSGNN):
                    out, _, _, _, _ = self.model(data_tuple)

            true_lab = data_tuple[0].y[:size][:, 1]
            out = out.view(-1)

            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
            y_pred = np.append(y_pred, pred_lab, axis=0)
            y_true = np.append(y_true, true_lab.cpu().detach().numpy())

        train_acc, _, auprc, train_f1, train_auc = utils.calculate_metrics(y_true, y_pred, y_score)
        print(f"Epoch: {epoch}, Train Loss: {tot_loss:.6f}, F1: {train_f1:.4f}, L2: {constraint_loss:.6f}")
        
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        valid_loss = 0
        steps = 0
        ppi_weight = None
        for data_tuple in zip(*self.valid_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            steps = steps + 1
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, x_list, ppi_weight = self.model(data_tuple)
                elif isinstance(self.model, DSGNN):
                    out, x_list, ppi_weight, constraint_dict, _ = self.model(data_tuple)

            true_lab = data_tuple[0].y[:size][:, 1]
            out = out.view(-1)
            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1
            y_pred = np.append(y_pred, pred_lab)
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
            y_true = np.append(y_true, true_lab.cpu().detach().numpy())
            if isinstance(self.model, Multi_GTN):
                valid_loss += self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch).item()
            elif isinstance(self.model, DSGNN):
                valid_loss += self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']), epoch).item()
                constraint_loss = constraint_dict['constraint_loss']
                valid_loss += constraint_loss * 1e-5
            valid_loss += loss
    

        valid_loss = valid_loss / steps
        acc, cf_matrix, auprc, f1, val_auc = utils.calculate_metrics(y_true, y_pred, y_score)
        print(f"Epoch: {epoch}, Valid Loss: {valid_loss:.6f}, F1: {f1:.4f}, TP: {cf_matrix[1, 1]}")


        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        test_loss = 0
        steps = 0
        ppi_weight = None
        for data_tuple in zip(*self.test_loader_list):
            data_tuple = [data.to(self.args['gpu']) for data in data_tuple]
            steps = steps + 1
            size = data_tuple[0].batch_size
            with torch.no_grad():
                if isinstance(self.model, Multi_GTN):
                    out, x_list, ppi_weight = self.model(data_tuple)
                elif isinstance(self.model, DSGNN):
                    out, x_list, ppi_weight, constraint_dict, _ = self.model(data_tuple)

            true_lab = data_tuple[0].y[:size][:, 1]
            out = out.view(-1)
            pred_lab = np.zeros(size)
            if size == 1:
                pred_lab[0] = 1 if out[0] > 0.5 else 0
            else:
                pred_lab[out.cpu() > 0.5] = 1
            y_pred = np.append(y_pred, pred_lab)
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
            y_true = np.append(y_true, true_lab.cpu().detach().numpy())
            if isinstance(self.model, Multi_GTN):
                test_loss += self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']),
                                             epoch).item()
            elif isinstance(self.model, DSGNN):
                test_loss += self.loss_func(out, true_lab.float(), x_list, ppi_weight.to(self.args['gpu']),
                                             epoch).item()
                constraint_loss = constraint_dict['constraint_loss']
                test_loss += constraint_loss * 1e-5
            test_loss += loss

        test_loss = test_loss / steps
        test_acc, test_cf_matrix, test_auprc, test_f1, test_auc = utils.calculate_metrics(y_true, y_pred, y_score)
        print(f"Epoch: {epoch}, test F1: {test_f1:.4f}, ACC: {test_acc:.4f}, " \
              f"AUROC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, TP: {test_cf_matrix[1, 1]}")

        out_dict = {
            'train_loss': tot_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss,
            'train_f1': train_f1,
            'valid_f1': f1,
            'test_f1': test_f1,
            'train_acc': train_acc,
            'valid_acc': acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_auprc': auprc,
            'valid_auprc': auprc,
            'test_auprc': test_auprc,
            'train_confusion_matrix': cf_matrix,
            'valid_confusion_matrix': cf_matrix,
            'test_confusion_matrix': test_cf_matrix,
            'ppi_weight': ppi_weight,
            'y_true': y_true,
            'y_score': y_score,
        }
        return out_dict


            

    def train_single_fold(self, fold, print_args, ckpt=None,):
        self.init_model()
        cell_line = utils.get_cell_line(configs['data_dir'])
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt)['state_dict'])
        self.drop_samples(fold, (1 - self.args['drop_neg']))
        self.load_data(fold)
        early_stop = 0
        log_path = os.path.dirname(self.args['log_file'])
        print(f"Log Path is /{log_path}/{self.log_name}")
        Id = utils.generate_random_name()
        if print_args:
            utils.print_config(self.args['log_file'], self.args)
            with open(self.args['log_file'], 'a') as f:
                print(f"Train: {self.dataset[0].train_mask[:, fold].sum().item()} "\
                      f"Valid: {self.dataset[0].valid_mask[:, fold].sum().item()} "\
                      f"Test: {self.dataset[0].test_mask[:, fold].sum().item()} ",
                      file=f)
                print(self.model, file=f, flush=True)
        with open(self.args['log_file'], 'a') as f:
                print(f"Id:{Id}, Model: {self.args['model']}", file=f)

        print("Start training")
        vmax_f1 = 0
        pos_lambda = self.args['pos_lambda']
        neg_lambda = self.args['neg_lambda']
        best_info = {
            'epoch': 0,  # 最佳 F1 分数对应的 epoch
            'train_f1': 0.0,  # 最佳训练集 F1 分数
            'valid_f1': 0.0,  # 最佳验证集 F1 分数
            'test_f1': 0.0,  # 最佳测试集 F1 分数
            'train_acc': 0.0,  # 最佳训练集准确率
            'valid_acc': 0.0,  # 最佳验证集准确率
            'test_acc': 0.0,  # 最佳测试集准确率
            'train_auc': 0.0,  # 最佳训练集 AUC
            'val_auc': 0.0,  # 最佳验证集 AUC
            'test_auc': 0.0,  # 最佳测试集 AUC
            'train_auprc': 0.0,  # 最佳训练集 AUPRC
            'valid_auprc': 0.0,  # 最佳验证集 AUPRC
            'test_auprc': 0.0,  # 最佳测试集 AUPRC
            'test_confusion_matrix': None,  # 最佳测试集混淆矩阵
            'ppi_weight': None,  # 最佳 PPI 权重
            'y_true': None,
            'y_score': None
        }
        for epoch in range(self.args['num_epochs']):
            self.args['pos_lambda'] = 0. if epoch < 100 else pos_lambda
            self.args['neg_lambda'] = 0. if epoch < 100 else neg_lambda
            out_dict = self.train_epoch(epoch)
            if early_stop > 50 and epoch > (self.args['num_epochs'] / 2):
                print('early stop: ' + str(early_stop) + 'epoch' + str(epoch))
                break
            #train_loss, valid_loss, f1, acc, auc, auprc, cf_matrix, train_f1, train_acc, ppi_weight = self.train_epoch(epoch)
            if out_dict['valid_f1'] >= best_info['valid_f1']:
                best_info['epoch'] = (epoch)
                best_info['train_f1'] = out_dict['train_f1']
                best_info['train_acc'] = out_dict['train_acc']
                best_info['train_auc'] = out_dict['train_auc']
                best_info['train_auprc'] = out_dict['train_auprc']
                best_info['valid_f1'] = out_dict['valid_f1']
                best_info['valid_acc'] = out_dict['valid_acc']
                best_info['val_auc'] = out_dict['val_auc']
                best_info['valid_auprc'] = out_dict['valid_auprc']
                best_info['test_f1'] = out_dict['test_f1']
                best_info['test_acc'] = out_dict['test_acc']
                best_info['test_auc'] = out_dict['test_auc']
                best_info['test_auprc'] = out_dict['test_auprc']
                best_info['test_confusion_matrix'] = out_dict['test_confusion_matrix']
                best_info['ppi_weight'] = out_dict['ppi_weight']
                best_info['y_true'] = out_dict['y_true']
                best_info['y_score'] = out_dict['y_score']
                # 将两个数组保存到一个文件中
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                              'scheduler': self.scheduler.state_dict()}
                if not os.path.exists(os.path.join(self.args['out_dir'], self.log_name)):
                    os.mkdir(os.path.join(self.args['out_dir'], self.log_name))
                if not os.path.exists(os.path.join(self.args['out_dir'], self.log_name, self.args['model'])):
                    os.mkdir(os.path.join(self.args['out_dir'], self.log_name, self.args['model']))
                model_dir = os.path.join(self.args['out_dir'], self.log_name, self.args['model'],
                                         f"{datetime.date.today()}_{self.args['model']}.pkl")
                torch.save(checkpoint, model_dir)
                print('best epoch, save to ' + str(model_dir))
                early_stop = 0
            else:
                early_stop += 1
        self.recover_drop(fold)
        utils.final_print(best_info, args)
        return vmax_f1, model_dir



    def train(self, ):
        args = self.args
        if args['model'] != 'DSGNN' or args['model'] != 'ECD_CDGINet':
            args['lr'] = 0.002
        ckpt = None
        print_args = True
        _, model_dir = self.train_single_fold(args['fold'], print_args, ckpt)



if __name__ == "__main__":
    args = arg_parse()
    configs = config_load.get()
    args.data_dir = configs['data_dir']
    args.model = configs['model']
    configs.update(vars(args))
    configs['gpu'] = f'cuda:{args.gpu}'
    configs['fold'] = args.fold
    configs['random_seed'] = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    cell_line = utils.get_cell_line(configs['data_dir'])
    configs['residual'] = 1

    
    if args.train:
        configs['mode'] = 'train'
        configs['log_file'] = os.path.join(configs['log_dir'], f"{cell_line}_{configs['mode']}")
        configs['log_file'] += '.txt'
        encoder = MultiPPI_Encoder(configs)
        encoder.train()
        sys.exit()


