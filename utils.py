import sys
import os
import torch
import random
import math
import torch.nn.functional as F
from src.main import *
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torch.nn as nn
import numpy as np
from src.resnet import resnet_block
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1,max_vali_auc, max_vali_recall, max_vali_precision, name, cur_epoch):
    test_nodes = getattr(dataCenter, ds+'_test')
    train_nodes = getattr(dataCenter, ds + '_train')
    val_nodes = getattr(dataCenter, ds+'_val')
    labels = getattr(dataCenter, ds+'_labels')

    models = [graphSage, classification]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    embs = graphSage(train_nodes)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[train_nodes]
    # assert len(labels_val) == len(predicts)
    # comps = zip(labels_val, predicts.data)

    vali_f1 = f1_score(labels_val, predicts.cpu().data,average="weighted")
    vali_recall = recall_score(labels_val, predicts.cpu().data,average="weighted")
    vali_precision = precision_score(labels_val, predicts.cpu().data,average="weighted")

    try:
        vali_auc = roc_auc_score(labels_val, predicts.cpu().data,average="weighted" )
    except ValueError:
        vali_auc = 0


    for param in params:
        param.requires_grad = True

    # torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, vali_f1))

    for param in params:
        param.requires_grad = True

    return vali_f1, vali_auc, vali_recall, vali_precision


def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 256
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

#  smote类不平衡
def src_upsample(adj, features, labels, idx_train, portion = 1.0, im_class_num = 1):
    c_largest = labels.max().item()
    # adj_back = adj.to_dense()
    chosen = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i ))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                    # print(chosen.shape)
                else:
                    # print(chosen.shape)
                    # print(new_chosen.shape)
                    chosen = torch.tensor(chosen)
                    new_chosen = torch.tensor(new_chosen)
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

            num = int(new_chosen.shape[0] * portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((torch.from_numpy(chosen), torch.from_numpy(new_chosen)),
                                   0)

    add_num = chosen.shape[0]

    for i in range(chosen.shape[0]):
        if chosen[i] in adj:
            adj[len(features)+i].add(adj[chosen[i]])
    # ipdb.set_trace()
    features_append = deepcopy(features[chosen, :])
    labels_append = deepcopy(labels[chosen])
    idx_train_append = np.arange(len(features), len(features) + add_num)
    features=np.concatenate((features, features_append), axis=0)
    labels=np.concatenate((labels, labels_append), axis=0)
    idx_train=np.concatenate((idx_train, idx_train_append), axis=0)


    return adj, features, labels, idx_train


def src_smote(adj, features, labels, idx_train, portion = 1.0, im_class_num = 1):
    c_largest = labels.max().item()
    # print(c_largest)
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        # print("new:",new_chosen)
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])
            print("c_po:",c_portion)
            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            print("c_po:",c_portion)
            portion_rest = portion - c_portion
            print(portion_rest)

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis = -1)

            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen, :]
        distance = squareform(pdist(chosen_embed.detach()))
        np.fill_diagonal(distance, distance.max() + 100)

        idx_neighbor = distance.argmin(axis = -1)

        interp_place = random.random()
        # print("chosen_embed:",chosen_embed)
        # print(chosen_embed.shape)
        # print(idx_neighbor)
        embed = chosen_embed + \
                (chosen_embed[idx_neighbor, :]
                 - chosen_embed) * interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    # print("append idtrain:", len(idx_train))
    adj = new_adj.to_sparse()
    return adj, features, labels, idx_train


def recon_upsample(embed, labels, idx_train, adj = None, portion = 1.0, im_class_num = 3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis = -1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min = 0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min = 0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)


# FGSM
class FGSM:
    def __init__(self, model: nn.Module, eps=0.1):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # 只攻击词embedding层
    def attack(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()		#使用sign（符号）函数，将对x求了偏导的梯度进行符号化
                param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

#  对抗训练 FGM
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


#  对抗训练 FGM-L1
class FGML1():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad, p=1)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 对抗训练FreeLB
class FreeLB(object):
    """
    Example
    model =
    loss_fun =
    freelb = FreeLB(loss_fun,adv_K=3,adv_lr=1e-2,adv_init_mag=2e-2)
    for batch_input, batch_label in data:
        inputs = {'input_ids':...,...,'labels':batch_label}
        #freelb.attack中进行了多次loss.backward()
        loss = freelb.attack(model,inputs)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    """
    def __init__(self, loss_fun, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0., adv_norm_type='l2',
                 base_model='bert'):
        """
        初始化
        :param loss_fun: 任务适配的损失函数
        :param adv_K: 每次扰动对抗的小步数，最少是1 一般是3
        :param adv_lr: 扰动的学习率1e-2
        :param adv_init_mag: 初始扰动的参数 2e-2
        :param adv_max_norm:0  set to 0 to be unlimited 扰动的大小限制 torch.clamp()等来实现
        :param adv_norm_type: ["l2", "linf"]
        :param base_model: 默认的bert
        """
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag  # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
        self.loss_fun = loss_fun

    def attack(self, model, inputs, labels, gradient_accumulation_steps=1):
        # model 可以放在初始化中

        input_ids = inputs['input_ids']

        # 得到初始化的embedding
        # 从bert模型中拿出embeddings层中的word_embeddings来进行input_ids到embedding的变换
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
            # embeds_init = model.encoder.embeddings.word_embeddings(input_ids)

        if self.adv_init_mag > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            # 类型和设备转换
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化

        for astep in range(self.adv_K):
            delta.requires_grad_()
            # bert transformer类模型在输入的时候inputs_embeds 和 input_ids 二选一 不然会报错。。。。。。源码
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            inputs['input_ids'] = None

            # 下游任务的模型，我这里在模型输出没有给出loss 要自己计算原始loss
            logits = model(inputs)
            loss = self.loss_fun(logits, labels)
            loss = loss / self.adv_K

            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if astep == self.adv_K - 1:
                # further updates on delta
                break

            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad

            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)

        return loss

# 对抗训练PGD
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class PGDdrop():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                # Use dropout regularization
                if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Dropout(p=0.2)):
                    self.model.eval()
                    with torch.no_grad():
                        param.data = self.model[-2](param.data)
                        param.data = self.model[-1](param.data)
                # If not using dropout, apply PGD attack as usual
                else:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = alpha * param.grad / norm
                        param.data.add_(r_at)
                        param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


criterion = torch.nn.CrossEntropyLoss()

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, max_vali_auc, max_vali_recall, max_vali_precision, name, epochs,b_sz, lr,epoches):
    classification.to(device)
    graphSage.to(device)
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr)
    # train classification, detached from the current graph
    #classification.init_params()
    # b_sz = 5
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds)
    for epoch in range(1):
        train_nodes = shuffle(train_nodes)
        # print(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            # print(labels_batch)
            embs_batch = features[nodes_batch]


            logists = classification(embs_batch)          #常规


            # # 常规
            # loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss /= len(nodes_batch)
            # # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            # loss.backward()

            # # fgm对抗
            # fgm = FGM(graphSage)
            # # fgm = FGM(classification)
            # # print("fgm 对抗训练开始：")
            # fgm.attack()  #duikang
            # # print("fgm 对抗训练开始：")
            # loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss /= len(nodes_batch)
            # # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            # loss.backward()
            # fgm.restore()#duikang


            # # fgmL1对抗
            # fgml1 = FGML1(graphSage)
            # # print("fgm 对抗训练开始：")
            # fgml1.attack()  #duikang
            # # print("fgm 对抗训练开始：")
            # loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss /= len(nodes_batch)
            # # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            # loss.backward()
            # fgml1.restore()#duikang

            # # FGM-dropout
            # class FGMdrop():
            #     def __init__(self, model):
            #         # Initializes the FGM attack class with the model to be attacked
            #         self.model = model
            #         # Creates an empty dictionary to store the original parameters of the model
            #         self.backup = {}
            #
            #     def attack(self, epsilon=1., emb_name='emb.'):
            #         # emb_name is the name of the embedding layer in the model, which needs to be specified
            #         # Iterates through each module in the model
            #         for module in self.model.modules():
            #             # Checks if the module is an instance of nn.Dropout
            #             if isinstance(module, nn.Dropout):
            #                 # Disables dropout during the attack phase
            #                 module.training = False
            #
            #         # Executes a forward pass through the model
            #         self.model.train()
            #         # outputs = self.model(inputs)
            #
            #         # Computes the loss and performs backpropagation
            #         # loss = criterion(outputs, targets)
            #         loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            #         loss /= len(nodes_batch)
            #         self.model.zero_grad()
            #         loss.backward()
            #
            #         # emb_name is the name of the embedding layer in the model, which needs to be specified
            #         # Iterates through each parameter in the model
            #         for name, param in self.model.named_parameters():
            #             # Checks if the parameter requires gradients and is in the embedding layer
            #             if param.requires_grad and emb_name in name:
            #                 # Saves the original parameter values
            #                 self.backup[name] = param.data.clone()
            #                 # Applies dropout to the parameter
            #                 F.dropout(param, p=epsilon, inplace=True)
            #
            #     def restore(self, emb_name='emb.'):
            #         # emb_name is the name of the embedding layer in the model, which needs to be specified
            #         # Iterates through each parameter in the model
            #         for name, param in self.model.named_parameters():
            #             # Checks if the parameter requires gradients and is in the embedding layer
            #             if param.requires_grad and emb_name in name:
            #                 # Checks if the parameter was saved in the backup dictionary
            #                 assert name in self.backup
            #                 # Restores the original parameter values
            #                 param.data = self.backup[name]
            #         # Resets the backup dictionary
            #         self.backup = {}
            # # fgmdropout对抗
            # fgmdrop = FGMdrop(graphSage)
            # # print("fgm 对抗训练开始：")
            # fgmdrop.attack()  #duikang
            # # print("fgm 对抗训练开始：")
            # loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss /= len(nodes_batch)
            # # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            # # loss.backward()
            # fgmdrop.restore()#duikang

            # # fgsm对抗
            # fgsm = FGSM(graphSage)
            # # print("fgm 对抗训练开始：")
            # fgsm.attack()  #duikang
            # # print("fgm 对抗训练开始：")
            # loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # loss /= len(nodes_batch)
            # # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            # loss.backward()
            # fgsm.restore()#duikang

            # # pgd对抗
            # pgd=PGD(graphSage)
            # # print("PDG")
            # K = 1
            # pgd.backup_grad()
            # for t in range(K):
            # 	pgd.attack(is_first_attack=(t==0))
            # 	if t != K - 1 :
            # 		graphSage.zero_grad()
            # 	else:
            # 		pgd.restore_grad()
            # 	loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            # 	loss /= len(nodes_batch)
            # 	# print("pgd 对抗训练结束：")
            # 	# loss.backward()
            # pgd.restore()

            # pgddrop对抗
            pgddrop = PGDdrop(graphSage)
            K = 3
            pgddrop.backup_grad()
            for t in range(K):
                pgddrop.attack(is_first_attack=(t == 0))
                graphSage.zero_grad()
                loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                loss /= len(nodes_batch)
                # loss.backward()
                pgddrop.restore_grad()
            pgddrop.restore()

            # ##  FreeLB对抗
            # loss_fun = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)/(len(nodes_batch))
            # model = graphSage
            # freelb = FreeLB(loss_fun, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2)
            # for batch_input, batch_label in embed1:
            #     inputs = {'input_ids':...,...,'labels':batch_label}
            #     # freelb.attack中进行了多次loss.backward()
            #     loss = freelb.attack(model, inputs)
            #     loss.backward()
            #     c_optimizer.step()
            #     model.zero_grad()


            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            print('now training epoch:{} Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epochs+1,index + 1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
        max_vali_f1, max_vali_auc, max_vali_recall, max_vali_precision = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, max_vali_auc, max_vali_recall, max_vali_precision, name, epoch)
        print("auc:{} f1:{} precision:{} recall:{} ".format(max_vali_auc,max_vali_f1,max_vali_precision,max_vali_recall))
    return classification, max_vali_f1, max_vali_auc, max_vali_recall, max_vali_precision

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method, lr):
    classification.to(device)
    graphSage.to(device)
    test_nodes = getattr(dataCenter, ds+'_test')
    val_nodes = getattr(dataCenter, ds+'_val')
    train_nodes = getattr(dataCenter, ds+'_train')
    labels = getattr(dataCenter, ds+'_labels')
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.SGD(params, lr)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    batches = math.ceil(len(train_nodes) / b_sz)

    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)

        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        embs_batch = graphSage(torch.from_numpy(nodes_batch).to(device))

        if learn_method == 'sup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage, classification
