import sys
import os
from src.utils import *
from collections import defaultdict
import numpy as np


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, dataSet='cora'):
        if dataSet == 'cora':
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = defaultdict(set)
            with open(cora_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    # print(info)
                    assert len(info) == 2
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj_lists)

        elif dataSet == 'pubmed':
            pubmed_content_file = self.config['file_path.pubmed_paper']
            pubmed_cite_file = self.config['file_path.pubmed_cites']

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            with open(pubmed_content_file) as fp:
                fp.readline()
                feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i
                    labels.append(int(info[1].split("=")[1]) - 1)
                    tmp_list = np.zeros(len(feat_map) - 2)
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        tmp_list[feat_map[word_info[0]]] = float(word_info[1])
                    feat_data.append(tmp_list)

            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = defaultdict(set)
            with open(pubmed_cite_file) as fp:
                fp.readline()
                fp.readline()
                for line in fp:
                    info = line.strip().split("\t")
                    paper1 = node_map[info[1].split(":")[1]]
                    paper2 = node_map[info[-1].split(":")[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj_lists)
        elif dataSet == 'MSR':
            msr_content_file = self.config['file_path.MSR_content']
            msr_cite_file = self.config['file_path.MSR_cite']
            feat_data = []
            indexs = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(msr_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    # print(info)
                    feat_data.append([float(x) for x in info[1:-1]])
                    indexs.append([int(x) for x in info[0:1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            indexs = np.asarray(indexs)
            # print(indexs)
            feat_data = np.asarray(feat_data)
            # print(feat_data.shape)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = defaultdict(set)
            with open(msr_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    # print(info)
                    assert len(info) == 2
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            # assert len(feat_data) == len(labels) == len(adj_lists)
            # indexs = feat_data.shape[0]
            # adj_lists, feat_data, labels, indexs = src_upsample(adj_lists, feat_data, labels, indexs, portion=0, im_class_num=1)
            # test_indexs, val_indexs, train_indexs = self._split_data(indexs)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
            # print('featdata.shape[0]::::::', feat_data.shape[0])
            # print(train_indexs)
            # smote
            # print(train_indexs)
            # print(adj_lists)
            # print(feat_data)
            # print(labels)
            # print(labels.shape)
            print("smote前label的长度是", len(labels))
            adj_lists, feat_data, labels, train_indexs = src_upsample(adj_lists, feat_data, labels, train_indexs, portion = 0, im_class_num = 1)
            print("smote后label的长度是", len(labels))

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj_lists)



    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)
        # print(num_nodes)

        test_size = num_nodes // test_split
        # print(test_size)
        val_size = num_nodes // val_split
        # print(val_size)
        train_size = num_nodes - (test_size + val_size)
        # print(train_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]

        return test_indexs, val_indexs, train_indexs


