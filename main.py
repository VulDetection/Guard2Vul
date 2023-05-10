import sys
import os
import torch
import argparse
import pyhocon
import random
from dataload import *
from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='MSR',)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--b_sz', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=128)
parser.add_argument('--cuda', action='store_true',default = 'True',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='margin')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--max_vali_auc', type=float, default=0)
parser.add_argument('--max_vali_recall', type=float, default=0)
parser.add_argument('--max_vali_precision', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='../experiments.conf')
# parser.add_argument('--setting', type=str, default='smote',
#                     choices=['no', 'upsampling', 'smote', 'reweight', 'embed_up', 'recon', 'newG_cls',
#                              'recon_newG'])
args = parser.parse_args()



if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	dataCenter = DataCenter(config)
	dataCenter.load_dataSet(ds)
	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
	graphSage.to(device)

	num_labels = len(set(getattr(dataCenter, ds+'_labels')))
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)

	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')


	import warnings
	warnings.filterwarnings("ignore")

	for epoch in range(args.epochs):
	# for epoch in range(1):
		print('----------------------EPOCH %d-----------------------' % (epoch+1))
		classification.to(device)
		graphSage.to(device)
		graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz,
												args.unsup_loss, device, args.learn_method, args.lr)
		classification, args.max_vali_f1, args.max_vali_auc, args.max_vali_recall, args.max_vali_precision = train_classification(
			dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.max_vali_auc,
			args.max_vali_recall, args.max_vali_precision, args.name, epoch, args.b_sz, args.lr, args.epochs)
		print('***** maxf1: ', args.max_vali_f1, '****** maxauc:', args.max_vali_auc, '****** maxrecall:',
			  args.max_vali_recall, '****** maxprecision:', args.max_vali_precision)
		import csv
		with open(r'result/{}/result-pgddrop3-0.2.csv'.format(args.dataSet), 'a+', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			# for row in range(0, totallensbt):
			# 	# for row in range(0, sbt.shape[0]):
			myList = []
			myList.append(args.max_vali_auc)
			myList.append(args.max_vali_f1)
			myList.append(args.max_vali_precision)
			myList.append(args.max_vali_recall)
			writer.writerow(myList)