import pickle
import random
import argparse
import numpy as np 
import torch 
from transformers import AutoModel, AutoProcessor
from transformers import AdamW
from model import * 
from dataset import * 
from trainer import Trainer
from torch.utils.data import DataLoader


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--dataset', type=str, default='data_bundle_new.pkl')
	parser.add_argument('--ptm', type=str, default='clip', choices = ['align', 'clip', 'blip'])
	parser.add_argument('--ptm_lr', type=float, default=1e-5)
	parser.add_argument('--model_lr', type=float, default=5e-4)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--weight_decay', type=float, default=1e-7)
	parser.add_argument('--load_path', type=str, default=None)
	parser.add_argument('--load_epoch', type=int, default=0)
	parser.add_argument('--do_train', default=False, action = 'store_true')
	parser.add_argument('--do_eval', default=False, action = 'store_true')
	parser.add_argument('--wandb', default=False, action = 'store_true')
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--save_path', type=str, default='./ckps/')
	parser.add_argument('--con_loss', default=False, action = 'store_true')
	parser.add_argument('--project', type=str)
	parser.add_argument('--method', type=str, choices=['onlyent','CI','CI_EF'])
	parser.add_argument('--con_type', type=str, choices = ['blc', 'all'])
	parser.add_argument('--task', type=str, choices=['classification','ranking'], required=True)
	parser.add_argument('--save_evidence', default=False, action = 'store_true')
	parser.add_argument('--threshold', type=float)


	arg = parser.parse_args()


	if arg.task == "ranking" and arg.method not in ['onlyent', 'CI']:
		parser.error("For ranking, you can only choose method from ['onlyent','CI']")

	if arg.task == 'ranking':
		identifier = 'ptm={}_bs={}_contype={}_conloss={}_method={}_task={}'.format(arg.ptm, arg.batch_size, arg.con_type, arg.con_loss, arg.method, arg.task) 
	elif arg.task == 'classification':
		identifier = 'ptm={}_bs={}_contype={}_conloss={}_method={}_threshold={}_task={}'.format(arg.ptm, arg.batch_size, arg.con_type, arg.con_loss, arg.method, arg.threshold, arg.task) 


	random.seed(arg.seed)
	np.random.seed(arg.seed)
	torch.manual_seed(arg.seed)

	device = torch.device("cuda:{}".format(arg.gpu) if torch.cuda.is_available() else "cpu")

	# Load dataset
	with open(f"resources/{arg.dataset}", 'rb') as f:
		data_bundle = pickle.load(f) 


	hypers = data_bundle['hypers'] 
	concepts = data_bundle['cons'] 
	con2id = { c: i for i, c in enumerate(sorted(concepts))} 
	id2con = { i: c for i, c in enumerate(sorted(concepts))} 


	ptm_name = {
		"clip": "openai/clip-vit-base-patch32",
		"align": "kakaobrain/align-base",
		"blip": "Salesforce/blip-image-captioning-base"
	}
	ptm_model = AutoModel.from_pretrained(ptm_name[arg.ptm])
	processor = AutoProcessor.from_pretrained(ptm_name[arg.ptm])
	model = MyModel(ptm_model, processor.tokenizer, n_con=len(concepts), device=device,  con_type = arg.con_type,
		id2con=id2con, model_name=arg.ptm, method = arg.method)

	if arg.task=='classification':
		train_data = MyDataset_classifi(data_bundle['trainset'],processor)
		valid_data = MyDataset_classifi(data_bundle['validset'],processor)
		test_data = MyDataset_classifi(data_bundle['testset'],processor)
	elif arg.task=='ranking':
		train_data = MyDataset_ranking(data_bundle['trainset'],processor)
		valid_data = MyDataset_ranking(data_bundle['validset'],processor)
		test_data = MyDataset_ranking(data_bundle['testset'],processor)


	train_loader = DataLoader(dataset = train_data, batch_size=arg.batch_size, shuffle=True)
	valid_loader = DataLoader(dataset = valid_data, batch_size=arg.batch_size * 2, shuffle=True)
	test_loader = DataLoader(dataset = test_data, batch_size=arg.batch_size * 2, shuffle=True)

	data_loaders = {
		'train': train_loader,
		'valid': valid_loader,
		'test': test_loader,
		'concepts': concepts,
		'hypers': hypers,
		'con2id': con2id,
		'id2con': id2con
	}



	no_decay = ["bias", "LayerNorm.weight"]
	param_group = [
		{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
									if ('encoder' not in n) and
									(not any(nd in n for nd in no_decay))],
		'weight_decay': arg.weight_decay},
		{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
									if ('encoder' not in n) and
									(any(nd in n for nd in no_decay))],
		'weight_decay': 0.0},
		{'lr': arg.ptm_lr, 'params': [p for n, p in model.named_parameters()
									if ('encoder' in n) and
									(not any(nd in n for nd in no_decay)) ], 
		'weight_decay': arg.weight_decay},
		{'lr': arg.ptm_lr, 'params': [p for n, p in model.named_parameters()
									if ('encoder' in n) and
									(any(nd in n for nd in no_decay))],
		'weight_decay': 0.0},
	]

	optimizer = AdamW(param_group)

	hyperparams = {
		'batch_size': arg.batch_size,
		'epoch': arg.epoch,
		'identifier': identifier,
		'load_path': arg.load_path,
		'evaluate_every': 1, 
		'update_every': 1,
		'load_epoch': arg.load_epoch,
		'ptm': arg.ptm,
		'do_train': arg.do_train,
		'do_eval': arg.do_eval,
		'wandb': arg.wandb,
		'con_type': arg.con_type,
		'con_loss': arg.con_loss,
		'project': arg.project,
		'method': arg.method,
		'threshold': arg.threshold,
		'task': arg.task,
		'save_path': arg.save_path,
		'save_evidence': arg.save_evidence
	}


	trainer = Trainer(data_loaders, model, optimizer, device, hyperparams)

	trainer.run()