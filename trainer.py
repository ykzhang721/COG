import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
import numpy as np
import wandb 
import pickle
import json

class Trainer:
	def __init__(self, data_loaders, model, optimizer, device, hyperparams, feature_extractor=None):
		self.data_loaders = data_loaders
		self.model = model
		self.feature_extractor = feature_extractor
		self.optimizer = optimizer
		self.device = device
		self.identifier = hyperparams['identifier']
		self.method = hyperparams['method']
		self.hyperparams = hyperparams
		self.save_folder = hyperparams['save_path']
		self.load_epoch = hyperparams['load_epoch']
		self.wandb = hyperparams['wandb']
		self.task = hyperparams['task']
		self.save_evidence = hyperparams['save_evidence']
		model.to(device)


		if not os.path.exists(self.save_folder):
			os.makedirs(self.save_folder)

		self.param_path_template = self.save_folder + self.identifier + '_epc={0}_metric={1}'  + '.pt'
		self.best_metric = {'MRR': 0, 'f1': 0 }
		self.best_epoch = {'MRR': -1, 'f1': -1 }

 
		if not os.path.exists(self.save_folder):
			os.makedirs(self.save_folder)

		load_path = hyperparams['load_path']

		if load_path != None:
			if os.path.exists(load_path):
				model.load_state_dict(torch.load(load_path), strict=False)
				print('Parameters loaded from {0}.'.format(load_path))
			else:
				print('Parameters {0} Not Found'.format(load_path))

	def run(self):
		if self.hyperparams['do_train']:
			self.train()
		if self.hyperparams['do_eval']:
			if self.task == 'classification':
				self.test_tc(epc=self.hyperparams['load_epoch'], split="test")
			elif self.task == 'ranking':
				self.test_lp(epc=self.hyperparams['load_epoch'], split="test")

	def train(self):
		model = self.model
		optimizer = self.optimizer
		device = self.device
		hyperparams = self.hyperparams
		epoch = hyperparams['epoch'] 
		train_loader = self.data_loaders['train']
		
		
		con2id = self.data_loaders['con2id']
		hypers = self.data_loaders['hypers']
		sigmoid = torch.nn.Sigmoid()
		bce_criterion = torch.nn.BCELoss(reduction='none')


		if hyperparams['wandb']:
			wandb.init(
				project=hyperparams['project'],
				name=self.identifier,
				config=hyperparams
			)
		model.train()


		for epc in range(self.load_epoch + 1, epoch):
			total_loss = 0 
			dataset_size = len(train_loader.dataset)
			
			n_batch = len(train_loader)

			for i_b, (ents, imgs, labels) in tqdm(enumerate(train_loader), total=n_batch):
				n_sample = len(ents)
				labels = torch.eye(n_sample).to(device)
				imgs = imgs.to(device)
				cons = get_cons_matrix(ents, con2id, hypers, device)
				e_preds = model.train_forward(imgs, None, ents)
				loss = 0 
				for i_e, preds in enumerate(e_preds):
					pos_idx = i_e 
					neg_idx = [ _ for _ in range(n_sample) if _ != i_e] 
					preds = preds.squeeze(1)
					l = bce_criterion(preds, labels[i_e])
					pos_loss = l[pos_idx]
					neg_loss = l[neg_idx].mean() 
					loss += pos_loss + neg_loss
				
				if hyperparams['con_loss']:
					con_preds = model.train_forward(imgs, cons, None)
					for i_e, preds in enumerate(con_preds): 
						preds = sigmoid(preds)
						labels = torch.zeros(preds.size())
						i_con = cons[i_e] 
						for j_e, j_con in enumerate(cons):
							labels[j_e] = torch.tensor([int((c in j_con)) for c in i_con])
						labels = labels.to(device)

						preds = preds.reshape(-1)
						labels = labels.reshape(-1)

						l = bce_criterion(preds, labels)
						pos_idx = labels.nonzero().reshape(-1)
						neg_idx = (1-labels).nonzero().reshape(-1)

						
						if min(pos_idx.shape) != 0:
							pos_loss = l[pos_idx].mean()
							loss += pos_loss

						if min(neg_idx.shape) != 0:
							neg_loss = l[neg_idx].mean()
							loss += neg_loss

				total_loss += loss.item()
				loss /= n_sample

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			avg_loss = total_loss / dataset_size 
			print('Train: Avg Loss {}'.format(avg_loss))

			if hyperparams['wandb']:
				wandb.log({"train_loss": avg_loss})

			if self.task == 'classification':
				self.test_tc(epc=self.hyperparams['load_epoch'], split="valid")
			elif self.task == 'ranking':
				self.test_lp(epc=self.hyperparams['load_epoch'], split="valid")

		if hyperparams['wandb']:wandb.finish()

	def test_tc(self, epc=-1, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams
		data_loader = self.data_loaders[split]
		con2id = self.data_loaders['con2id']
		hypers = self.data_loaders['hypers']

			
		model.eval()
		with torch.no_grad():
			TP = 0 
			FP = 0 
			FN = 0 
			TN = 0

			n_batch = len(data_loader)
			cfhe = self.calculate_cfhe(data_loader)
			evidences = {}
			for i_b, (ents, imgs, labels) in tqdm(enumerate(data_loader), total=n_batch):
				imgs = imgs.to(device)
				labels = labels.to(device)
				cons = get_cons_matrix(ents, con2id, hypers, device)
				
				if self.method == "CI" or self.method == "onlyent":
					pred_labels = model.eval_forward_tc(imgs, None, ents, None, self.hyperparams) # Concept Integration or Use only entity name
				elif self.method == "CI_EF":
					if self.save_evidence and split == 'test':
						pred_labels, evidence = model.eval_forward_tc(imgs, cons, ents, cfhe, self.hyperparams,True) # Concept Integration + Evidence Fusion; Save evidence
						evidences.update(evidence)
					else:
						pred_labels = model.eval_forward_tc(imgs, cons, ents, cfhe, self.hyperparams) # Concept Integration + Evidence Fusion
				pos_idx = [index for index, _ in enumerate(labels) if _ == 1]
				neg_idx = [index for index, _ in enumerate(labels) if _ == 0]
				
				TP += (pred_labels == labels).float()[pos_idx].sum().item()
				FN += (pred_labels != labels).float()[pos_idx].sum().item()
				FP += (pred_labels != labels).float()[neg_idx].sum().item()
				TN += (pred_labels == labels).float()[neg_idx].sum().item()

			precision = TP / max(TP+FP, 1)
			recall = TP / max(TP+FN, 1)
			f1 = 2 * (precision * recall) / max(precision+recall, 1)
			acc = (TP + TN) / (TP + FN + FP + TN)
			
			
			if evidences:
				with open(f'{self.save_folder}/{self.identifier}_evidence.json','w') as f:
					json.dump(evidences, f, ensure_ascii=False, indent=4)

			if split == 'valid':
				print('Valid: precision {:.4f} recall {:.4f} f1 {:.4f} Accuracy:{:.4f}'.format(precision, recall, f1, acc))
				if hyperparams['wandb']:
					wandb.log({"valid_precision": precision,
					"valid_recall":recall, "valid_accracy": acc , "valid_f1":f1})

			if split == 'test':
				print('Test: precision {:.4f} recall {:.4f} f1 {:.4f} Accuracy:{:.4f}'.format(precision, recall, f1, acc))

		if split == 'valid':
			self.save_model(epc, 'f1', f1)
			model.train() 


	def test_lp(self, epc=-1, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams

		data_loader = self.data_loaders[split]
		con2id = self.data_loaders['con2id']
		hypers = self.data_loaders['hypers']
	
		model.eval()
		with torch.no_grad():
			MR = 0
			MRR = 0
			hits1 = 0
			hits3 = 0
			hits10 = 0

			dataset_size = len(data_loader.dataset)
			n_batch = len(data_loader)
			cfhe = self.calculate_cfhe(data_loader)
			

			for i_b, (ents, imgs, labels) in tqdm(enumerate(data_loader), total=n_batch):
				imgs = imgs.to(device)
				labels = labels.to(device)
				cons = get_cons_matrix(ents, con2id, hypers, device)
				preds = model.eval_forward_lp(imgs, ents) # Concept Integration or Use only entity name
				for i, pred in enumerate(preds):
					sorted_tensor, sorted_indices = torch.sort(pred, descending=True)
					position = torch.where(sorted_indices == labels[i])[0].item() + 1
					MR += position
					MRR += 1 / position
					hits1 += self.cal_hits(position,1)
					hits3 += self.cal_hits(position,3)
					hits10 += self.cal_hits(position,10)

			MR /= dataset_size
			MRR /= dataset_size
			hits1 /= dataset_size
			hits3 /= dataset_size
			hits10 /= dataset_size
			
			if split == 'valid':
				print('Valid: MR {:.4f} MRR {:.4f} H@1 {:.4f} H@3 {:.4f} H@10 {:.4f}'.format(MR, MRR, hits1, hits3, hits10))
				if hyperparams['wandb']:
					wandb.log({"MR": MR,
					"MRR":MRR, "H@1": hits1 , "H@3": hits3, "H@10": hits10})

			if split == 'test':
				print('Test: MR {:.4f} MRR {:.4f} H@1 {:.4f} H@3 {:.4f} H@10 {:.4f}'.format(MR, MRR, hits1, hits3, hits10))
				
		if split == 'valid':
			self.save_model(epc, 'MRR', MRR)
			model.train()


	def cal_hits(self, position, num):
		return int(position <= num)



	def update_metric(self, epc, name, score):
		if (score > self.best_metric[name]) :
			self.best_metric[name] = score
			self.best_epoch[name] = epc
			print('! Metric {0} Updated as: {1:.2f}'.format(name, score*100))
			return True
		else:
			return False

	def save_model(self, epc, metric, metric_val):
		save_path = self.param_path_template.format(epc, metric)
		last_path = self.param_path_template.format(self.best_epoch[metric], metric)

		if self.update_metric(epc, metric, metric_val):
			if os.path.exists(last_path) and save_path != last_path and epc >= self.best_epoch[metric]:
				os.remove(last_path)
				print('Last parameters {} deleted'.format(last_path))
			
			torch.save(self.model.state_dict(), save_path)
			print('Parameters saved into ', save_path)


	def calculate_cfhe(self, data):
		cfhe = {}
		hypers = self.data_loaders['hypers']
		entity = set()
		hypos_cfhe = {}
		for (entities, imgs, labels) in data:
			for e in entities:
				entity.add(e)
				cons = hypers[e]
				for con in cons:
					hypos_cfhe.setdefault(con, set())
					hypos_cfhe[con].add(e)

		for e in entity:
			cfhe[e] = {}
			cons = hypers[e]
			for con in cons:
				if len(hypos_cfhe[con]) >= 10:
					cfhe[e][con] = ( (1 / math.log(len(hypos_cfhe[con]), 10)) - (1 / len(entity))) / (1 - (1 / len(entity))) 
				else:
					cfhe[e][con] = ( 1 - (1 / len(entity))) / (1 - (1 / len(entity))) 
		return cfhe



def get_cons_matrix(entities, con2id, hypers, device):
	cons = []
	for i_e, entity in enumerate(entities):
		ent_cons = torch.LongTensor([con2id[con] for con in hypers[entity]]).to(device)
		cons.append(ent_cons)
	return cons 