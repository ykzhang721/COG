from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch


class MyDataset_classifi(Dataset):
	def __init__(self, file_list, feature_extractor):
		self.file_list = file_list
	
		self.transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
			]
		)
		self.feature_extractor = feature_extractor


	def __len__(self):
		self.filelength = len(self.file_list)
		return self.filelength

	def __getitem__(self, idx):
		'''
		img_path = self.file_list[idx]
		img = Image.open(img_path).convert("RGB")
		img_transformed = self.transform(img)
		label = lab2id[img_path.split("/")[2]]
		path = img_path.split('/')[-1]
		return img_transformed, label, path
		'''
		ent, images, label = self.file_list[idx] 

		img = Image.open(images).convert("RGB")
		img_transformed = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'][0]
		return ent, img_transformed, label 


class MyDataset_ranking(Dataset):
	def __init__(self, file_list, feature_extractor):
		self.file_list = file_list
	
		self.transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
			]
		)
		self.feature_extractor = feature_extractor


	def __len__(self):
		self.filelength = len(self.file_list)
		return self.filelength

	def __getitem__(self, idx):
		'''
		img_path = self.file_list[idx]
		img = Image.open(img_path).convert("RGB")
		img_transformed = self.transform(img)
		label = lab2id[img_path.split("/")[2]]
		path = img_path.split('/')[-1]
		return img_transformed, label, path
		'''
		ent, images, label = self.file_list[idx] 

		if isinstance(images, list):
			i = []
			for image in images:
				img = Image.open(image).convert("RGB")
				img_transformed = self.feature_extractor(images=img, return_tensors="pt")['pixel_values']
				i.append(img_transformed)
			return ent, torch.cat(i,dim=0), torch.tensor(label)
		else:
			img = Image.open(images).convert("RGB")
			img_transformed = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'][0]
			return ent, img_transformed, label 
