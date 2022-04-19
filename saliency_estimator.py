import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision
import torchvision.transforms as transforms
import numpy as np

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_params(model):
	params = []
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		params.append(param)
	return params

def model_grads(model):
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		grads.append(0. if param.grad is None else param.grad + 0.)
	return grads

def quick_data(dataloader, n_classes, n_samples):
	datas = [[] for _ in range(n_classes)]
	labels = [[] for _ in range(n_classes)]
	mark = dict()
	dataloader_iter = iter(dataloader)
	while True:
		inputs, targets = next(dataloader_iter)
		for idx in range(inputs.shape[0]):
			x, y = inputs[idx:idx+1], targets[idx:idx+1]
			category = y.item()
			if len(datas[category]) == n_samples:
				mark[category] = True
				continue
			datas[category].append(x)
			labels[category].append(y)
		if len(mark) == n_classes:
			break

	X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
	return X, y

### Gradient ###
def cal_grad_loss(net, trainloader, n_samples=10, n_classes=5):
	net.eval()
	d_in, d_out = quick_data(trainloader, n_classes, n_samples)
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for num_class in range(n_classes):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets.to(device))
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.pow(2).detach().clone() for gbase1, g1 in zip(gbase, gradsH) ]

	gbase = [gbase1 / n_classes for gbase1 in gbase]

	return gbase

def cal_grad_feature(net, trainloader, n_samples=10, n_classes=5):
	net.eval()
	d_in, d_out = quick_data(trainloader, n_classes, n_samples)
	base_params = model_params(net.module.features)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for num_class in range(n_classes):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net.module.features(inputs.to(device))
		loss = outputs.norm().pow(2)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.pow(2).detach().clone() for gbase1, g1 in zip(gbase, gradsH) ]

	gbase = [gbase1 / n_classes for gbase1 in gbase]

	return gbase

def cal_grad_logits(net, trainloader, n_samples=10, n_classes=5):
	net.eval()
	d_in, d_out = quick_data(trainloader, n_classes, n_samples)
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for num_class in range(n_classes):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs.to(device))
		loss = outputs.norm().pow(2)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.detach().clone() for gbase1, g1 in zip(gbase, gradsH) ]

	gbase = [gbase1 / n_classes for gbase1 in gbase]

	return gbase

### Loss gradient based importance ###
def cal_saliency_grad_loss(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_loss(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	list_imp = [(gvec[ind]).detach().clone() for ind in range(len(gvec)-2)]
	return list_imp

### Logits gradient based importance ###
def cal_saliency_grad_logits(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_logits(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	list_imp = [(gvec[ind].pow(2)).detach().clone() for ind in range(len(gvec)-2)]
	return list_imp

### Feature gradient based importance ###
def cal_saliency_grad_features(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_feature(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	list_imp = [(gvec[ind]).detach().clone() for ind in range(len(gvec))]
	return list_imp

### EWC for Loss importance ###
def cal_saliency_ewc(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_loss(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	l_params = model_params(net)
	list_imp = [(gvec[ind]).detach().clone() for ind in range(len(l_params)-2)]
	return list_imp

### EWC for Logits importance ###
def cal_saliency_ewc_logits(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_logits(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	l_params = model_params(net)
	list_imp = [(gvec[ind].pow(2)).detach().clone() for ind in range(len(gvec))]
	return list_imp

### EWC for features importance ###
def cal_saliency_ewc_features(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_feature(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	l_params = model_params(net)
	list_imp = [(gvec[ind]).detach().clone() for ind in range(len(gvec))]
	return list_imp

### MAS importance ###
def cal_saliency_mas(net, trainloader, n_samples, n_classes=5):
	gvec = cal_grad_logits(net, trainloader, n_samples=n_samples, n_classes=n_classes)
	list_imp = [(gvec[ind]).abs().detach().clone() for ind in range(len(gvec)-2)]
	return list_imp

### L2-norm based importance ###
def cal_saliency_l2(net):
	l_params = model_params(net)
	list_imp = [l_params[ind].abs().detach().clone() for ind in range(len(l_params)-2)]
	return list_imp

### Vanilla importance ###
def cal_saliency_vanilla(net):
	l_params = model_params(net)
	list_imp = [(1/2) * torch.ones_like(l_params[ind]) for ind in range(len(l_params)-2)]
	return list_imp

### Random importance ###
def cal_saliency_rand(net):
	l_params = model_params(net)
	list_imp = [torch.rand_like(l_params[ind]) for ind in range(len(l_params)-2)]
	return list_imp

### General importance estimation function ###
def cal_saliency(net, saliency_measure, dataloader, n_classes=5):
	n_samples = len(dataloader)//n_classes
	if(saliency_measure=='grad_loss'):
		nlist = cal_saliency_grad_loss(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='ewc'):
		nlist = cal_saliency_ewc(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='l2'):
		nlist = cal_saliency_l2(net)
	elif(saliency_measure=='mas'):
		nlist = cal_saliency_mas(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='grad_logits'):
		nlist = cal_saliency_grad_logits(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='ewc_logits'):
		nlist = cal_saliency_ewc_logits(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='grad_features'):
		nlist = cal_saliency_grad_features(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='ewc_features'):
		nlist = cal_saliency_ewc_features(net, dataloader, n_samples=n_samples, n_classes=n_classes)
	elif(saliency_measure=='vanilla'):
		nlist = cal_saliency_vanilla(net)
	elif(saliency_measure=='rand'):
		nlist = cal_saliency_rand(net)

	return nlist