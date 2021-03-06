import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import multiprocessing
import os.path
import csv
import copy
import joblib
from torchvision import datasets
import torchvision
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")
from pdb import set_trace as bp

USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

cache = joblib.Memory(location='_cache', verbose=0)


from meta_module import *

global_step = 0

class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        
    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A
            
            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
    


def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1
    
    target = target_cls(training=should_train)
    optimizee = w(target_to_opt())
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)
                    
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            try:
                gradients = detach_var(p.grad.view(cur_sz, 1))
                updates, new_hidden, new_cell = opt_net(
                    gradients,
                    [h[offset:offset+cur_sz] for h in hidden_states],
                    [c[offset:offset+cur_sz] for c in cell_states]
                )
                for i in range(len(new_hidden)):
                    hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                    cell_states2[i][offset:offset+cur_sz] = new_cell[i]
                result_params[name] = p + updates.view(*p.size()) * out_mul
                result_params[name].retain_grad()
            except:
                result_params[name] = p
            
            offset += cur_sz
            
        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
                
            all_losses = None

            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
            
    return all_losses_ever


@cache.cache
def fit_optimizer(target_cls, target_to_opt, unroll=20, optim_it=100, n_epochs=10000, n_tests=100, lr=0.001, out_mul=1.0, test_target=None):
    opt_net = w(OptimizerOneLayer())
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    
    best_net = None
    best_loss = 100000000000000000
    
    for epoch in tqdm(range(n_epochs)):
        print("train")
        #for _ in tqdm(range(20)):
        do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True)

        if epoch % 100 == 0:
            if test_target is not None:
                loss_record = np.mean(np.stack([
                    do_fit(opt_net, meta_opt, target_cls, test_target, unroll, optim_it, n_epochs, out_mul, should_train=False)
                    for _ in tqdm(range(n_tests))
                ]), 0)
            else:
                loss_record = np.mean(np.stack([
                    do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=False)
                    for _ in tqdm(range(n_tests))
                ]), 0)
        loss_record = loss_record.reshape(-1)
        loss = loss_record[-1]
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())
        import pickle
        pickle.dump(loss_record, open("rnnprop_epoch_{}.pkl".format(epoch), 'wb'))
    return best_loss, best_net
  

class CIFAR10Loss:
    def __init__(self, training=True):
        if training:
            dataset = datasets.CIFAR10(
            './data/CIFAR10', train=True, download=True,
            transform=torchvision.transforms.Compose([
                
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        else:
            dataset = datasets.CIFAR10(
            './data/CIFAR10', train=False, download=True,
            transform=torchvision.transforms.Compose([
                
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, num_workers=3, pin_memory=True)

        self.iter_loader = iter(self.loader)
        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        try:
            batch = next(self.iter_loader)
        except:
            self.iter_loader = iter(self.loader)
            batch = next(self.iter_loader)

        return batch

from resnets_meta import resnet20
class CIFAR10ResNet(MetaModule):
    def __init__(self):
        super().__init__()
        self.net = resnet20()
        self.loss = nn.CrossEntropyLoss()
    
    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 3, 32, 32)))
        out = w(Variable(out))
        inp = self.net(inp)
        l = self.loss(inp, out)
        return l


import pickle
MNIST_optimizer = pickle.load(open("dm.pkl", 'rb'))
opt_net = w(Optimizer(preproc=True))
opt_net.load_state_dict(MNIST_optimizer)
meta_opt = optim.Adam(opt_net.parameters(), lr=0.01)
loss_record = np.mean(np.stack([
                    do_fit(opt_net, meta_opt, CIFAR10Loss, CIFAR10ResNet, 20, 10000, 1, 0.01, should_train=False)
                    for _ in tqdm(range(10))
                ]), 0)

pickle.dump(loss_record, open("dm_cifar10_res20_best.pkl", 'wb'))
