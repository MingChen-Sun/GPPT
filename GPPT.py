import argparse
import get_args
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import dgl.nn.pytorch as dglnn
from model import SAGE
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import sklearn.linear_model as lm
import sklearn.metrics as skm
import utils


import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,center_num):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes=n_classes
        self.center_num=center_num
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))

        self.prompt=nn.Linear(n_hidden,self.center_num,bias=False)
        
        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2*n_hidden,n_classes,bias=False))
        
        
    def model_to_array(self,args):
        s_dict = torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in np.arange(1, len(keys), 1):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res
    def array_to_model(self, args):
        arr=self.model_to_array(args)
        m_m=torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')#+str(args.gpu))
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():
            length = torch.prod(torch.tensor(param.shape))
            s_dict[name] = arr[indice:indice + length].view(param.shape)
            indice = indice + length
        self.load_state_dict(s_dict)
    
    def load_parameters(self, args):
        self.args=args
        self.array_to_model(args)
    def weigth_init(self,graph,inputs,label,index):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)
        graph.ndata['h']=h
        graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neighbor'))
        neighbor=graph.ndata['neighbor']
        h=torch.cat((h,neighbor),dim=1)
        
        features=h[index]
        labels=label[index.long()]
        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(features.detach().cpu())
        
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data.copy(temp)
        

        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0)
        for i in range(self.center_num):
            self.pp[i].weight.data.copy(temp)
        
    
    def update_prompt_weight(self,h):
        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(h.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data.copy(temp)

    def get_mul_prompt(self):
        pros=[]
        for name,param in self.named_parameters():
            if name.startswith('pp.'):
                pros.append(param)
        return pros
        
    def get_prompt(self):
        for name,param in self.named_parameters():
            if name.startswith('prompt.weight'):
                pro=param
        return pro
    
    def get_mid_h(self):
        return self.fea

    def forward(self, graph, inputs):
        if self.dropout==False:
            h=inputs
        else:
            h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h_dst = h[:graph[l].num_dst_nodes()]  # <---
            h = layer(graph[l], (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout!=False:
                    h = self.dropout(h)
        h = self.activation(h)
        h_dst = self.activation(h_dst)
        neighbor=h_dst
        h=torch.cat((h,neighbor),dim=1)
        self.fea=h 

        out=self.prompt(h)
        index=torch.argmax(out, dim=1)
        out=torch.FloatTensor(h.shape[0],self.n_classes).cuda()
        for i in range(self.center_num):
            out[index==i]=self.pp[i](h[index==i])
        return out
    
def main(args):
    utils.seed_torch(args.seed)
    g,features,labels,in_feats,n_classes,n_edges,train_nid,val_nid,test_nid,device=utils.get_init_info(args)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.NodeDataLoader(g,train_nid.int(),sampler,device=device,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=0)
    model = GraphSAGE(in_feats,args.n_hidden,n_classes,args.n_layers,F.relu,args.dropout,args.aggregator_type,args.center_num)
    model.to(device)
    model.load_parameters(args)
    model.weigth_init(g,features,labels,train_nid)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_all=[]
    loss_all=[]
    for epoch in range(args.n_epochs):
        model.train()
        acc = utils.evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
        acc_all.append(acc)
        t0 = time.time()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            logits = model(mfgs, inputs)
            loss = F.cross_entropy(logits, lab)
            
            loss_all.append(loss.cpu().data)
            loss=loss+args.lr_c*utils.constraint(device,model.get_mul_prompt())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embedding_save=model.get_mid_h().detach().clone().cpu().numpy()
            data=pd.DataFrame(embedding_save)
            label=pd.DataFrame(lab.detach().clone().cpu().numpy())
            data.to_csv("./data.csv",index=None,header=None)
            label.to_csv("./label.csv",index=None,header=None)
            pd.DataFrame(torch.cat(model.get_mul_prompt(),axis=1).detach().clone().cpu().numpy()).to_csv("./data_p.csv",index=None,header=None)
            model.update_prompt_weight(model.get_mid_h())
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0, loss.item(),acc))
    
    pd.DataFrame(acc_all).to_csv('./res/gs_pre_pro_mul_pro_center_c_nei_'+args.dataset+'.csv',index=None,header=None)
    pd.DataFrame(loss_all).to_csv('./res/gs_pre_pro_mul_pro_center_c_nei_'+args.dataset+'_loss.csv',index=None,header=None)


    acc = utils.evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
    
    print("Test Accuracy {:.4f}".format(np.mean(acc_all[-10:])))


if __name__ == '__main__':
    args=get_args.get_my_args()
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
