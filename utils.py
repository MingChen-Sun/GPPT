import argparse
import random
import torch
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from ogb.nodeproppred  import DglNodePropPredDataset
import numpy as np
import os
from sklearn.metrics import accuracy_score
from get_args import get_my_args

def node_mask(train_mask,mask_rate):
    mask_rate=int(mask_rate*10)
    count=0
    for i in range(train_mask.shape[0]):
        if train_mask[i]==True:
            count=count+1
            if count<=mask_rate:
                train_mask[i]=False
                count=count+1
            if count==10:
                count=0
    return train_mask
def my_load_data(args):
    if args.dataset=='cora' or args.dataset=='citeseer' or args.dataset=='pubmed' or args.dataset=='reddit':
        data = load_data(args)
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = data.graph.number_of_edges()
    elif args.dataset=='Fraud_yelp' or args.dataset=='Fraud_amazon':
        if args.dataset=='Fraud_yelp':
            data = dgl.data.FraudDataset('yelp')
        else:
            data = dgl.data.FraudDataset('amazon')
        g = data[0]
        g=dgl.to_homogeneous(g,ndata=['feature','label','train_mask','val_mask','test_mask'])
        features = g.ndata['feature'].to(torch.float32)
        labels = g.ndata['label'].view(1,-1)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = data.graph.number_of_edges()
    elif args.dataset=='CoraFull':
        data = dgl.data.CoraFullDataset()
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        ind=torch.Tensor(random.choices([0,1,2],weights=[0.3,0.1,0.6],k=features.shape[0]))
        g.ndata['train_mask']= (ind==0)
        g.ndata['val_mask']= (ind==1)
        g.ndata['test_mask']= (ind==2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset=='AmazonCoBuyComputer' or args.dataset=='AmazonCoBuyPhoto' or args.dataset=='CoauthorCS' :
        if args.dataset=='AmazonCoBuyComputer':
            data = dgl.data.AmazonCoBuyComputerDataset()
        elif args.dataset=='AmazonCoBuyPhoto':
            data = dgl.data.AmazonCoBuyPhotoDataset()
        elif args.dataset=='CoauthorCS':
            data = dgl.data.CoauthorCSDataset()
        g = data[0]
        features = g.ndata['feat']  # get node feature
        labels = g.ndata['label']  # get node labels
        ind=torch.Tensor(random.choices([0,1,2],weights=[0.1,0.3,0.6],k=features.shape[0]))
        g.ndata['train_mask']= (ind==0)
        g.ndata['val_mask']= (ind==1)
        g.ndata['test_mask']= (ind==2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset=='ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        
        split_idx = dataset.get_idx_split()
        g, labels = dataset[0]
        g = dgl.add_reverse_edges(g)
        features = g.ndata['feat']
        g.ndata['label']=labels.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['train']]=True
        g.ndata['train_mask']= ind.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['valid']]=True
        g.ndata['val_mask']= ind.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['test']]=True
        g.ndata['test_mask']= ind.view(-1,)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = dataset.num_classes
        n_edges = g.number_of_edges()
        labels=labels.view(-1,)
        pass
    else:
        g=None
        features=None
        labels=None
        train_mask=None
        val_mask=None
        test_mask=None
        in_feats=None
        n_classes=None
        n_edges=None
    return g,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges

def evaluate(model, graph, nid, batch_size, device,sample_list):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    valid_dataloader = dgl.dataloading.NodeDataLoader(graph, nid.int(), sampler,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0,device=device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
    return accuracy
    
def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))
            
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_init_info(args):
    g,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges=my_load_data(args)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))
    
    if args.gpu < 0:
        device='cpu'
    else:
        device='cuda:'+str(args.gpu)
        torch.cuda.set_device(args.gpu)
        
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if args.gpu >= 0:
        g = g.int().to(args.gpu)
    return g,features,labels,in_feats,n_classes,n_edges,train_nid,val_nid,test_nid,device
if __name__ == '__main__':
    args = get_my_args()
    print(args)
    info=my_load_data(args)
    g=info[0]
    labels=torch.sparse.sum(g.adj(),1).to_dense().int().view(-1,)
    print(labels)
    li=list(set(labels.numpy()))
    for i in range(labels.shape[0]):
        labels[i]=li.index(labels[i])
    print(set(labels.numpy()))
    #print(info)