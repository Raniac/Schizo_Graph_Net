import time
import random
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
logging.basicConfig(level=logging.ERROR, format='[%(asctime)s %(levelname)s] %(message)s')
logger = logging.getLogger()
hdlr = logging.FileHandler('logs/train_val.log')
# hdlr = logging.FileHandler('logs/train_val_' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.log')
hdlr.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
logger.addHandler(hdlr)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops

from utils.loader import fromConnMat2Edges
from models import Net_191106

# ==== load connectivity matrix from pickle ====
with open('/workspace/schizo_graph_net/data/bennyray_191107_347_bcn.pkl', 'rb') as pkl_file:
        conn_mats = pickle.load(pkl_file)
logging.info('Data size: {:d}'.format(len(conn_mats)))

train_data_list = []
test_data_list = []
nc_counter = 0
sz_counter = 0
conn_mats_keys = list(conn_mats.keys())
random.shuffle(conn_mats_keys)
for subj in conn_mats_keys:
    if subj[:2] == 'NC':
        if nc_counter < 150:
            train_data_list.append(fromConnMat2Edges(conn_mats[subj], 0))
        else:
            test_data_list.append(fromConnMat2Edges(conn_mats[subj], 0))
        nc_counter += 1
    else:
        if sz_counter < 100:
            train_data_list.append(fromConnMat2Edges(conn_mats[subj], 1))
        else:
            test_data_list.append(fromConnMat2Edges(conn_mats[subj], 1))
        sz_counter += 1

# ==== Create dataset with multiple data
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=True)

if torch.cuda.is_available():
    logging.info('Using GPU')
else:
    logging.info('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_191106().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.t()[0])
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    
    return total_loss / len(train_data_list)

def test():
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y.t()[0]).sum().item()
    
    return correct / len(test_data_list)

for epoch in range(1, 201):
    loss = train()
    test_acc = test()
    logging.info('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_acc))
