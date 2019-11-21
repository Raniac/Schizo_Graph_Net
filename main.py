import time
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

from utils.loader import fromPickle2Dataset
from models import *

# ==== Create dataset with multiple data
train_dataset, test_dataset = fromPickle2Dataset('/workspace/schizo_graph_net/data/bennyray_191107_347_bcn.pkl')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if torch.cuda.is_available():
    logging.info('Using GPU')
else:
    logging.info('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_191120().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()

    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    train_loss = total_loss / len(train_dataset)
    train_acc = correct / len(train_dataset)
    
    return train_loss, train_acc

def test():
    model.eval()

    total_loss = 0
    correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = F.nll_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    test_loss = total_loss / len(test_dataset)
    test_acc = correct / len(test_dataset)
    
    return test_loss, test_acc

for epoch in range(1, 101):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    logging.info('Epoch {:03d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
