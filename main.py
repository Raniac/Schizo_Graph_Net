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
from torch.optim import lr_scheduler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops

from utils.loader import *
from models import *

# ==== Create dataset with multiple data
# train_dataset, test_dataset = fromPickle2Dataset('/workspace/schizo_graph_net/data/bennyray_191107_347_bcn.pkl')
# train_dataset, test_dataset = fromPickle2DatasetWithFeature('/workspace/schizo_graph_net/data/bennyray_191107_347_bcn.pkl', '/workspace/schizo_graph_net/data/RANIAC_181210_345_sfMRI_90.csv')
# train_dataset, test_dataset = fromTxt2Dataset('/workspace/schizo_graph_net/data/ByDPABI/')
train_dataset, test_dataset = fromTxt2DatasetWithFeature('/workspace/schizo_graph_net/data/test_dpabi/', '/workspace/schizo_graph_net/data/RANIAC_181210_345_sfMRI_90.csv')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if torch.cuda.is_available():
    logging.info('Using GPU')
else:
    logging.info('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_191225().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
# TODO add learning-rate scheduler.
scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

def train(data_loader, data_size):
    model.train()

    total_loss = 0
    correct = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    train_loss = total_loss / data_size
    train_acc = correct / data_size
    
    return train_loss, train_acc

def test(data_loader, data_size):
    model.eval()

    total_loss = 0
    correct = 0
    predicted_y = []
    original_y = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = F.nll_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
        predicted_y.extend(out.max(dim=1)[1])
        original_y.extend(data.y)
        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    test_loss = total_loss / data_size
    test_acc = correct / data_size
    test_out = (predicted_y, original_y)
    
    return test_loss, test_acc, test_out

for epoch in range(1, 241):
    scheduler.step()
    train_loss, train_acc = train(train_loader, len(train_dataset))
    test_loss, test_acc, _ = test(test_loader, len(test_dataset))
    logging.info('Epoch {:03d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, train_loss, train_acc, test_loss, test_acc))

test_loss, test_acc, test_out = test(test_loader, len(test_dataset))
for idx in range(len(test_out[0])):
    test_out[0][idx] = test_out[0][idx].item()
    test_out[1][idx] = test_out[1][idx].item()
print(test_out[0])
print(test_out[1])
