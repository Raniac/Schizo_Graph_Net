# ==== from connectivity matrix to edges ====
def fromConnMat2Edges(conn_mat, label):
    edge_index_tmp = [[], []]
    edge_attr_tmp = []

    for idx, itm in enumerate(conn_mat):
        edge_index_tmp[0].extend([idx for i in range(idx+1, len(itm))])
        edge_index_tmp[1].extend([i for i in range(idx+1, len(itm))])
        for jdx in range(idx+1, len(itm)):
            edge_attr_tmp.append([itm[jdx]])

    edge_index = torch.tensor(edge_index_tmp, dtype=torch.long)
    # where 0, 1 are the node indeces
    # the shape of edge_index is [2, num_edges]

    edge_attr = torch.tensor(edge_attr_tmp, dtype=torch.float)
    # where the list items are the edge feature vectors
    # the shape of edge_attr is [num_edges, num_edge_features]

    x = torch.tensor([[1] for i in range(0, 90)], dtype=torch.float)
    # where the list items are the node feature vectors
    # the shape of x is [num_nodes, num_node_features]

    y = torch.tensor([[label]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

# ==== load connectivity matrix from pickle ====
import pickle

with open('/workspace/test_pyg/bennyray_191007_347_bcn.pkl', 'rb') as pkl_file:
        conn_mats = pickle.load(pkl_file)
print(len(conn_mats))

import torch
from torch_geometric.data import Data

data_list = []

for subj in conn_mats.keys():
    if subj[:2] == 'NC':
        data_list.append(fromConnMat2Edges(conn_mats[subj], 0))
    else:
        data_list.append(fromConnMat2Edges(conn_mats[subj], 1))

# ==== Create dataset with multiple data
from torch_geometric.data import DataLoader

loader = DataLoader(data_list, batch_size=32, shuffle=True)

for batch in loader:
    pass # do operations on batches

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

if torch.cuda.is_available():
    print('Using GPU')
else:
    print('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

corrects = 0
c = 0
for batch in loader:
    print('Computing batch ' + str(c))
    data = batch.to(device)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.long().squeeze())
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred.eq(data.y.t()[0].long()).sum().item())
    corrects += correct
    
    c += 1

acc = corrects / len(conn_mats)
print('Accuracy: {:.4f}'.format(acc))
