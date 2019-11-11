import torch
torch.manual_seed(5)

from torch_geometric.datasets import GeometricShapes
from torch_geometric.data import DataLoader

dataset = GeometricShapes(root='/tmp/geometric_shapes')

# Filter dataset to only contain a circle and a square.
dataset = dataset[torch.tensor([0, 4])]

loader = DataLoader(dataset, batch_size=2, shuffle=False)

data = next(iter(loader))  # Get first mini-batch.

import torch_geometric.transforms as T

dataset.transform = T.SamplePoints(num=128)
data = next(iter(loader))  # Get first mini-batch.

from torch_geometric.nn import fps

mask = fps(data.pos, data.batch, ratio=0.25)

# Create radius graph.
from torch_geometric.nn import radius

assign_index = radius(data.pos, data.pos[mask], 0.4,
                      data.batch, data.batch[mask])

import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class PointNetLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        # Message passing with max aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialize a MLP.
        self.mlp = Sequential(Linear(in_dim, out_dim),
                              ReLU(),
                              Linear(out_dim, out_dim))
        
    def forward(self, pos, pos_sampled, assign_index):
        # Start propagating messages.
        return self.propagate(assign_index, pos=(pos, pos_sampled))
    
    def message(self, pos_j, pos_i):
        # Generate messages.
        return self.mlp(pos_j - pos_i)  

import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = PointNetLayer(3, 32)
        self.classifier = Linear(32, dataset.num_classes)
        
    def forward(self, pos, batch):
        # 1. Sample farthest points.
        mask = fps(pos, batch, ratio=0.25)
        
        # 2. Dynamically generate message passing connections.
        row, col = radius(pos, pos[mask], 0.3, batch, batch[mask])
        assign_index = torch.stack([col, row], dim=0)  # Transpose.
        
        # 3. Start bipartite message passing.
        x = self.conv(pos, pos[mask], assign_index)
       
        # 4. Global Pooling.
        x = global_max_pool(x, batch[mask])
        
        # 5. Classifier.
        return self.classifier(x)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        # data = data.to(torch.device('cuda'))
        out = model(data.pos, data.batch)
        loss = F.nll_loss(F.log_softmax(out, dim=1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

for epoch in range(1, 10):
    loss = train()
    print('Epoch: {:01d}, Loss: {:.4f}'.format(epoch, loss))

