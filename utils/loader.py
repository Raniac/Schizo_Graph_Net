def fromConnMat2Edges(conn_mat, label):
    """
    :type conn_mat: List
    :type label: int
    :rtype: Torch Data Object
    """
    import torch
    from torch_geometric.data import Data

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