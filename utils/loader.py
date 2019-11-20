def fromPickle2Dataset(pkl_path):
    """
    :type pkl_path: String
    :rtype train_dataset: List
    :rtype test_dataset: List
    """
    import pickle
    import logging
    import random

    with open(pkl_path, 'rb') as pkl_file:
        conn_mats = pickle.load(pkl_file)
    logging.info('Data size: {:d}'.format(len(conn_mats)))

    train_dataset = []
    test_dataset = []
    nc_counter = 0
    sz_counter = 0
    conn_mats_keys = list(conn_mats.keys())
    random.shuffle(conn_mats_keys)
    for subj in conn_mats_keys:
        if subj[:2] == 'NC':
            if nc_counter < 150:
                train_dataset.append(fromConnMat2Edges(conn_mats[subj], 0))
            else:
                test_dataset.append(fromConnMat2Edges(conn_mats[subj], 0))
            nc_counter += 1
        else:
            if sz_counter < 100:
                train_dataset.append(fromConnMat2Edges(conn_mats[subj], 1))
            else:
                test_dataset.append(fromConnMat2Edges(conn_mats[subj], 1))
            sz_counter += 1

    return train_dataset, test_dataset

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
        edge_index_tmp[0].extend([idx for i in range(idx, len(itm))])
        edge_index_tmp[1].extend([i for i in range(idx, len(itm))])
        for jdx in range(idx, len(itm)):
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

    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data