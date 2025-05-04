import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_wallet_graph_combined(data_dir):

    edge_path = os.path.join(data_dir, 'AddrAddr_edgelist.csv')
    edge_df = pd.read_csv(edge_path, skiprows=1)
    edge_df.columns = ['input_address', 'output_address']

    addr_set = pd.unique(edge_df[['input_address', 'output_address']].values.ravel())
    addr_map = {addr: idx for idx, addr in enumerate(addr_set)}

    src = edge_df['input_address'].map(addr_map)
    dst = edge_df['output_address'].map(addr_map)

    edges_np = np.vstack([src.values, dst.values])
    edge_index = torch.from_numpy(edges_np).long()

    node_path = os.path.join(data_dir, 'wallets_features_classes_combined.csv')
    df = pd.read_csv(node_path)

    df = df[df['address'].isin(addr_map)]
    df['node_id'] = df['address'].map(addr_map)
    df = df.sort_values('node_id')

    x = torch.tensor(df.drop(columns=['address', 'class', 'node_id']).values, dtype=torch.float)

    # encoding class labels: 2 = licit > 0, 1 = illicit > 1, none > -1
    label_map = {2: 0, 1: 1}
    y = df['class'].map(label_map).fillna(-1).astype(int)
    y = torch.tensor(y.values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

