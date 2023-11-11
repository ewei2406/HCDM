import dgl

def load_DGL(dataname: str) -> dgl.data.DGLDataset:
  if dataname == 'cora':
    return dgl.data.CoraGraphDataset()