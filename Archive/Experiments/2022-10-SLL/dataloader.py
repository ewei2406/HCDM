import dgl
import Dataset
import torch

def to_DGL(d):
  dataset = dgl.from_scipy(d.adj)
  dataset.ndata['feat'] = torch.tensor(d.features.todense())
  def to_bool(idx, size):
    bools = torch.zeros(size)
    bools[idx] = 1
    return bools == 1
  dataset.ndata['train_mask'] = to_bool(d.idx_train, dataset.num_nodes())
  dataset.ndata['test_mask'] = to_bool(d.idx_test, dataset.num_nodes())
  dataset.ndata['val_mask'] = to_bool(d.idx_val, dataset.num_nodes())
  dataset.ndata['label'] = torch.tensor(d.labels)
  return dataset

def load_DGL(dataname: str) -> dgl.data.DGLDataset:
  if dataname == 'cora':
    return dgl.data.CoraGraphDataset()[0]
  elif dataname == 'citeseer':
    return dgl.data.CiteseerGraphDataset()[0]
  elif dataname in ['flickr', 'BlogCatalog', 'Polblogs']:
    return load_cached(dataname)

def load_cached(dataname: str) -> dgl.DGLGraph:
  try:
    d = dgl.load_graphs(f'./dataset/{dataname}.bin')[0][0]
  except:
    print(f'data not found. Attempting to download {dataname}...')
    d = to_DGL(Dataset.Dataset(root='./rawdata/', name=dataname))
    dgl.data.utils.save_graphs(f'./dataset/{dataname}.bin', [d], {"glabel": torch.tensor([0])})

  return d

if __name__ == '__main__':

  d = load_DGL('flickr')
  d = load_DGL('BlogCatalog')
  d = load_DGL('Polblogs')
  print(d)
  # d = load_DGL('flickr')
  # dgl.data.utils.save_graphs('./dataset', [d], {"glabel": torch.tensor([0])})
  # print(d)