import dgl
import random
import torch

def subgraph_node2vec_random_walk(num_nodes, graph: dgl.DGLGraph, p=1, q=1):
    out = []
    seen = dgl.sampling.node2vec_random_walk(
        graph, random.randrange(graph.num_nodes()), p, q, walk_length=num_nodes * 2).squeeze()
    i = 0
    while len(out) < num_nodes:
        if seen[i] not in out:
            out.append(seen[i].item())
        
        i += 1

        if i == len(seen):
            seen = dgl.sampling.node2vec_random_walk(
                graph, random.randrange(graph.num_nodes()), p, q, walk_length=num_nodes * 2).squeeze()
            i = 0

    out = torch.tensor(out)
    return dgl.DGLGraph.subgraph(graph, out)

def subgraph_random(num_nodes, graph: dgl.DGLGraph):
    return dgl.DGLGraph.subgraph(graph, random.sample(range(graph.num_nodes()), num_nodes))

def subgraph_cluster(num_nodes, graph: dgl.DGLGraph, num_roots=1, max_dist=10):
    startNode = random.randint(0, graph.num_nodes())
    out = []
    stack = [startNode]
    while len(out) < num_nodes:
        curNode = stack.pop()
        if curNode not in out:
            out.append(curNode)
        children = graph.out_edges(curNode)[1].tolist()
        stack = children + stack

    out = torch.tensor(out)
    return dgl.DGLGraph.subgraph(graph, out)