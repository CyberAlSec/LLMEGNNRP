import torch
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.constants import troy_pound
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix
import random
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
from data_utils.load import load_data


def sample_subgraph(G: nx.Graph, target_size: int, seed_size: int = 100, hop: int = 2) -> list:
    """
    Sample a set of approximately target_size nodes using GraphSAGE-style neighborhood sampling.

    Returns:
    - sampled: list of original node IDs
    """
    nodes = list(G.nodes())
    seeds = set(random.sample(nodes, min(seed_size, len(nodes))))
    frontier = set(seeds)
    sampled = set(seeds)

    for _ in range(hop):
        if len(sampled) >= target_size:
            break
        next_frontier = set()
        # determine per-node sample count to approximate growth
        k = max(1, int((target_size - len(sampled)) / max(len(frontier), 1)))
        for u in frontier:
            neighbors = list(G.neighbors(u))
            if not neighbors:
                continue
            sampled_neighbors = set(random.sample(neighbors, min(len(neighbors), k)))
            next_frontier.update(sampled_neighbors)
        sampled.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    # if still under target, add random nodes
    if len(sampled) < target_size:
        remaining = set(nodes) - sampled
        additional = set(random.sample(remaining, min(len(remaining), target_size - len(sampled))))
        sampled.update(additional)

    return list(sampled)


def generate_subgraph(data_name, target_size):

    data,num_classes,text = load_data(data_name, use_text=True, use_gpt=False, seed=0)

    G = to_networkx(data, to_undirected=True)

    # Sample subgraph nodes (~12000)
    node_idx_list = sample_subgraph(G, target_size=target_size, seed_size=100, hop=2)
    node_idx = torch.tensor(node_idx_list, dtype=torch.long)

    # 4) Extract induced subgraph edge_index and relabel nodes
    sub_edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)

    # 5) Optionally remove duplicate edges (subgraph utility already ensures no duplicates)
    unique_edges = sub_edge_index

    # 6) Build PyG Data object for the subgraph
    sub_data = Data(
        x=data.x[node_idx],
        edge_index=unique_edges,
        y=data.y[node_idx],
        node_idx=node_idx,
        text = [text[i] for i in node_idx_list]
    )
    torch.save(sub_data,f"./data/{data_name}-subgraph.pt")



if __name__ == "__main__":
    generate_subgraph("arxiv_2023",13167)

