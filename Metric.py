import argparse

import numpy as np
import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import digamma, gammaln
import networkx as nx
from torch_sparse import SparseTensor

from data_utils.load import load_data, get_structual_attack
from main import load_features


def _to_numpy(x):
    """
    Ensure tensor or numpy array is converted to numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_cluster_metrics(embeddings, labels):
    embeddings = _to_numpy(embeddings)
    labels = _to_numpy(labels)
    sil = 100*silhouette_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    return {'silhouette': sil, 'davies_bouldin': db}

def structural_homophily(edge_index, labels):
    labels = _to_numpy(labels)
    src, dst = _to_numpy(edge_index)
    same = (labels[src] == labels[dst]).sum()
    return same / len(src)


def embedding_homophily(embeddings, labels, k=5):
    embeddings = _to_numpy(embeddings)
    labels = _to_numpy(labels)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    _, idxs = nbrs.kneighbors(embeddings)
    match = 0
    N = embeddings.shape[0]
    for i in range(N):
        knn = idxs[i, 1:]
        match += (labels[knn] == labels[i]).sum()
    return 100*match / (N * k)


def compute_neighbor_similarity(embeddings, adj_matrix):
    embeddings = _to_numpy(embeddings)
    adj = _to_numpy(adj_matrix)
    sim_matrix = cosine_similarity(embeddings)
    N = embeddings.shape[0]
    sims = []
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        sims.append(np.mean(sim_matrix[i, neighbors]))
    return 100*float(np.mean(sims))


def neighbor_consistency(embeddings, labels, adj_matrix, k=5):
    return {
        'neighbor_similarity': compute_neighbor_similarity(embeddings, adj_matrix),
        'embedding_homophily': embedding_homophily(embeddings, labels, k)
    }


def estimate_mutual_info_knn(X, Y, k=5):
    X = _to_numpy(X)
    Y = _to_numpy(Y).ravel()
    return 100*float(np.mean(mutual_info_regression(X, Y, n_neighbors=k)))


def compute_structural_features(adj_matrix):
    """
    Compute a set of local topological features for each node.
    Features include:
      - degree
      - clustering coefficient
      - PageRank
      - average neighbor degree
    Returns: np.ndarray of shape (N, F)
    """
    import networkx as nx
    A = _to_numpy(adj_matrix)
    G = nx.from_numpy_array(A)
    N = A.shape[0]
    # degree
    deg = np.array([d for _, d in G.degree()])
    # clustering coefficient
    clust = np.array(list(nx.clustering(G).values()))
    # pagerank
    pr = np.array(list(nx.pagerank(G).values()))
    # average neighbor degree
    andeg = np.array(list(nx.average_neighbor_degree(G).values()))
    return np.vstack([deg, clust, pr, andeg]).T

def estimate_structure_mutual_info(embeddings, adj_matrix, k=5):
    """
    Estimate mutual information I(H;S) between node embeddings H and
    local structural feature vectors S.

    embeddings: np.ndarray (N, D)
    adj_matrix: np.ndarray (N, N)
    k: neighbors for MI regression
    Returns average I(H; s_j) over structural features.
    """
    X = _to_numpy(embeddings)
    S = compute_structural_features(adj_matrix)
    F = S.shape[1]
    mi_list = []
    # For each structural feature dimension, estimate MI between that feature and embeddings
    for f in range(F):
        y = S[:, f]
        mi_dims = mutual_info_regression(X, y, n_neighbors=k)
        mi_list.append(np.mean(mi_dims))
    return 100*float(np.mean(mi_list))

def build_adj_list(edge_index, N):
    adj = [[] for _ in range(N)]
    for u, v in edge_index.T:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def edge_to_matrix(edge_index, N):
    mat = np.zeros((N, N), dtype=int)
    for u, v in edge_index.T:
        mat[u, v] = 1
        mat[v, u] = 1
    return mat


def analyze(emb_clean, emb_poison, edge_index_clean, edge_index_poison, labels):
    """
    Compare robustness metrics between clean and poisoned graphs.

    emb_clean, emb_poison: np.ndarray (N, D)
    edge_index_clean, edge_index_poison: np.ndarray (2, E)
    labels: np.ndarray (N,)
    model: optional GNN embedding function
    features: optional torch.Tensor (N, Dfeat)
    """
    N = labels.shape[0]
    if isinstance(edge_index_clean, SparseTensor):
        edge_index_clean=torch.tensor([edge_index_clean.row(),edge_index_clean.col()])
    adj_clean_list = build_adj_list(edge_index_clean, N)
    adj_poison_list = build_adj_list(edge_index_poison, N)
    adj_clean_mat = edge_to_matrix(edge_index_clean, N)
    adj_poison_mat = edge_to_matrix(edge_index_poison, N)


    results = ""
    # Embedding Separability.
    results += f'cluster_clean:{compute_cluster_metrics(emb_clean, labels)}'
    results += f'cluster_poison:{compute_cluster_metrics(emb_poison, labels)}'
    #
    # Hompohily
    results += f'h_emb_clean:{embedding_homophily(emb_clean, labels)}:.2f'
    results += f'h_emb_poison:{embedding_homophily(emb_poison, labels)}:.2f'

    # neighbor consistency
    results += f'neighbor_sim_clean:{compute_neighbor_similarity(emb_clean, adj_clean_mat)}'
    results += f'neighbor_sim_poison:{compute_neighbor_similarity(emb_poison, adj_poison_mat)}'

    # embedding–label mutual information
    results += f'mi_clean_labels:{estimate_mutual_info_knn(emb_clean, labels)}'
    results += f'mi_poison_labels:{estimate_mutual_info_knn(emb_poison, labels)}'
    # embedding-structural mutual information
    results += f'mi_struct_c: {estimate_structure_mutual_info(emb_clean, adj_clean_mat)}'
    results += f'mi_struct_p {estimate_structure_mutual_info(emb_poison, adj_poison_mat)}'
    return results


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_attack',type=str, default=None)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    dataset = args.dataset
    attack = args.attack

    data, _ = load_data(dataset,attack_type=attack)
    edge = data.edge_index
    _, modified_adj = get_structual_attack(dataset,"mettack",0.2,0)
    version_names = [
        "Shallow", "TAPE", "KEA", "LLAMA", "ChatGPT",
        "Linq", "SimTeg", "E5-Large", "ModernBert"
    ]
    version_transforms = [
        "OGB", "E", "knowsep", "LLAMA", "ChatGPT",
        "Linq", "SimTeg", "E5", "ModernBert"
    ]
    for version_name, version_transform in zip(version_names, version_transforms):
        print(f"{version_name}")
        features, labels = load_features(version_transform, data, 0, args)
        Afeatures, Alabels = load_features(version_transform, data, 0, args)

        features,Afeatures = features.cpu(),Afeatures.cpu()

        if attack is None:
            result = analyze(features, features, edge, modified_adj.cpu(), labels)
        else:
            result = analyze(features, Afeatures, edge, edge, labels)

        print(f"{result}")