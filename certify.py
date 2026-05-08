import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from torch_geometric.utils import dropout_edge, add_random_edge, coalesce, remove_self_loops
from tqdm import tqdm


def get_perturbed_adj(edge_index, num_nodes, p_plus, p_minus,is_symmetric=True,device="cuda:0"):
    # --- 1. 删边 (严格伯努利) ---
    mask = torch.rand(edge_index.size(1), device=device) > p_minus
    new_edges = edge_index[:, mask]

    # --- 2. 加边 (近似伯努利加速版) ---
    if p_plus > 0:
        if is_symmetric:
            total_slots = num_nodes * (num_nodes - 1) // 2
        else:
            total_slots = num_nodes * (num_nodes - 1)

        n_added = np.random.binomial(total_slots, p_plus)

        if n_added > 0:
            added_edges = torch.randint(0, num_nodes, (2, n_added), device=device)
            if is_symmetric:
                reverse_edges = added_edges[[1, 0]]
                added_edges = torch.cat([added_edges, reverse_edges], dim=1)

            new_edges = torch.cat([new_edges, added_edges], dim=1)

    # --- 3. 规范化 ---
    new_edges, _ = remove_self_loops(new_edges)
    new_edges = coalesce(new_edges, num_nodes=num_nodes)
    return new_edges

# ==========================================
# 2. 完整的图结构采样逻辑 (包含加边与删边)
# ==========================================
def sample_perturbed_edges(edge_index, num_nodes, p_plus, p_minus,device=torch.device('cuda:0')):
    """
    严谨的采样逻辑：
    1. 删边：对现有边以 p_minus 概率删除
    2. 加边：对不存在的边以 p_plus 概率增加
    """
    # --- 删边逻辑 ---
    row, col = edge_index
    # 只处理上三角以避免无向图重复采样（如果是无向图）
    mask = torch.rand(row.size(0)) > p_minus
    new_edge_index = edge_index[:, mask]

    # --- 加边逻辑 (高效实现) ---
    if p_plus > 0:
        # 计算潜在边的总数: N*(N-1)/2 - 当前边数
        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_current_edges = edge_index.size(1) // 2
        num_empty_slots = num_possible_edges - num_current_edges

        # 从二项分布中采样：实际要增加多少条边
        n_added = np.random.binomial(num_empty_slots, p_plus)

        if n_added > 0:
            added_edges = []
            while len(added_edges) < n_added:
                u = np.random.randint(0, num_nodes)
                v = np.random.randint(0, num_nodes)
                if u != v:
                    # 这里简化了“检查边是否已存在”的逻辑以提升速度
                    # 在稀疏图中，随机撞到现有边的概率极低
                    added_edges.append([u, v])
                    added_edges.append([v, u])

            added_edges_tensor = torch.tensor(added_edges, dtype=torch.long).t()
            new_edge_index = torch.cat([new_edge_index, added_edges_tensor.to(device)], dim=1)

    return new_edge_index

def certify_test_set(model, data,num_classes, test_indices,device):
    """
    对测试集（或指定节点列表）进行批量认证。
    返回：认证结果列表和汇总统计。
    """


    # p_minus: 删边的概率 (例如 0.5)
    # p_plus: 加边的概率 (对于稀疏图，通常设为很小的值或 0)
    p_minus = 0.4
    p_plus = 0.0001  # 如果只关注删边鲁棒性
    n_samples = 10000  # 采样数，越大界越紧，建议 1000+
    alpha = 0.05  # 置信度 (1-alpha=95%)
    num_nodes = data.num_nodes
    # 初始化全图投票计数器 [节点数, 类别数]
    all_votes = np.zeros((num_nodes, num_classes))

    print(f">>> 正在进行并行采样 (N={n_samples}, p_minus={p_minus}, p_plus={p_plus})...")

    with torch.inference_mode():
        for _ in tqdm(range(n_samples)):
            noisy_edges = get_perturbed_adj(data.edge_index, num_nodes,
                                                 p_plus, p_minus, True)
            logits = model(data.x, noisy_edges)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_votes[np.arange(num_nodes), preds] += 1

    # ==========================================
    # 3. 严格的半径计算逻辑 (修复 Neyman-Pearson 似然比)
    # ==========================================
    test_indices = np.where(data.test_mask.cpu().numpy())[0]
    all_radii = []

    eps = 1e-15
    # 【理论严谨】如果攻击者仅实施“删边”攻击，其最坏情况的对数似然比代价为 ln((1-p_add)/p_del)
    # 当 p_del=0.4, p_add=1e-4 时，代价约为 ln(0.9999/0.4) ≈ 0.916
    # 当 p_del=0.1, p_add=1e-4 时，代价约为 ln(0.9999/0.1) ≈ 2.302
    cost_deletion_only = np.log((1 - p_plus) / (p_minus + eps))

    print(">>> 正在计算抵御【删边攻击】的认证半径...")
    for idx in test_indices:
        node_counts = all_votes[idx]
        top_class = np.argmax(node_counts)
        n_top = node_counts[top_class]

        # Clopper-Pearson 置信下界 (1-alpha 置信度)
        p_lower = proportion_confint(n_top, n_samples, alpha=2 * alpha, method='beta')[0]

        radius = 0
        if p_lower > 0.5:
            # Neyman-Pearson 预算
            budget = np.log(p_lower / (1 - p_lower + eps))
            # 半径 = 预算 / 删边单步代价
            radius = int(np.floor(budget / cost_deletion_only))

        # 只有在干净图上预测正确，且认证半径>0，才计入真实半径
        is_correct = (top_class == data.y[idx].item())
        actual_radius = radius if is_correct else 0
        all_radii.append(actual_radius)

    all_radii = np.array(all_radii)
    mcr = np.mean(all_radii)

    print("\n" + "=" * 50)
    print(f"平均认证半径 (MCR_del): {mcr:.4f} 条边")
    print("-" * 50)
    print("不同删边数量阈值下的认证准确率 (Certified Accuracy against Deletions):")

    max_display_radius = min(5, int(np.max(all_radii)) if len(all_radii) > 0 else 5)
    for r in range(1, max_display_radius + 1):
        ca_at_r = np.sum(all_radii >= r) / len(all_radii)
        print(f"  - R_del >= {r}: CA = {ca_at_r:.2%}")
    print("=" * 50)
