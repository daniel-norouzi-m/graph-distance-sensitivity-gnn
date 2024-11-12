import torch
from torch_geometric.utils import to_networkx
import networkx as nx


# Helper function to compute the graph Laplacian
def compute_laplacian(adj):
    degree = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree)
    laplacian = degree_matrix - adj
    return laplacian

# Helper function to compute the spectral distance between two matrices
def spectral_distance(matrix1, matrix2):
    eigenvalues1 = torch.linalg.eigvalsh(matrix1)
    eigenvalues2 = torch.linalg.eigvalsh(matrix2)
    distance = torch.sqrt(torch.sum((eigenvalues1 - eigenvalues2) ** 2)).item()
    return distance

# Helper function to compute the edit distance between two adjacency matrices
def edit_distance(matrix1, matrix2):
    distance = torch.sum(torch.abs(matrix1 - matrix2)).item()
    return distance

# Helper function to compute the DeltaCon belief propagation matrix S
def compute_deltacon_matrix(adj, epsilon=1e-3):
    degree_matrix = torch.diag(torch.sum(adj, dim=1))
    identity_matrix = torch.eye(adj.size(0)).to(adj.device)
    try:
        S = torch.linalg.inv(identity_matrix + epsilon**2 * degree_matrix - epsilon * adj)
    except RuntimeError:
        # In case of singular matrix, add a small identity for regularization
        S = torch.linalg.inv(identity_matrix + epsilon**2 * degree_matrix - epsilon * adj + 1e-5 * torch.eye(adj.size(0)).to(adj.device))
    return S

# Helper function to compute the DeltaCon distance
def deltacon_distance(S1, S2):
    distance = torch.sqrt(torch.sum((torch.sqrt(S1) - torch.sqrt(S2))**2)).item()
    return distance

# Helper function to compute the effective resistance matrix R
def compute_resistance_matrix(laplacian):
    try:
        laplacian_pseudo_inverse = torch.linalg.pinv(laplacian)
    except RuntimeError:
        # In case of singular matrix, add a small identity for regularization
        laplacian_pseudo_inverse = torch.linalg.pinv(laplacian + 1e-5 * torch.eye(laplacian.size(0)).to(laplacian.device))

    resistance_matrix = torch.diag(laplacian_pseudo_inverse)[:, None] + torch.diag(laplacian_pseudo_inverse)[None, :] - 2 * laplacian_pseudo_inverse
    return resistance_matrix

# Helper function to compute the resistance-perturbation distance
def resistance_perturbation_distance(R1, R2):
    distance = torch.sum(torch.abs(R1 - R2)).item()
    return distance

# Helper function to calculate betweenness centrality using NetworkX
def calculate_betweenness_centrality(data):
    G = to_networkx(data, to_undirected=True)
    centrality = nx.betweenness_centrality(G)
    centrality_scores = torch.tensor([centrality[i] for i in range(data.num_nodes)], dtype=torch.float32)
    return centrality_scores