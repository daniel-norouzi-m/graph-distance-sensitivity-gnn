import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.datasets import Planetoid
import numpy as np
import pandas as pd
import random
from sklearn.semi_supervised import LabelPropagation
from tqdm import tqdm
import traceback
import warnings

from graph_distance_metrics import (
    compute_laplacian,
    spectral_distance,
    edit_distance,
    compute_deltacon_matrix,
    deltacon_distance,
    compute_resistance_matrix,
    resistance_perturbation_distance,
    calculate_betweenness_centrality,
)


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, with_relu=True, dropout=0.5):
        super(GCN, self).__init__()
        self.with_relu = with_relu
        self.dropout = dropout
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc1(x)
        if self.with_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        

# Function to perform attacks and collect results        
def perform_attack(
    dataset_name: str,
    n_iterations: int,
    attack_type: str
  ) -> pd.DataFrame:
    """
    Perform targeted attacks on a graph dataset and compute various graph distance metrics and accuracy changes.

    Args:
        dataset_name (str): Name of the dataset ('Cora' or 'CiteSeer').
        n_iterations (int): Number of attack iterations (number of attacked nodes).
        attack_type (str): Type of attack ('random', 'fg_random', 'fg_betweenness').

    Returns:
        pd.DataFrame: DataFrame containing the results of the attacks.
    """
    assert attack_type in ['random', 'fg_random', 'fg_betweenness'], "Invalid attack_type."

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load the dataset
    dataset = Planetoid(root=f'~/somewhere/{dataset_name}', name=dataset_name)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Step 1: Add self-loops to the edge index
    edge_index_with_loops, _ = add_self_loops(data.edge_index)

    # Step 2: Convert edge_index to a dense adjacency matrix (A + I_N)
    original_dense_adj = to_dense_adj(edge_index_with_loops)[0].to(device)
    modified_adj = original_dense_adj.clone().detach()

    # Step 3: Compute the degree matrix
    degree = torch.sum(modified_adj, dim=1)

    # Step 4: Compute D^(-0.5)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # Avoid infinities

    # Step 5: Normalize the adjacency matrix with symmetric normalization
    normalized_adj = degree_inv_sqrt.view(-1, 1) * modified_adj * degree_inv_sqrt.view(1, -1)

    # Compute original graph metrics
    original_laplacian = compute_laplacian(modified_adj)
    original_deltacon_matrix = compute_deltacon_matrix(modified_adj)
    original_resistance_matrix = compute_resistance_matrix(original_laplacian)

    # Initialize the GCN model
    model = GCN(dataset.num_node_features, 16, dataset.num_classes, with_relu=True, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the GCN model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, normalized_adj)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Calculate original GCN accuracy
    model.eval()
    with torch.no_grad():
        out = model(data.x, normalized_adj)
        pred = out.argmax(dim=1)

    # Initialize results dictionary
    results = {
        "attack_type": [],
        "dataset": [],
        "iteration": [],
        "adj_spectral_distance": [],
        "laplacian_spectral_distance": [],
        "edit_distance": [],
        "deltacon_distance": [],
        "resistance_distance": [],
        "gcn_accuracy": [],
        "label_propagation_accuracy": []
    }

    # Calculate betweenness centrality if needed
    if attack_type == 'fg_betweenness':
        betweenness_centrality_scores = calculate_betweenness_centrality(data)
        _, sorted_nodes = torch.topk(betweenness_centrality_scores, k=n_iterations)
        target_nodes = sorted_nodes.tolist()
    else:
        target_nodes = []

    # For 'random' attack, we'll modify links randomly
    # For 'fg_random' and 'fg_betweenness', we'll perform fast gradient attack
    for i in tqdm(range(n_iterations + 1), desc=f"Performing {attack_type} attack on {dataset_name}"):
        if i != 0:
            if attack_type == 'random':
                # Select a random node
                target_node = random.choice(range(data.num_nodes))
            elif attack_type == 'fg_random':
                # Select a random node from the remaining nodes
                target_node = random.choice(range(data.num_nodes))
            elif attack_type == 'fg_betweenness':
                # Select the next node based on betweenness centrality
                if i < len(target_nodes):
                    target_node = target_nodes[i]
                else:
                    target_node = random.choice(range(data.num_nodes))
            else:
                raise ValueError("Invalid attack_type.")

            if attack_type in ['fg_random', 'fg_betweenness']:
                # Perform fast gradient attack
                normalized_adj.requires_grad_(True)
                model.train()
                optimizer.zero_grad()
                out = model(data.x, normalized_adj)
                loss = F.nll_loss(out[target_node].unsqueeze(0), data.y[target_node].unsqueeze(0))
                loss.backward()
                adj_grad = normalized_adj.grad.clone().detach()

                # Symmetrize the gradients
                g_hat = (adj_grad + adj_grad.t()) / 2

                # Extract the most significant links based on gradient magnitude
                topk = 20  # Number of links to consider for attack
                g_hat_flat = g_hat.abs().flatten()
                if topk > g_hat_flat.size(0):
                    topk = g_hat_flat.size(0)
                _, indices = torch.topk(g_hat_flat, k=topk)
                rows = indices // g_hat.size(0)
                cols = indices % g_hat.size(0)

                # Apply the attack by modifying the adjacency matrix based on gradient sign
                for row, col in zip(rows.tolist(), cols.tolist()):
                    if row != col:
                        gradient_sign = torch.sign(g_hat[row, col]).item()
                        modified_adj[row, col] += gradient_sign

                        if modified_adj[row, col] == 1:
                            # Existing edge: potentially remove it
                            z = min(1, modified_adj[row, col])
                        else:
                            # Non-existing edge: potentially add it
                            z = max(0, modified_adj[row, col])

                        modified_adj[row, col] = z
                        modified_adj[col, row] = z

                # Re-normalize the adjacency matrix
                degree = torch.sum(modified_adj, dim=1)
                degree_inv_sqrt = degree.pow(-0.5)
                degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
                normalized_adj = degree_inv_sqrt.view(-1, 1) * modified_adj * degree_inv_sqrt.view(1, -1)
                normalized_adj = normalized_adj.detach()

            elif attack_type == 'random':
                # Perform random attack by adding/removing edges randomly for the target node
                num_edges = 20  # Number of edges to modify
                for _ in range(num_edges):
                    # Decide to add or remove an edge
                    if random.random() > 0.5 and torch.sum(modified_adj[target_node]).item() > 0:
                        # Remove a random existing edge
                        existing_edges = modified_adj[target_node].nonzero(as_tuple=True)[0].tolist()
                        existing_edges = [e for e in existing_edges if e != target_node]
                        if existing_edges:
                            edge_to_remove = random.choice(existing_edges)
                            modified_adj[target_node, edge_to_remove] = 0
                            modified_adj[edge_to_remove, target_node] = 0
                    else:
                        # Add a random edge
                        potential_nodes = list(set(range(data.num_nodes)) - set(modified_adj[target_node].nonzero(as_tuple=True)[0].tolist()) - {target_node})
                        if potential_nodes:
                            edge_to_add = random.choice(potential_nodes)
                            modified_adj[target_node, edge_to_add] = 1
                            modified_adj[edge_to_add, target_node] = 1
                # Re-normalize the adjacency matrix
                degree = torch.sum(modified_adj, dim=1)
                degree_inv_sqrt = degree.pow(-0.5)
                degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
                normalized_adj = degree_inv_sqrt.view(-1, 1) * modified_adj * degree_inv_sqrt.view(1, -1)
                normalized_adj = normalized_adj.detach()
            else:
                raise ValueError("Invalid attack_type.")


        # Compute graph metrics after the attack
        adj_spectral_dist = spectral_distance(original_dense_adj, modified_adj)
        laplacian = compute_laplacian(modified_adj)
        laplacian_spectral_dist = spectral_distance(original_laplacian, laplacian)
        edit_dist = edit_distance(original_dense_adj, modified_adj)
        deltacon_matrix = compute_deltacon_matrix(modified_adj)
        deltacon_dist = deltacon_distance(original_deltacon_matrix, deltacon_matrix)
        resistance_matrix = compute_resistance_matrix(laplacian)
        resistance_dist = resistance_perturbation_distance(original_resistance_matrix, resistance_matrix)

        # Evaluate GCN accuracy on the modified adjacency matrix
        model.eval()
        with torch.no_grad():
            out = model(data.x, normalized_adj)
            pred = out.argmax(dim=1)
            gcn_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Label Propagation - Use built-in LabelPropagation from sklearn
                labels = data.y.cpu().numpy()  # Ground truth labels
                # Evaluate accuracy for label propagation on test nodes only
                modified_adj_np = modified_adj.cpu().numpy()

                # Label propagation on modified adjacency matrix (after betweenness centrality modification)
                label_prop_model = LabelPropagation()
                label_prop_model.fit(modified_adj_np[data.train_mask.cpu().numpy()], labels[data.train_mask.cpu().numpy()])

                # Predict labels for all nodes on the modified graph
                propagated_labels_modified = label_prop_model.predict(modified_adj_np)

                # Evaluate accuracy on the test set using the modified adjacency matrix
                label_prop_acc_modified = (propagated_labels_modified[data.test_mask.cpu().numpy()] == data.y[data.test_mask].cpu().numpy()).sum() / data.test_mask.sum().item()

        except BaseException:
            print(traceback.format_exc())
            # In case LabelPropagation fails due to connectivity issues
            label_prop_acc_modified = np.nan

        # Append results
        results["attack_type"].append(attack_type)
        results["dataset"].append(dataset_name)
        results["iteration"].append(i + 1)
        results["adj_spectral_distance"].append(adj_spectral_dist)
        results["laplacian_spectral_distance"].append(laplacian_spectral_dist)
        results["edit_distance"].append(edit_dist)
        results["deltacon_distance"].append(deltacon_dist)
        results["resistance_distance"].append(resistance_dist)
        results["gcn_accuracy"].append(gcn_acc)
        results["label_propagation_accuracy"].append(label_prop_acc_modified)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    return df_results