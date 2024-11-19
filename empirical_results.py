#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#%%
# Define the datasets and attack types
datasets = ['Cora', 'CiteSeer']
attack_types = ['random', 'fg_random', 'fg_betweenness']

# Create a list to store the dataframes
dataframes = []

# Define the directory where CSV files are stored
data_dir = "attack_results"  # Adjust this path if necessary

# Load each dataframe from its CSV using its config names
for dataset_name in datasets:
    for attack_type in attack_types:
        filename = f"{dataset_name}_{attack_type}.csv"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['dataset'] = dataset_name
            df['attack_type'] = attack_type
            dataframes.append(df)
        else:
            print(f"File not found: {filepath}")

# Check if dataframes list is empty
if not dataframes:
    raise ValueError("No dataframes loaded. Please check the CSV files and paths.")

# Define the metrics and accuracies
metrics = ["adj_spectral_distance", "laplacian_spectral_distance", "edit_distance",
           "deltacon_distance", "resistance_distance", "gcn_accuracy", "label_propagation_accuracy"]

distance_metrics = ["adj_spectral_distance", "laplacian_spectral_distance", "edit_distance",
                    "deltacon_distance", "resistance_distance"]

metrics_and_accuracies = metrics  # For correlation plots

#%%
# ============================================================
# Plot 1: Line plots for evolution of distance metrics and accuracies
# ============================================================

# Create subplot titles for the top row (metrics names)
subplot_titles = []
for row in range(6):
    for col in range(7):
        if row == 0:
            title = metrics[col].replace('_', ' ').title()
        else:
            title = ''
        subplot_titles.append(title)

# Create the subplot figure
fig = make_subplots(rows=6, cols=7,
                    vertical_spacing=0.02, horizontal_spacing=0.04,
                    subplot_titles=subplot_titles)

# Loop over each dataframe and plot the metrics
for i, df in enumerate(dataframes):
    row = i + 1  # Row index starts from 1 in Plotly
    dataset_name = df['dataset'].iloc[0]
    attack_type = df['attack_type'].iloc[0]
    yaxis_title = f"{dataset_name}_{attack_type}"
    for j, metric in enumerate(metrics):
        col = j + 1  # Column index starts from 1
        # Plot the line plot of the metric over iterations
        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df[metric],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ),
            row=row, col=col
        )
        # Set y-axis title for the first column
        if col == 1:
            fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
        # Set x-axis title for the last row
        if row == 6:
            fig.update_xaxes(title_text='Iteration', row=row, col=col)
        # Set the subplot title for the top row (already set via subplot_titles)

# Update layout
fig.update_layout(height=1200, width=1400, showlegend=False,
                  title_text="Evolution of Distance Metrics and Accuracies")

# Save the figure
fig.write_image("figs/line_plots.png", format='png', scale=4)

#%%
# ============================================================
# Plot 2: Pearson correlation heatmaps for each result dataframe
# ============================================================

# Create the subplot figure
fig = make_subplots(rows=2, cols=3,
                    subplot_titles=[f"{df['dataset'].iloc[0]}_{df['attack_type'].iloc[0]}" for df in dataframes],
                    vertical_spacing=0.12, horizontal_spacing=0.12)

# Loop over each dataframe and plot the correlation heatmap
for i, df in enumerate(dataframes):
    row = i // 3 + 1  # Row index
    col = i % 3 + 1   # Column index
    # Compute Pearson correlation matrix
    corr_matrix = df[metrics_and_accuracies].corr(method='pearson')
    # Create heatmap
    heatmap = go.Heatmap(
        z=corr_matrix.values,
        x=metrics_and_accuracies,
        y=metrics_and_accuracies,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation") if (col == 3) else dict(showticklabels=False),
        showscale=True if (col == 3) else False
    )
    # Add heatmap to figure
    fig.add_trace(heatmap, row=row, col=col)
    # Update axes labels
    fig.update_xaxes(tickangle=45, row=row, col=col)
    fig.update_yaxes(row=row, col=col)

# Update layout
fig.update_layout(height=1200, width=1800, title_text="Pearson Correlation Heatmaps")

# Save the figure
fig.write_image("figs/pearson_correlation_heatmaps.png", format='png', scale=4)

#%%
# ============================================================
# Plot 3: Spearman rank correlation heatmaps for each result dataframe
# ============================================================

# Create the subplot figure
fig = make_subplots(rows=2, cols=3,
                    subplot_titles=[f"{df['dataset'].iloc[0]}_{df['attack_type'].iloc[0]}" for df in dataframes],
                    vertical_spacing=0.12, horizontal_spacing=0.12)

# Loop over each dataframe and plot the correlation heatmap
for i, df in enumerate(dataframes):
    row = i // 3 + 1  # Row index
    col = i % 3 + 1   # Column index
    # Compute Spearman correlation matrix
    corr_matrix = df[metrics_and_accuracies].corr(method='spearman')
    # Create heatmap
    heatmap = go.Heatmap(
        z=corr_matrix.values,
        x=metrics_and_accuracies,
        y=metrics_and_accuracies,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation") if (col == 3) else dict(showticklabels=False),
        showscale=True if (col == 3) else False
    )
    # Add heatmap to figure
    fig.add_trace(heatmap, row=row, col=col)
    # Update axes labels
    fig.update_xaxes(tickangle=45, row=row, col=col)
    fig.update_yaxes(row=row, col=col)

# Update layout
fig.update_layout(height=1200, width=1800, title_text="Spearman Rank Correlation Heatmaps")

# Save the figure
fig.write_image("figs/spearman_correlation_heatmaps.png", format='png', scale=4)

#%%
# ============================================================
# PCA Analysis and Plot of Cumulative Explained Variance
# ============================================================

# Create the subplot figure
fig = make_subplots(rows=2, cols=3, shared_yaxes='all',
                    subplot_titles=[f"{df['dataset'].iloc[0]}_{df['attack_type'].iloc[0]}" for df in dataframes],
                    vertical_spacing=0.07, horizontal_spacing=0.07)

# Loop over each dataframe and perform PCA
pca_results = []  # Store PCA results for later use in factor models
for i, df in enumerate(dataframes):
    row = i // 3 + 1  # Row index
    col = i % 3 + 1   # Column index
    # Extract the distance metrics
    X = df[distance_metrics].values
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Perform PCA
    pca = PCA()
    pca.fit(X_scaled)
    pca_results.append((pca, scaler))  # Save for factor model
    # Get cumulative explained variance
    cum_exp_variance = np.cumsum(pca.explained_variance_ratio_)
    # Number of components
    n_components = len(cum_exp_variance)
    # Plot the cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, n_components + 1),
            y=cum_exp_variance,
            mode='lines+markers',
            fill='tozeroy'
            # name='Cumulative Explained Variance'
        ),
        row=row, col=col
    )
    # Add hline at threshold=0.9
    fig.add_shape(
        type='line',
        x0=1, x1=n_components,
        y0=0.999, y1=0.999,
        line=dict(color='red', dash='dash'),
        row=row, col=col
    )
    # Update axes
    fig.update_xaxes(title_text='Number of Components', row=row, col=col)
    fig.update_yaxes(title_text='Cumulative Explained Variance', range=[0.98, 1], row=row, col=col)

# Update layout
fig.update_layout(height=1200, width=1800, title_text="PCA Cumulative Explained Variance", showlegend=False)

# Save the figure
fig.write_image("figs/pca_cumulative_explained_variance.png", format='png', scale=4)

#%%
# ============================================================
# Factor Model: Regression of factors to explain GCN accuracy
# ============================================================

# Create a DataFrame to store R-squared values
r_squared_df = pd.DataFrame(index=datasets, columns=attack_types)

# Define the number of principal components to test
max_pca_components = 5  # You can adjust this as needed

# Create a list to store R-squared values for plotting
r_squared_plot_data = []

for idx, df in enumerate(dataframes):
    dataset_name = df['dataset'].iloc[0]
    attack_type = df['attack_type'].iloc[0]
    # Get the PCA and scaler from earlier
    pca, scaler = pca_results[idx]
    # Extract the distance metrics
    X = df[distance_metrics].values
    # Standardize the data using the same scaler
    X_scaled = scaler.transform(X)
    # Transform data using PCA
    X_pca_full = pca.transform(X_scaled)  # All principal components
    # Dependent variable: GCN accuracy
    y = df['gcn_accuracy'].values
    
    # Compute R-squared for different numbers of principal components
    r_squared_values = []
    for n_components in range(1, max_pca_components + 1):
        X_pca = X_pca_full[:, :n_components]
        # Fit linear regression model
        reg = LinearRegression()
        reg.fit(X_pca, y)
        # Compute R-squared
        r_squared = reg.score(X_pca, y)
        r_squared_values.append(r_squared)
    
    # Store R-squared values for the current configuration
    r_squared_df.loc[dataset_name, attack_type] = r_squared_values[-1]  # Last R-squared value with max components
    
    # Append data for plotting
    r_squared_plot_data.append({
        'dataset': dataset_name,
        'attack_type': attack_type,
        'r_squared_values': r_squared_values
    })

# Define subplot grid dimensions
rows = 2
cols = 3

# Create subplot titles based on dataset and attack type
subplot_titles = [f"{item['dataset']} - {item['attack_type']}" for item in r_squared_plot_data]

# Create the subplot figure
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.1,
    vertical_spacing=0.12
)


# Determine the global min and max R-squared values for consistent coloring
all_r_squared = [val for item in r_squared_plot_data for val in item['r_squared_values']]
min_r_squared = min(all_r_squared)
max_r_squared = max(all_r_squared)

# Iterate over each subplot configuration
for i, item in enumerate(r_squared_plot_data):
    row = (i // cols) + 1
    col = (i % cols) + 1
    dataset = item['dataset']
    attack = item['attack_type']
    r2_values = item['r_squared_values']
    num_components = list(range(1, max_pca_components + 1))
    
    # Create a bar trace
    trace = go.Bar(
        x=num_components,
        y=r2_values,
        marker=dict(
            color=r2_values,
            cmin=min_r_squared,
            cmax=max_r_squared,
            colorbar=dict(
                title="R^2",
                titleside="top",
                ticks="outside",
                showticksuffix="last"
            )
        ),
        text=[f"{val:.3f}" for val in r2_values],
        textposition='auto',
        name=f"{dataset} - {attack}"
    )
    
    # Add the trace to the subplot
    fig.add_trace(trace, row=row, col=col)
    
    # Update x-axis
    fig.update_xaxes(title_text="Number of Principal Components", row=row, col=col)
    # Update y-axis
    fig.update_yaxes(title_text="R^2", range=[0.75, 1], row=row, col=col)

# Update overall layout
fig.update_layout(
    height=1200,  
    width=1800,   
    title_text="R-squared Values of Factor Models Across Configurations",
    showlegend=False
)

# Save the figure as a PNG image with specified scale
fig.write_image("figs/factor_model_r_squared.png", format='png', scale=4)

# Optionally display the figure in an interactive window (requires an appropriate environment)
fig.show()
