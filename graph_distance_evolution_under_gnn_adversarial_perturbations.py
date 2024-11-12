import os

from graph_attacks import perform_attack

datasets = ['Cora', 'CiteSeer', 'PubMed']
attack_types = ['random', 'fg_random', 'fg_betweenness']
n_iterations = 60

# Define the output directory
output_dir = "attack_results"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Loop over each dataset
for dataset_name in datasets:
    # Loop over each attack type
    for attack_type in attack_types:
        print(f"\nPerforming '{attack_type}' attack on '{dataset_name}' for {n_iterations} iterations...")
        try:
            # Perform the attack and get the results DataFrame
            df = perform_attack(dataset_name, n_iterations, attack_type)

            # Construct the filename with configuration details
            filename = f"{dataset_name}_{attack_type}.csv"

            # Define the full path to save the CSV
            filepath = os.path.join(output_dir, filename)

            # Save the DataFrame to CSV
            df.to_csv(filepath, index=False)

            print(f"Results saved to '{filepath}'.")
        except Exception as e:
            print(f"An error occurred while performing the attack: {e}")