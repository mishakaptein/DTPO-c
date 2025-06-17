import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="out", help="the directory with result files"
)
args = parser.parse_args()

all_results = []
for result_dir in os.listdir(args.output_dir):
    path_prefix = f"{args.output_dir}/{result_dir}"

    if not os.path.isdir(path_prefix):
        continue

    result_path = f"{path_prefix}/results.json"
    config_path = f"{path_prefix}/config.json"

    try:
        with open(result_path) as file:
            results = json.load(file)
    except:
        print(f"{path_prefix} missing the result file, skipping")
        continue

    try:
        with open(config_path) as file:
            config = json.load(file)
    except:
        print(f"{path_prefix} missing the config file, skipping")
        continue

    results.update(config)
    all_results.append(results)

result_df = pd.DataFrame(all_results)

keep_columns = [
    "env_name",
    "num_discrete_actions",
    "mean_return",
    "sem_return",
    "seed",
]
result_df = result_df[keep_columns]

# Sort by discretization resolution
result_df.sort_values(by=["env_name", "num_discrete_actions", "seed"], inplace=True)

mean_table = pd.pivot_table(
    result_df,
    index="num_discrete_actions",
    columns="env_name",
    values="mean_return",
    aggfunc="mean"
)

sem_table = pd.pivot_table(
    result_df,
    index="num_discrete_actions",
    columns="env_name",
    values="mean_return",
    aggfunc="sem"
)

# Sort resolutions in ascending order
mean_table = mean_table.sort_index()
sem_table = sem_table.sort_index()

combined_table = mean_table.copy()
for i in range(mean_table.shape[0]):
    for j in range(mean_table.shape[1]):
        combined_table.iloc[i, j] = f"{mean_table.iloc[i, j]:.2f} \\tiny $\\pm$ {sem_table.iloc[i, j]:.2f}"

combined_table = combined_table.reset_index()
combined_table = combined_table.rename(columns={"num_discrete_actions": "Actions"})

latex_output_path = f"{args.output_dir}/dtpo_discretization_comparison.tex"
combined_table.to_latex(latex_output_path, index=False, escape=False)

print(combined_table)