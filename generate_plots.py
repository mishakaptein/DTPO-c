import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import csv


def load_results_for_action_counts(action_counts, seeds, base_dir="out"):
    results = defaultdict(list)

    for seed in seeds:
        pattern = f"{base_dir}/Pendulum-v1-BangBang_dtpo_*_{seed}"
        exp_dirs = sorted(Path("../DTPO-c-content").glob(pattern))

        for exp_dir in exp_dirs:
            try:
                with open(exp_dir / "config.json", "r") as f:
                    config = json.load(f)
                    n_actions = config["num_discrete_actions"]

                if n_actions not in action_counts:
                    continue

                with open(exp_dir / "results.json", "r") as f:
                    data = json.load(f)
                    results[n_actions].append({
                        'mean_return': data['mean_return'],
                        'mean_discounted_return': data['mean_discounted_return'],
                        'iterations': data['iterations'],
                        'seed': seed,
                        'learning_curve': data['mean_discounted_returns'],
                        'final_return': data['mean_return']
                    })
            except (FileNotFoundError, KeyError) as e:
                print(f"Skipping {exp_dir} - {str(e)}")
                continue

    return results

def calculate_averages(results):
    averaged = {}
    for n_actions, runs in results.items():
        if not runs:
            continue

        averaged[n_actions] = {
            'iterations': runs[0]['iterations'],
            'mean_return': np.mean([r['mean_return'] for r in runs]),
            'std_return': np.std([r['mean_return'] for r in runs]),
            'mean_discounted_return': np.mean([r['mean_discounted_return'] for r in runs]),
            'std_discounted_return': np.std([r['mean_discounted_return'] for r in runs]),
            'num_seeds': len(runs),
            'seeds_used': [r['seed'] for r in runs]
        }
    return averaged

def save_mean_values(averaged_results, output_dir="out/comparison_plots"):
    with open(f"{output_dir}/mean_values.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Actions',
            'Mean Return',
            'Std Return',
            'Mean Discounted Return',
            'Std Discounted Return',
            'Number of Seeds',
            'Seeds Used'
        ])

        for n_actions, data in sorted(averaged_results.items()):
            writer.writerow([
                n_actions,
                data['mean_return'],
                data['std_return'],
                data['mean_discounted_return'],
                data['std_discounted_return'],
                data['num_seeds'],
                ','.join(map(str, data['seeds_used']))
            ])

def calculate_best_performance(results):
    best_results = defaultdict(list)

    for n_actions, runs in results.items():
        for run in runs:
            best_return = max(run['learning_curve'])
            best_results[n_actions].append({
                'best_return': best_return,
                'seed': run['seed']
            })

    averaged = {}
    for n_actions, runs in best_results.items():
        best_returns = [r['best_return'] for r in runs]
        averaged[n_actions] = {
            'max_best_return': np.max(best_returns),
            'mean_best_return': np.mean(best_returns),
            'best_returns': best_returns,
            'seeds_used': [r['seed'] for r in runs]
        }
    return averaged


def plot_distribution(results, output_dir):
    plt.figure(figsize=(12, 6))
    data = []

    for n_actions, runs in results.items():
        for run in runs:
            data.append({
                'Actions': n_actions,
                'Undiscounted Return': run['final_return']
            })

    df = pd.DataFrame(data)
    sns.boxplot(x='Actions', y='Undiscounted Return', data=df, color='skyblue')

    plt.grid(True, axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
    plt.title("Return Distribution Across Discretization Resolutions")
    plt.xticks(rotation=45)

    plt.savefig(f"{output_dir}/action_distribution_boxplot.png")
    plt.savefig(f"{output_dir}/action_distribution_boxplot.pdf")
    plt.close()

if __name__ == "__main__":
    action_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
    seeds = [1, 2, 3, 4, 5, 6]
    output_dir = "out_plots/comparison_plots"

    print(f"Using action counts: {action_counts}")
    print(f"Using seeds: {seeds}")

    raw_results = load_results_for_action_counts(action_counts, seeds)
    averaged_results = calculate_averages(raw_results)
    best_results = calculate_best_performance(raw_results)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/averaged_results.json", "w") as f:
        json.dump(averaged_results, f, indent=4)

    save_mean_values(averaged_results, output_dir)
    plot_distribution(raw_results, output_dir)

    print("\n=== Results Summary ===")
    for n_actions, data in sorted(averaged_results.items()):
        print(f"{n_actions} actions (seeds: {data['seeds_used']}):")
        print(f"Undiscounted Return: {data['mean_return']:.2f}, std: {data['std_return']:.2f}")
        print(f"Discounted Return: {data['mean_discounted_return']:.2f}, std: {data['std_discounted_return']:.2f}")
        print(f"Averaged over {data['num_seeds']} seeds\n")