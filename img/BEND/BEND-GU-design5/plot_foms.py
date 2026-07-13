import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_trajectory(foms, split_index, adding_num, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = np.abs(foms) 
    
    plt.scatter(indices[:split_index], transmissions[:split_index], 
                s=5, c='red', alpha=0.6, label='Initial Samples')
                
    if len(foms) > split_index:
        plt.scatter(indices[split_index:], transmissions[split_index:], 
                    s=5, c='blue', alpha=0.6, label='Optimized Samples')
    
    plt.axvline(x=split_index, color='red', linestyle='--', linewidth=2, label='Opt Start')
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Transmission', fontsize=16)
    plt.title('Optimization Trajectory', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ax = plt.gca()
    if adding_num > 0 and len(foms) > split_index:
        max_iter = max(0, (len(foms) - split_index) // adding_num)
        if max_iter > 0:
            step = max(1, max_iter // 10)
            iter_ticks = np.arange(0, max_iter + 1, step)
            
            physical_ticks = split_index + iter_ticks * adding_num
            ax.set_xticks(physical_ticks)
            ax.set_xticklabels([str(t) for t in iter_ticks])
            
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "transmission_trajectory.png"), dpi=150)
    plt.close()

def plot_fom_history(fom_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration", fontsize=18)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Best FOM", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    with open('final_result.json', 'r') as f:
        data = json.load(f)
        
    foms = data['results']['all_evaluated_foms']
    fom_history = data['results']['fom_evolution_history']
    
    # Read hyperparameters for split index and adding num
    split_index = data['hyperparameters']['INIT_SIM_COUNT']
    adding_num = data['hyperparameters']['SAMPLES_PER_ITER']
    
    # Generate the plots in the current directory
    output_folder = '.'
    plot_optimization_trajectory(foms, split_index, adding_num, output_folder)
    plot_fom_history(fom_history, output_folder)
    
    print(f"Successfully generated plots: fom_trajectory.png and fom_evolution.png")
