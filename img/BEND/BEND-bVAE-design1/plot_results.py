import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 絕對路徑指定到 shared_modules
shared_path = "/home/oscar3102/FMQA/90_degree_bend_waveguide"

if shared_path not in sys.path:
    sys.path.append(shared_path)

import meep as mp
from shared_modules.meep_utils import (
    RESOLUTION, DPML, MDM_LC, WG_LENGTH, WG_WIDTH,
    SX, SY, CELL, N_SIO2, N_SI, SIO2_MEDIUM, SI_MEDIUM,
    WL_CEN, FCEN, DF, NFREQ
)

def get_meep_geometry(design_grid_64x64):
    IMG_SIZE = 64
    grid_mod = design_grid_64x64.reshape(IMG_SIZE, IMG_SIZE).copy()
    mid = IMG_SIZE // 2
    wg_half_px = 4 
    
    grid_mod[mid-wg_half_px:mid+wg_half_px, 0:5] = 1.0 
    grid_mod[-5:, mid-wg_half_px:mid+wg_half_px] = 1.0 

    meep_weights = np.ascontiguousarray(grid_mod.T)
    grid = mp.MaterialGrid(mp.Vector3(IMG_SIZE, IMG_SIZE), SIO2_MEDIUM, SI_MEDIUM, weights=meep_weights)
    
    design_region = mp.Block(size=mp.Vector3(MDM_LC, MDM_LC, mp.inf), center=mp.Vector3(0, 0), material=grid)
    input_wg = mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), center=mp.Vector3(-MDM_LC/2 - (WG_LENGTH+DPML)/2 + 0.05, 0), material=SI_MEDIUM)
    output_wg = mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), center=mp.Vector3(0, MDM_LC/2 + (WG_LENGTH+DPML)/2 - 0.05), material=SI_MEDIUM)
    
    return [design_region, input_wg, output_wg]

def plot_optimization_trajectory(foms, split_index, adding_num, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = np.abs(foms) 
    
    plt.scatter(indices[:split_index], transmissions[:split_index], 
                s=5, c='red', alpha=0.6, label='Initial Samples')
                
    if len(foms) > split_index:
        plt.scatter(indices[split_index:], transmissions[split_index:], 
                    s=5, c='blue', alpha=0.6, label='Optimized Samples')
    
    plt.axvline(x=split_index, color='red', linestyle='--', linewidth=2, label='Optimized Start')
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Transmission', fontsize=16)
    plt.title('Optimization Trajectory', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    ax = plt.gca()
    if adding_num > 0 and len(foms) > split_index:
        max_iter = max(0, (len(foms) - split_index) // adding_num)
        if max_iter > 0:
            step = max(1, max_iter // 10)
            iter_ticks = np.arange(0, max_iter + 1, step)
            
            physical_ticks = split_index + iter_ticks * adding_num
            ax.set_xticks(physical_ticks)
            ax.set_xticklabels([str(t) for t in iter_ticks])
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "transmission_trajectory.png"), dpi=150)
    plt.close()

def plot_fom_history(fom_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Best FOM (Transmission)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "/home/oscar3102/跑論文的圖/BEND/BEND-bVAE-design1"
    print(f"Processing directory: {base_dir}")
    import glob
    latent_files = glob.glob(os.path.join(base_dir, "best_latent_*.npy"))
    latent_file = latent_files[0] if latent_files else None
    structure_file = os.path.join(base_dir, "best_structure_64x64.npy")
    json_file = os.path.join(base_dir, "training_log.json")
    output_dir = os.path.join(base_dir, "replot_results")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 繪製 Latent Space
    if latent_file and os.path.exists(latent_file):
        print("Plotting Latent Space...")
        latent_2d = np.load(latent_file)
        plt.figure(figsize=(4, 4))
        plt.imshow(latent_2d, cmap='gray', interpolation='none')
        plt.title(f"Best Latent Representation ({latent_2d.shape[0]}x{latent_2d.shape[1]})")
        
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, latent_2d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, latent_2d.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)
        
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.savefig(os.path.join(output_dir, f"best_latent_{latent_2d.shape[0]}x{latent_2d.shape[1]}.png"), dpi=300)
        plt.close()
    else:
        print(f"File not found: {latent_file}")

    # 2. 繪製 二值化結構圖
    binary_grid = None
    if os.path.exists(structure_file):
        print("Plotting Binarized Structure...")
        binary_grid = np.load(structure_file)
        plt.figure(figsize=(5, 5))
        plt.imshow(binary_grid, cmap='gray_r', origin='lower')
        plt.title("Binarized Structure (64x64)")
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.savefig(os.path.join(output_dir, "binarized_structure.png"), dpi=300)
        plt.close()
    else:
        print(f"File not found: {structure_file}")

    # 3. 讀取 JSON 並繪製 Optimization Trajectory 與 Best FoM vs Iteration
    if os.path.exists(json_file):
        print("Reading JSON and plotting metrics...")
        with open(json_file, 'r') as f:
            experiment_log = json.load(f)

        fom_history = experiment_log["results"]["fom_evolution_history"]
        all_foms = experiment_log["results"]["all_evaluated_foms"]
        split_index = experiment_log["hyperparameters"]["INIT_SIM_COUNT"]
        adding_num = experiment_log["hyperparameters"].get("SAMPLES_PER_ITER", 30)

        # Best FoM vs Iteration
        plot_fom_history(fom_history, output_dir)

        # Optimization Trajectory
        plot_optimization_trajectory(all_foms, split_index, adding_num, output_dir)
    else:
        print(f"File not found: {json_file}")

    # 4. 重新繪製 DFT 場圖
    if binary_grid is not None:
        print("Running MEEP simulation for DFT Field...")
        geometry = get_meep_geometry(binary_grid)
        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(FCEN, fwidth=DF),
                center=mp.Vector3(-SX / 2 + DPML + 0.2, 0),
                size=mp.Vector3(0, WG_WIDTH),
                direction=mp.X,
                eig_band=1,
                eig_parity=mp.EVEN_Y
            )
        ]
        
        sim = mp.Simulation(
            cell_size=CELL,
            boundary_layers=[mp.PML(DPML)],
            geometry=geometry,
            sources=sources,
            resolution=RESOLUTION,
            default_material=SIO2_MEDIUM
        )
        
        dft_obj = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(0,0), size=mp.Vector3(SX, SY))
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, MDM_LC/2 + WG_LENGTH/2), 1e-4))
        
        eps_data = sim.get_epsilon()
        ez_data = sim.get_dft_array(dft_obj, mp.Ez, 0)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow((np.abs(ez_data)**2).T, interpolation='spline36', cmap='jet', origin='lower', extent=[-SX/2, SX/2, -SY/2, SY/2])
        ax.contour(eps_data.T, levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.3, extent=[-SX/2, SX/2, -SY/2, SY/2])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = plt.colorbar(im, cax=cax)
        colorbar.set_label('Intensity $|E_z|^2$', fontsize=16)
        colorbar.ax.tick_params(labelsize=14)
        
        ax.set_title("Bend Waveguide DFT Field Distribution", fontsize=18)
        ax.set_xlabel("x ($\mu m$)", fontsize=16)
        ax.set_ylabel("y ($\mu m$)", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.savefig(os.path.join(output_dir, "final_field.png"), dpi=300)
        plt.close()
        print("DFT Field plot saved.")
    else:
        print("Cannot plot DFT field because binary grid is not available.")

if __name__ == "__main__":
    main()
