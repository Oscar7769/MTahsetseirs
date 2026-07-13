import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import meep as mp

# ==============================================================================
# SECTION 0: GLOBAL PARAMETERS
# ==============================================================================

RESOLUTION = 40
DPML = 1.0
MDM_LC = 5.0      # 設計區域大小
WG_LENGTH = 1.0   # 波導長度
WG_WIDTH = 1.0    # 波導寬度

SX = 2 * DPML + MDM_LC + 2 * WG_LENGTH
SY = 2 * DPML + MDM_LC + 2 * WG_LENGTH
CELL = mp.Vector3(SX, SY, 0)

N_SIO2 = 1.44
N_SI = 3.48
SIO2_MEDIUM = mp.Medium(index=N_SIO2)
SI_MEDIUM = mp.Medium(index=N_SI)

wl_cen = 1.55 
FCEN = 1 / wl_cen
DF = 0.1 * FCEN
NFREQ = 1

KERNEL_SIZE = 5
KERNEL_SIGMA = 1.0
TANH_BETA = 50
TANH_ETA = 0.5

# ==============================================================================
# SECTION 1: GEOMETRY
# ==============================================================================

def gaussian_kernel(size=KERNEL_SIZE, sigma=KERNEL_SIGMA):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def tanh_projection(x, beta=TANH_BETA, eta=TANH_ETA):
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den

def enforce_reflection_symmetry(matrix_quadrant):
    """將四分之一的矩陣 (Q) 擴充為具備四重對稱 (上下、左右、對角線) 的完整矩陣"""
    Q = np.array(matrix_quadrant)
    Q_upper = np.triu(Q)
    Q_symmetric = Q_upper + np.triu(Q, 1).T 
    top_half = np.hstack((Q_symmetric, np.fliplr(Q_symmetric)))
    bottom_half = np.hstack((np.flipud(Q_symmetric), np.flipud(np.fliplr(Q_symmetric))))
    return np.vstack((top_half, bottom_half))

def get_projected_density_matrix(binary_vector, grid_rows, grid_cols):
    vec_len = len(binary_vector)
    if vec_len == grid_rows * grid_cols:
        grid_matrix = np.array(binary_vector).astype(float).reshape((grid_rows, grid_cols))
    elif vec_len == (grid_rows // 2) * (grid_cols // 2):
        q_rows = grid_rows // 2
        q_cols = grid_cols // 2
        Q = np.array(binary_vector).astype(float).reshape((q_rows, q_cols))
        grid_matrix = enforce_reflection_symmetry(Q)
    else:
        raise ValueError(f"Invalid binary_vector size: {vec_len}")

    kernel = gaussian_kernel(size=KERNEL_SIZE, sigma=KERNEL_SIGMA)
    try:
        from scipy.signal import convolve2d
        density = convolve2d(grid_matrix, kernel, mode='same', boundary='symm')
    except Exception:
        k = kernel.shape[0]
        pad = k // 2
        img_p = np.pad(grid_matrix, ((pad, pad), (pad, pad)), mode='reflect')
        density = np.zeros_like(grid_matrix)
        for i in range(grid_rows):
            for j in range(grid_cols):
                density[i, j] = np.sum(img_p[i:i+k, j:j+k] * kernel)
    
    projected_density = tanh_projection(density, beta=TANH_BETA, eta=TANH_ETA)
    projected_density = np.clip(projected_density, 0.0, 1.0)
    return projected_density

def create_projected_geometry(binary_vector, grid_rows, grid_cols):
    density = get_projected_density_matrix(binary_vector, grid_rows, grid_cols)
    weights = density.flatten()
    material_grid = mp.MaterialGrid(mp.Vector3(grid_cols, grid_rows), SIO2_MEDIUM, SI_MEDIUM, weights=weights)
    material_grid.smoothing_radius = 1.0 
    design_block = mp.Block(size=mp.Vector3(MDM_LC, MDM_LC, mp.inf), center=mp.Vector3(), material=material_grid)
    return [design_block]

def replot_all():
    npy_path = "/home/oscar3102/跑論文的圖/Crossing/Crossing-GU-Design2/best_config_10x10.npy"
    json_path = "/home/oscar3102/跑論文的圖/Crossing/Crossing-GU-Design2/final_result.json"
    output_folder = "/home/oscar3102/跑論文的圖/Crossing/Crossing-GU-Design2"
    
    print("Processing:")
    print(f"  NPY: {npy_path}")
    print(f"  JSON: {json_path}")
    print(f"  Output Folder: {output_folder}")
    
    print(f"Loading NPY: {npy_path}")
    best_config_NxN = np.load(npy_path)
    grid_rows, grid_cols = best_config_NxN.shape
    # 由於當初存檔時做了 config_NxN.T，所以這裡要轉回來
    best_config = best_config_NxN.T.flatten()
    
    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    foms = np.array(data['results']['all_evaluated_foms'])
    fom_history = data['results']['fom_evolution_history']
    split_index = data['hyperparameters']['INIT_SIM_COUNT']
    adding_num = data['hyperparameters']['SAMPLES_PER_ITER']
    
    # 1. 繪製 Best FoM vs Iteration 圖
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FoM vs Iteration", fontsize=22)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("FoM", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "replot_fom_evolution.png"), dpi=300)
    plt.close()
    
    # 2. 繪製 Optimization Trajectory 圖
    plt.figure(figsize=(12, 6))
    iterations = np.arange(1, len(foms) + 1)
        
    plt.scatter(iterations[:split_index], foms[:split_index], 
                s=5, c='red', alpha=0.6, label='Initial Samples')
                
    if len(foms) > split_index:
        plt.scatter(iterations[split_index:], foms[split_index:], 
                    s=5, c='blue', alpha=0.6, label='Optimized Samples')
    
    if len(foms) > split_index:
        plt.axvline(x=split_index + 0.5, color='red', linestyle='--', linewidth=2, label='Optimized Start')
    
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Optimization FOM', fontsize=20)
    plt.title('Optimization Trajectory', fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=15, loc='upper right')
    
    ax = plt.gca()
    if len(foms) >= split_index:
        num_iters = int(np.ceil((len(foms) - split_index) / adding_num))
        step = max(1, num_iters // 10)
        tick_iters = np.arange(0, num_iters + 1, step)
        tick_positions = split_index + tick_iters * adding_num
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_iters])
    else:
        ax.set_xticks([split_index])
        ax.set_xticklabels(['0'])
        
    ax.tick_params(axis='both', which='major', labelsize=16)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "replot_fom_trajectory.png"), dpi=300)
    plt.close()

    # 3. 繪製 Best config 圖
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(best_config_NxN, cmap='gray_r', origin='lower', 
              extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2]) 
    
    ax.set_xticks(np.linspace(-MDM_LC/2, MDM_LC/2, grid_cols+1), minor=True)
    ax.set_yticks(np.linspace(-MDM_LC/2, MDM_LC/2, grid_rows+1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0) 
    ax.set_axisbelow(False)
    
    ax.set_title(f"Best Binary Configuration ({grid_rows}x{grid_cols})", fontsize=22)
    ax.set_xlabel("x ($\mu$m)", fontsize=20)
    ax.set_ylabel("y ($\mu$m)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"replot_best_config_{grid_rows}x{grid_cols}.png"), dpi=300)
    plt.close()
    
    # 4. MEEP 模擬與畫 DFT 圖
    print("Running MEEP simulation to generate DFT and smoothed structure...")
    mdm_structure = create_projected_geometry(best_config, grid_rows, grid_cols)
    
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2    # 西
    through_wg_center_x = MDM_LC / 2 + (WG_LENGTH + DPML) / 2   # 東
    cross_top_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2    # 北
    cross_bot_center_y = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2   # 南
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), center=mp.Vector3(input_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), center=mp.Vector3(through_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), center=mp.Vector3(0, cross_top_center_y), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), center=mp.Vector3(0, cross_bot_center_y), material=SI_MEDIUM),
    ]
    
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH) 
    mon_x_through = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_top = MDM_LC / 2 + WG_LENGTH / 2
    mon_y_cross_bot = -MDM_LC / 2 - WG_LENGTH / 2
    monitor_size_y = mp.Vector3(0, WG_WIDTH * 3) # X方向波導的截面
    monitor_size_x = mp.Vector3(WG_WIDTH * 5, 0) # Y方向波導的截面 
    
    mode_definitions = ['TE0', 'TE1']
    
    for mode_name in mode_definitions:
        print(f"Simulating Mode: {mode_name}")
        mp.Simulation(cell_size=CELL, resolution=1, boundary_layers=[]).reset_meep()
        if mode_name == 'TE1':
            sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                          center=src_center, size=src_size, direction=mp.X,
                                          eig_kpoint=mp.Vector3(1, 0, 0), eig_match_freq=True,
                                          eig_band=2)]
        else: # TE0
            sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                          center=src_center, size=src_size, direction=mp.X, 
                                          eig_band=1, eig_parity=mp.EVEN_Y)]
    
        sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
        if mode_name == 'TE1':
            norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=monitor_size_y), direction=mp.X)
            flux_through = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(mon_x_through, 0), size=monitor_size_y), direction=mp.X)
            flux_cross_top = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(0, mon_y_cross_top - 0.5), size=monitor_size_x), direction=mp.Y)
            flux_cross_bot = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(0, mon_y_cross_bot - 0.5), size=monitor_size_x), direction=mp.Y)
            dft_monitor = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(), size=mp.Vector3(SX, SY))
        
            sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(mon_x_through, 0), 1e-4))
        else: # TE0
            norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=monitor_size_y))
            flux_through = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(mon_x_through, 0), size=monitor_size_y), direction=mp.X)
            flux_cross_top = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(0, mon_y_cross_top), size=monitor_size_x), direction=mp.Y)
            flux_cross_bot = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(0, mon_y_cross_bot), size=monitor_size_x), direction=mp.Y)
            dft_monitor = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(), size=mp.Vector3(SX, SY))
        
            sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(mon_x_through, 0), 1e-4))
            
        ez_dft_data = sim.get_dft_array(dft_monitor, mp.Ez, 0)
        eps_data = sim.get_epsilon()
        intensity = np.abs(ez_dft_data)**2
        
        x = np.linspace(-SX/2, SX/2, intensity.shape[0])
        y = np.linspace(-SY/2, SY/2, intensity.shape[1])

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   cmap='jet', origin='lower')
        ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Intensity $|Ez|^2$', size=18)
        cbar.ax.tick_params(labelsize=16)
        
        mode_latex = "$TE_0$" if mode_name == 'TE0' else "$TE_1$"
        ax.set_title(f"DFT Field Distribution ({mode_latex})", fontsize=22)
        ax.set_xlabel("x ($\mu$m)", fontsize=20)
        ax.set_ylabel("y ($\mu$m)", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"replot_Optimized_Final_Structure_{mode_name}.png"), dpi=300)
        plt.close(fig)

    # 5. 二值化結構圖 Smoothed Binary Structure
    x_mask = (x >= -MDM_LC/2) & (x <= MDM_LC/2)
    y_mask = (y >= -MDM_LC/2) & (y <= MDM_LC/2)
    eps_design = eps_data[np.ix_(x_mask, y_mask)]
    
    threshold_eps = (N_SI**2 + N_SIO2**2) / 2
    binary_design = (eps_design > threshold_eps).astype(int)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_design.T, extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2], 
              cmap='gray_r', origin='lower')
    ax.set_title("Smoothed Binary Structure", fontsize=22)
    ax.set_xlabel(r"x ($\mu$m)", fontsize=20)
    ax.set_ylabel(r"y ($\mu$m)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "replot_Smoothed_Binary_Structure.png"), dpi=300)
    plt.close(fig)
    
    print("All plots generated successfully!")

if __name__ == '__main__':
    replot_all()
