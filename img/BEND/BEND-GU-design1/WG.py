"""
================================================================================
基於 FMQA 的光子 90度轉彎器 (TE0 Bend) 逆向設計
- 修改版: 左側輸入 -> 中間轉彎 -> 上方輸出 (TE0 模式) -
- 修復: 移除無效的 mp.EVEN_X 參數 -
================================================================================

執行方式：
  mpirun -np 4 python mdm_inverse_design_bend.py --name JobName
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.ndimage 

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dimod
from dwave.samplers import SimulatedAnnealingSampler

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Warning: dwave-ocean-sdk not installed. QA mode will fail.")

import meep as mp
from factorization_machine import FactorizationMachine

# ==============================================================================
# SECTION 0: GLOBAL PARAMETERS
# ==============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

WORK_TAG = 1
STOP_TAG = 2

RESOLUTION = 40
DPML = 1.0
MDM_LC = 4.0      # 設計區域大小
WG_LENGTH = 1.0   # 波導長度
WG_WIDTH = 0.5    # 波導寬度

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
NFREQ = 100

# ==============================================================================
# SECTION 1: GEOMETRY & HELPERS
# ==============================================================================

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def tanh_projection(x, beta, eta=0.5):
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den

def get_projected_density_matrix(binary_vector, grid_rows, grid_cols, sigma=0.5, kernel_size=5, beta=50, eta=0.5):
    grid_matrix = np.array(binary_vector).astype(float).reshape((grid_rows, grid_cols))
    kernel = gaussian_kernel(size=kernel_size, sigma=sigma)
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
    
    projected_density = tanh_projection(density, beta=beta, eta=eta)
    projected_density = np.clip(projected_density, 0.0, 1.0)
    return projected_density

def apply_port_constraints(binary_vector, grid_rows, grid_cols):
    """
    設定波導端口約束：
    1. 左側輸入 (Left Input)
    2. 上方輸出 (Top Output)
    """
    grid_matrix = np.array(binary_vector).reshape((grid_rows, grid_cols))
    
    pixel_height = MDM_LC / grid_rows
    pixel_width = MDM_LC / grid_cols
    
    pad = 1 # 緩衝區像素

    # --- 1. 左側輸入端口 (Input Port at Left, y=0) ---
    y_in_top = WG_WIDTH / 2
    y_in_bot = -WG_WIDTH / 2
    
    r_in_start = int(np.floor((MDM_LC/2 - y_in_top) / pixel_height))
    r_in_end = int(np.ceil((MDM_LC/2 - y_in_bot) / pixel_height))
    
    r_in_start = max(0, r_in_start)
    r_in_end = min(grid_rows, r_in_end)

    grid_matrix[r_in_start:r_in_end, 0] = 1.0
    
    if r_in_start - pad >= 0:
        grid_matrix[r_in_start-pad:r_in_start, 0] = 0.0
    if r_in_end + pad <= grid_rows:
        grid_matrix[r_in_end:r_in_end+pad, 0] = 0.0

    # --- 2. 上方輸出端口 (Output Port at Top, x=0) ---
    x_out_left = -WG_WIDTH / 2
    x_out_right = WG_WIDTH / 2
    
    c_out_start = int(np.floor((x_out_left - (-MDM_LC/2)) / pixel_width))
    c_out_end = int(np.ceil((x_out_right - (-MDM_LC/2)) / pixel_width))
    
    c_out_start = max(0, c_out_start)
    c_out_end = min(grid_cols, c_out_end)
    
    grid_matrix[0, c_out_start:c_out_end] = 1.0
    
    if c_out_start - pad >= 0:
        grid_matrix[0, c_out_start-pad:c_out_start] = 0.0
    if c_out_end + pad <= grid_cols:
        grid_matrix[0, c_out_end:c_out_end+pad] = 0.0

    return grid_matrix.flatten()

def create_projected_geometry(binary_vector, grid_rows, grid_cols, beta=50):
    constrained_vector = apply_port_constraints(binary_vector, grid_rows, grid_cols)
    density = get_projected_density_matrix(constrained_vector, grid_rows, grid_cols, beta=beta)
    weights = density.flatten()
    material_grid = mp.MaterialGrid(mp.Vector3(grid_cols, grid_rows), SIO2_MEDIUM, SI_MEDIUM, weights=weights)
    material_grid.smoothing_radius = 1.0 
    design_block = mp.Block(size=mp.Vector3(MDM_LC, MDM_LC, mp.inf), center=mp.Vector3(), material=material_grid)
    return [design_block]

def generate_smooth_random_config(rows, cols, threshold=0.5):
    small_r, small_c = max(1, rows // 2), max(1, cols // 2)
    noise = np.random.rand(small_r, small_c)
    smooth_noise = scipy.ndimage.zoom(noise, zoom=2, order=1)
    smooth_noise = smooth_noise[:rows, :cols]
    binary = (smooth_noise > threshold).astype(int).flatten()
    return binary

# ==============================================================================
# SECTION 2: OPTIMIZATION SIMULATION
# ==============================================================================

def evaluate_mdm_fom(binary_vector, grid_rows, grid_cols):
    mp.Simulation(cell_size=CELL, resolution=1, boundary_layers=[]).reset_meep()
    mdm_structure = create_projected_geometry(binary_vector, grid_rows, grid_cols, beta=50)
    
    # 定義固定波導
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2
    input_wg_center_y = 0
    
    output_wg_center_x = 0
    output_wg_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2
    
    fixed_geometry = [
        # 左側輸入
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(input_wg_center_x, input_wg_center_y), material=SI_MEDIUM),
        # 上方輸出
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(output_wg_center_x, output_wg_center_y), material=SI_MEDIUM),
    ]
    
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH * 3) 

    monitor_y_pos = MDM_LC / 2 + WG_LENGTH / 2
    monitor_size = mp.Vector3(WG_WIDTH * 3, 0) 

    # 僅使用 TE0 (band 1, even parity for input)
    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                  center=src_center, size=src_size, direction=mp.X, 
                                  eig_band=1, eig_parity=mp.EVEN_Y)]
    
    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
    # Monitors
    norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, 
                                     mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=src_size))
    
    flux_out = sim.add_mode_monitor(FCEN, DF, NFREQ, 
                                    mp.ModeRegion(center=mp.Vector3(0, monitor_y_pos), size=monitor_size), 
                                    direction=mp.Y) 

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, monitor_y_pos), 1e-6))
    
    # 計算傳輸率
    input_flux = sim.get_eigenmode_coefficients(norm_flux, [1], eig_parity=mp.EVEN_Y).alpha[0, 0, 0]
    input_power = np.abs(input_flux)**2 + 1e-12
    
    # [FIXED] 移除無效的 eig_parity=mp.EVEN_X
    out_coeff = sim.get_eigenmode_coefficients(flux_out, [1]).alpha[0, 0, 0]
    output_power = np.abs(out_coeff)**2
    
    transmission = output_power / input_power
    
    sim.reset_meep()
    
    return -transmission

# ==============================================================================
# SECTION 3: VISUALIZATION & ANALYSIS
# ==============================================================================

def perform_detailed_final_analysis(best_config, grid_rows, grid_cols, output_folder):
    print(f"\n>>> Starting Final Detailed MEEP Analysis (TE0 Bend) <<<")
    
    mdm_structure = create_projected_geometry(best_config, grid_rows, grid_cols, beta=50)
    
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2
    output_wg_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(input_wg_center_x, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(0, output_wg_center_y), material=SI_MEDIUM),
    ]
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH * 3)
    
    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                  center=src_center, size=src_size, direction=mp.X, 
                                  eig_band=1, eig_parity=mp.EVEN_Y)]

    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
    monitor_y_pos = MDM_LC / 2 + WG_LENGTH / 2
    monitor_size = mp.Vector3(WG_WIDTH * 3, 0)
    
    norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=src_size))
    flux_out = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(0, monitor_y_pos), size=monitor_size), direction=mp.Y)
    
    dft_monitor = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(), size=mp.Vector3(SX, SY))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, monitor_y_pos), 1e-5))
    
    ez_dft_data = sim.get_dft_array(dft_monitor, mp.Ez, 0)
    eps_data = sim.get_epsilon()
    intensity = np.abs(ez_dft_data)**2
    
    res_input = sim.get_eigenmode_coefficients(norm_flux, [1], eig_parity=mp.EVEN_Y)
    input_power = np.abs(res_input.alpha[0, 0, 0])**2 + 1e-12
    refl_power = np.abs(res_input.alpha[0, 0, 1])**2
    
    res_out = sim.get_eigenmode_coefficients(flux_out, [1])
    output_power = np.abs(res_out.alpha[0, 0, 0])**2
    
    trans = output_power / input_power
    refl = refl_power / input_power
    loss = 1.0 - trans - refl
    
    trans_db = 10 * np.log10(trans + 1e-9)
    
    print(f"  [Result] Transmission: {trans:.4f} ({trans_db:.2f} dB)")
    print(f"  [Result] Reflection:   {refl:.4f}")
    print(f"  [Result] Loss/Scatter: {loss:.4f}")

    # Plotting Fix: Use Transpose (.T) and origin='lower'
    x = np.linspace(-SX/2, SX/2, intensity.shape[0])
    y = np.linspace(-SY/2, SY/2, intensity.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # [修正] 使用 .T (轉置) 配合 origin='lower'，而不是 np.rot90
    im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   cmap='jet', origin='lower')
    
    # [修正] 結構輪廓同樣處理
    ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()], 
               levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
    
    fig.colorbar(im, ax=ax, label='Intensity |Ez|^2')
    ax.set_title(f"TE0 90-Degree Bend\nT = {trans:.2%} ({trans_db:.1f} dB)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "final_analysis_TE0_field.png"), dpi=150)
    plt.close(fig)

    detailed_results = {
        "Transmission": float(trans),
        "Transmission_dB": float(trans_db),
        "Reflection": float(refl),
        "Loss": float(loss)
    }
    return detailed_results

def generate_gds(binary_vector, grid_rows, grid_cols, output_folder):
    # GDS generation placeholder
    pass 

def plot_optimization_trajectory(foms, split_index, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = -foms
    
    plt.scatter(indices, transmissions, s=5, c='blue', alpha=0.5, label='Samples')
    plt.axvline(x=split_index, color='red', linestyle='--', linewidth=2, label='Opt Start')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Transmission (TE0)', fontsize=12)
    plt.title('Optimization Trajectory', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "transmission_trajectory.png"), dpi=150)
    plt.close()

def plot_chain_break_history(history, output_folder):
    if 'chain_break_trends' not in history: return
    avg_breaks = history['chain_break_trends']['avg']
    if not avg_breaks: return
    iterations = np.arange(1, len(avg_breaks) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, avg_breaks, marker='o', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Chain Break Fraction')
    plt.title('D-Wave Chain Breaks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "chain_break_history.png"), dpi=150)
    plt.close()

# ==============================================================================
# SECTION 4: MASTER/WORKER LOGIC
# ==============================================================================

def train_fm_model(model, X_train, Y_train, num_epoch, learning_rate, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    device = next(model.parameters()).device
    X_tensor = torch.from_numpy(X_train).float().to(device)
    
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-8
    Y_scaled = (Y_train - y_mean) / y_std
    Y_tensor = torch.from_numpy(Y_scaled).float().view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    start_time = time.time()
    model.train() 
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
    return model, loss_history, time.time() - start_time, (y_mean, y_std)

def worker_node():
    grid_dims = comm.bcast(None, root=0)
    mp.verbosity(0)
    while True:
        status = MPI.Status()
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG: break
        task_idx, config = data
        fom = evaluate_mdm_fom(config, grid_dims['rows'], grid_dims['cols'])
        comm.send((task_idx, fom), dest=0, tag=WORK_TAG)

def parallel_evaluate(tasks, fom_cache, grid_rows, grid_cols):
    num_tasks = len(tasks)
    results = [None] * num_tasks
    tasks_to_run = []
    for i, config in enumerate(tasks):
        if tuple(config) in fom_cache: results[i] = fom_cache[tuple(config)]
        else: tasks_to_run.append((i, config))

    if not tasks_to_run: return results

    num_to_run = len(tasks_to_run)
    sent_jobs = 0
    jobs_done = 0
    for worker_rank in range(1, min(size, num_to_run + 1)):
        comm.send(tasks_to_run[sent_jobs], dest=worker_rank, tag=WORK_TAG)
        sent_jobs += 1

    while jobs_done < num_to_run:
        status = MPI.Status()
        task_idx, fom = comm.recv(source=MPI.ANY_SOURCE, tag=WORK_TAG, status=status)
        results[task_idx] = fom
        fom_cache[tuple(tasks[task_idx])] = fom
        jobs_done += 1
        if sent_jobs < num_to_run:
            comm.send(tasks_to_run[sent_jobs], dest=status.Get_source(), tag=WORK_TAG)
            sent_jobs += 1
    return results

def master_node():
    parser = argparse.ArgumentParser(description="FMQA Bend Inverse Design")
    parser.add_argument('--name', type=str, default='bend_te0', help='Job Name')
    args = parser.parse_args()
    job_name = args.name

    start_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Master Node Started on {device} (Job: {job_name}) ---")
    print("--- Mode: TE0 90-Degree Bend (Left -> Top) ---")

    PARAMS = {
        'GRID_ROWS': 8, 
        'GRID_COLS': 8,
        'INIT_DATASET_SIZE': 200, 
        'ITERATIONS': 30,            
        'ADDING_NUM': 60,            
        'NUM_EPOCHS': 800,
        'LEARNING_RATE': 1.0e-3,
        'NUM_READS': 1000,
        'K_FACTOR': 8,
        'SAMPLER_TYPE': 'QA'
    }
    NUM_VARS = PARAMS['GRID_ROWS'] * PARAMS['GRID_COLS']
    
    BASE_PATH = os.getcwd() 
    RESULTS_BASE_DIR = os.path.join(BASE_PATH, "Results_Bend")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{job_name}_{timestamp}"
    output_folder = os.path.join(RESULTS_BASE_DIR, run_dir_name)
    os.makedirs(output_folder, exist_ok=True)
    
    comm.bcast({'rows': PARAMS['GRID_ROWS'], 'cols': PARAMS['GRID_COLS']}, root=0)
    
    # 1. Init Dataset
    print(f"--- Generating Initial Dataset ({PARAMS['INIT_DATASET_SIZE']}) ---")
    configs = np.array([
        generate_smooth_random_config(PARAMS['GRID_ROWS'], PARAMS['GRID_COLS']) 
        for _ in range(PARAMS['INIT_DATASET_SIZE'])
    ])
    
    fom_cache = {}
    foms = np.array(parallel_evaluate(configs, fom_cache, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS']))
    for i, c in enumerate(configs): fom_cache[tuple(c)] = foms[i]
    
    best_fom = np.min(foms)
    best_config = configs[np.argmin(foms)]
    print(f"Initial Best Transmission: {-best_fom:.2%}")

    history = {'chain_break_trends': {'avg': [], 'max': []}, 'timing_metrics': {'fm_train_time': [], 'new_data_sim_time': []}}
    
    # 2. Optimization Loop
    for i in range(PARAMS['ITERATIONS']):
        print(f"\n=== Iteration {i+1}/{PARAMS['ITERATIONS']} ===")
        
        # A. Train FM
        model = FactorizationMachine(input_size=NUM_VARS, factorization_size=PARAMS['K_FACTOR']).to(device)
        model, _, t_train, _ = train_fm_model(model, configs, foms, PARAMS['NUM_EPOCHS'], PARAMS['LEARNING_RATE'])
        history['timing_metrics']['fm_train_time'].append(t_train)
        
        # B. Build Q
        bias, h, Q = model.get_bhQ() 
        Q_dict = {(r, c): Q[r, c] for r in range(Q.shape[0]) for c in range(r+1, Q.shape[1]) if Q[r, c] != 0}
        bqm = dimod.BinaryQuadraticModel(h, Q_dict, bias, dimod.BINARY)
        
        # C. Sample
        sampleset = None
        if PARAMS['SAMPLER_TYPE'] == "QA" and DWAVE_AVAILABLE:
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], label=f"{job_name}_{i}")
                if 'chain_break_fraction' in sampleset.record.dtype.names:
                     history['chain_break_trends']['avg'].append(float(np.mean(sampleset.record['chain_break_fraction'])))
            except:
                print("QA failed, using SA")
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'])
        else:
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'])
            
        sampleset = sampleset.aggregate()
        sample_configs = sampleset.record['sample'][np.argsort(sampleset.record['energy'])]
        
        # D. Simulate New
        unique_new = [s for s in sample_configs if tuple(s) not in {tuple(x) for x in configs}]
        new_configs = np.array(unique_new[:PARAMS['ADDING_NUM']])
        
        if new_configs.size > 0:
            t_sim_start = time.time()
            new_foms = parallel_evaluate(new_configs, fom_cache, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'])
            configs = np.vstack([configs, new_configs])
            foms = np.concatenate([foms, new_foms])
            history['timing_metrics']['new_data_sim_time'].append(time.time() - t_sim_start)
            
            curr_min = np.min(foms)
            if curr_min < best_fom:
                best_fom = curr_min
                best_config = configs[np.argmin(foms)]
                print(f"*** New Best Transmission: {-best_fom:.2%} ***")
        else:
            print("No new unique configs found.")

    # 3. Finalize
    for i in range(1, size): comm.send(None, dest=i, tag=STOP_TAG)
    
    print("\n=== Finalizing ===")
    plot_optimization_trajectory(foms, PARAMS['INIT_DATASET_SIZE'], output_folder)
    plot_chain_break_history(history, output_folder)
    final_res = perform_detailed_final_analysis(best_config, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'], output_folder)
    
    result_data = {
        "best_transmission": float(-best_fom),
        "best_config": best_config.tolist(),
        "final_analysis": final_res
    }
    with open(os.path.join(output_folder, "final_result.json"), 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"Done. Results in {output_folder}")

if __name__ == "__main__":
    if size > 1 and rank != 0: worker_node()
    else: master_node()