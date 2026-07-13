"""
FMQA (SA) Inverse Design — 90° Bend Waveguide (10×10 Grid)
===========================================================
方法:  Factorization Machine + Simulated Annealing (FMQA-SA)
目標:  設計 SOI 90° 彎曲波導，左側輸入 → 上方輸出，最大化 TE0 穿透率
編碼:  Gaussian Upsampling (GU)
       - 10×10 binary grid (100 variables) → Gaussian Kernel 平滑 → Tanh Projection 二值化
       - 透過 MEEP MaterialGrid 映射到 4µm × 4µm 設計區域

策略:
  - FM Warm-Start: 模型在迴圈外初始化一次，跨迭代持續訓練
  - SA num_sweeps=1000: 增加 SA 探索深度
  - NFREQ=1: 修正為只監測中心波長 (λ=1.55µm)，避免取到錯誤頻率
  - smoothing_radius=1.0: 使用 MEEP 預設子像素平滑
  - Auto-fill Fallback: SA 樣本不足時自動補齊隨機 config
"""

import os
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
NFREQ = 1

KERNEL_SIZE = 5
KERNEL_SIGMA = 1.0
TANH_BETA = 50
TANH_ETA = 0.5
SMOOTH_THRESHOLD = 0.5

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

def get_projected_density_matrix(binary_vector, grid_rows, grid_cols):
    grid_matrix = np.array(binary_vector).astype(float).reshape((grid_rows, grid_cols))
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

def generate_smooth_random_config(rows, cols):
    small_r, small_c = max(1, rows // 2), max(1, cols // 2) 
    noise = np.random.rand(small_r, small_c)
    smooth_noise = scipy.ndimage.zoom(noise, zoom=2, order=1) 
    smooth_noise = smooth_noise[:rows, :cols]
    binary = (smooth_noise > SMOOTH_THRESHOLD).astype(int).flatten()
    return binary

# ==============================================================================
# SECTION 2: OPTIMIZATION SIMULATION
# ==============================================================================

def evaluate_mdm_fom(binary_vector, grid_rows, grid_cols):
    mp.Simulation(cell_size=CELL, resolution=1, boundary_layers=[]).reset_meep()
    mdm_structure = create_projected_geometry(binary_vector, grid_rows, grid_cols)
    
    input_wg_center_x = -MDM_LC / 2 - (WG_LENGTH + DPML) / 2
    input_wg_center_y = 0
    output_wg_center_x = 0
    output_wg_center_y = MDM_LC / 2 + (WG_LENGTH + DPML) / 2
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), 
                 center=mp.Vector3(input_wg_center_x, input_wg_center_y), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), 
                 center=mp.Vector3(output_wg_center_x, output_wg_center_y), material=SI_MEDIUM),
    ]
    
    full_geometry = fixed_geometry + mdm_structure
    
    src_center = mp.Vector3(-SX / 2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH) 
    monitor_y_pos = MDM_LC / 2 + WG_LENGTH / 2
    monitor_size = mp.Vector3(WG_WIDTH * 3, 0) 

    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), 
                                  center=src_center, size=src_size, direction=mp.X, 
                                  eig_band=1, eig_parity=mp.EVEN_Y)]
    
    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], 
                        geometry=full_geometry, sources=sources, 
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
    norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, 
                                     mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=src_size))
    flux_out = sim.add_mode_monitor(FCEN, DF, NFREQ, 
                                    mp.ModeRegion(center=mp.Vector3(0, monitor_y_pos), size=monitor_size), 
                                    direction=mp.Y) 

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, monitor_y_pos), 1e-5))
    
    input_flux = sim.get_eigenmode_coefficients(norm_flux, [1], eig_parity=mp.EVEN_Y).alpha[0, 0, 0]
    input_power = np.abs(input_flux)**2 + 1e-12
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
    
    mdm_structure = create_projected_geometry(best_config, grid_rows, grid_cols)
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
    src_size = mp.Vector3(0, WG_WIDTH)
    
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

    x = np.linspace(-SX/2, SX/2, intensity.shape[0])
    y = np.linspace(-SY/2, SY/2, intensity.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   cmap='inferno', origin='lower')
    ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()], 
               levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
    fig.colorbar(im, ax=ax, label='Intensity |Ez|^2')
    ax.set_title(f"Optimized Final Structure", fontsize=16)
    ax.set_xlabel("x (um)", fontsize=14)
    ax.set_ylabel("y (um)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Optimized_Final_Structure.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 修正 2: 精準擷取 MEEP 中央 Design Region 的介電係數，轉換為 Binary 結構
    # --------------------------------------------------------------------------
    x_mask = (x >= -MDM_LC/2) & (x <= MDM_LC/2)
    y_mask = (y >= -MDM_LC/2) & (y <= MDM_LC/2)
    eps_design = eps_data[np.ix_(x_mask, y_mask)]
    
    threshold_eps = (N_SI**2 + N_SIO2**2) / 2
    binary_design = (eps_design > threshold_eps).astype(int)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # 透過 eps_data 直接抓取，並使用 .T 與 origin='lower' 確保與物理場圖 100% 重合
    ax.imshow(binary_design.T, extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2], 
              cmap='gray_r', origin='lower')
    ax.set_title("Smoothed Binary Structure", fontsize=16)
    ax.set_xlabel("x (um)", fontsize=14)
    ax.set_ylabel("y (um)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Smoothed_Binary_Structure.png"), dpi=300)
    plt.close(fig)
    # --------------------------------------------------------------------------

    return {
        "Transmission": float(trans),
        "Transmission_dB": float(trans_db),
        "Reflection": float(refl),
        "Loss": float(loss)
    }

def plot_fom_history(fom_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Best FOM (Transmission)", fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

def plot_optimization_trajectory(foms, split_index, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = np.abs(foms) 
    
    plt.scatter(indices[:split_index], transmissions[:split_index], 
                s=5, c='red', alpha=0.6, label='Initial Random Samples')
                
    if len(foms) > split_index:
        plt.scatter(indices[split_index:], transmissions[split_index:], 
                    s=5, c='blue', alpha=0.6, label='FMQA Optimized Samples')
    
    plt.axvline(x=split_index, color='red', linestyle='--', linewidth=2, label='Opt Start')
    
    plt.xlabel('Sample Index (Total Evaluations)', fontsize=14)
    plt.ylabel('Transmission (TE0)', fontsize=14)
    plt.title('Optimization Trajectory', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "transmission_trajectory.png"), dpi=150)
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
    
    print(f"  [FM Training] Dataset Size: {len(X_train)} samples")
    
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
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    - Epoch [{epoch+1:3d}/{num_epoch}], Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
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
    PARAMS = {
        'GRID_ROWS': 10,
        'GRID_COLS': 10,
        'INIT_DATASET_SIZE': 1000,
        'ITERATIONS': 100,
        'ADDING_NUM': 30,
        'NUM_EPOCHS': 1000,
        'LEARNING_RATE': 1.0e-3,
        'NUM_READS': 1500,
        'K_FACTOR': 8,
        'SAMPLER_TYPE': 'SA'
    }
    parser = argparse.ArgumentParser(description="FMQA Bend Inverse Design")
    parser.add_argument('--name', type=str, default=f'GU_{PARAMS["INIT_DATASET_SIZE"]}_{PARAMS["ITERATIONS"]}x{PARAMS["ADDING_NUM"]}', help='Job Name')
    args = parser.parse_args()
    job_name = args.name
    total_meep_time = 0.0
    start_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Master Node Started on {device} (Job: {job_name}) ---")

    
    NUM_VARS = PARAMS['GRID_ROWS'] * PARAMS['GRID_COLS']
    
    BASE_PATH = os.getcwd() 
    RESULTS_BASE_DIR = os.path.join(BASE_PATH, "Results_Bend")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{job_name}_{timestamp}"
    output_folder = os.path.join(RESULTS_BASE_DIR, run_dir_name)
    os.makedirs(output_folder, exist_ok=True)
    
    comm.bcast({'rows': PARAMS['GRID_ROWS'], 'cols': PARAMS['GRID_COLS']}, root=0)
    
    print(f"--- Generating Initial Dataset ({PARAMS['INIT_DATASET_SIZE']}) ---")
    configs = np.array([
        generate_smooth_random_config(PARAMS['GRID_ROWS'], PARAMS['GRID_COLS']) 
        for _ in range(PARAMS['INIT_DATASET_SIZE'])
    ])
    
    fom_cache = {}
    t_init_meep_start = time.time()
    foms = np.array(parallel_evaluate(configs, fom_cache, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS']))
    for i, c in enumerate(configs): fom_cache[tuple(c)] = foms[i]
    
    best_fom = np.min(foms)
    best_config = configs[np.argmin(foms)]
    print(f"Initial Best Transmission: {-best_fom:.2%}")

    history = {'chain_break_trends': {'avg': [], 'max': []}, 'timing_metrics': {'fm_train_time': [], 'new_data_sim_time': []}}
    best_fom_history = [-best_fom] 

    # Initialize FM model ONCE before the loop (warm-start)
    model = FactorizationMachine(input_size=NUM_VARS, factorization_size=PARAMS['K_FACTOR']).to(device)

    for i in range(PARAMS['ITERATIONS']):
        print(f"\n=== Iteration {i+1}/{PARAMS['ITERATIONS']} ===")
        
        # A. Train FM (warm-start: reuse model from previous iteration)
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
                sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system4.1'))
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], label=f"{job_name}_{i}")
                if 'chain_break_fraction' in sampleset.record.dtype.names:
                     history['chain_break_trends']['avg'].append(float(np.mean(sampleset.record['chain_break_fraction'])))
            except:
                print("QA failed, using SA")
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=1000)
        else:
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=1000)
            
        sampleset = sampleset.aggregate()
        sample_configs = sampleset.record['sample'][np.argsort(sampleset.record['energy'])]
        
        # --------------------------------------------------------------------------
        # 修正 3: 新增自動補齊機制 (Auto-fill Fallback) 確保補足 ADDING_NUM
        # --------------------------------------------------------------------------
        unique_new = [s for s in sample_configs if tuple(s) not in {tuple(x) for x in configs}]
        new_configs_list = unique_new[:PARAMS['ADDING_NUM']]
        
        shortfall = PARAMS['ADDING_NUM'] - len(new_configs_list)
        if shortfall > 0:
            print(f"  [Notice] Sampler only yielded {len(new_configs_list)} unique configs. Auto-filling {shortfall} random configs.")
            existing_set = {tuple(x) for x in configs} | {tuple(x) for x in new_configs_list}
            attempts = 0
            while len(new_configs_list) < PARAMS['ADDING_NUM'] and attempts < shortfall * 10:
                rand_c = generate_smooth_random_config(PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'])
                if tuple(rand_c) not in existing_set:
                    new_configs_list.append(rand_c)
                    existing_set.add(tuple(rand_c))
                attempts += 1
                
        new_configs = np.array(new_configs_list)
        # --------------------------------------------------------------------------
        
        if new_configs.size > 0:
            num_new_samples = len(new_configs)
            print(f"  Evaluating {num_new_samples} new candidates...")
            
            t_sim_start = time.time()
            new_foms = parallel_evaluate(new_configs, fom_cache, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'])
            sim_duration = time.time() - t_sim_start
            total_meep_time += sim_duration
            configs = np.vstack([configs, new_configs])
            foms = np.concatenate([foms, new_foms])
            history['timing_metrics']['new_data_sim_time'].append(time.time() - t_sim_start)
            
            curr_min = np.min(foms)
            if curr_min < best_fom:
                best_fom = curr_min
                best_config = configs[np.argmin(foms)]
                print(f"  *** Breakthrough! New Best Transmission: {-best_fom:.2%} ***")
                
            print(f"  Added {num_new_samples} samples. Global Best T: {-best_fom:.4f}")
        else:
            print("  No new unique configs found. Skipping simulation.")
            
        best_fom_history.append(-best_fom)

    for i in range(1, size): comm.send(None, dest=i, tag=STOP_TAG)
    
    print("\n=== Finalizing ===")
    
    plot_fom_history(best_fom_history, output_folder)
    plot_optimization_trajectory(foms, PARAMS['INIT_DATASET_SIZE'], output_folder)
    
    # --------------------------------------------------------------------------
    # 修正 1: 儲存與繪製 8x8 Config (單純轉置 .T 即可完美對齊物理場)
    # --------------------------------------------------------------------------
    config_8x8 = np.array(best_config).reshape((PARAMS['GRID_ROWS'], PARAMS['GRID_COLS']))
    
    # 反對角轉置 + 旋轉 180 度 = 單純的轉置 (.T)
    final_config_8x8 = config_8x8.T 
    
    np.save(os.path.join(output_folder, "best_config_8x8.npy"), final_config_8x8)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(final_config_8x8, cmap='gray_r', origin='lower', 
              extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2]) # 保持對齊物理尺度
    
    # 加入網格線 (保持對齊邊緣)
    ax.set_xticks(np.linspace(-MDM_LC/2, MDM_LC/2, PARAMS['GRID_COLS']+1), minor=True)
    ax.set_yticks(np.linspace(-MDM_LC/2, MDM_LC/2, PARAMS['GRID_ROWS']+1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0) 
    
    ax.set_title(f"Best Binary Configuration ({PARAMS['GRID_ROWS']}x{PARAMS['GRID_COLS']})", fontsize=16)
    ax.set_xlabel("X (um)", fontsize=14)
    ax.set_ylabel("Y (um)", fontsize=14)
    plt.savefig(os.path.join(output_folder, f"best_config_{PARAMS['GRID_ROWS']}x{PARAMS['GRID_COLS']}.png"), dpi=300)
    plt.close()
    # --------------------------------------------------------------------------
    final_res = perform_detailed_final_analysis(best_config, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'], output_folder)
    t_final_meep_start = time.time()
    final_res = perform_detailed_final_analysis(best_config, PARAMS['GRID_ROWS'], PARAMS['GRID_COLS'], output_folder)
    total_meep_time += time.time() - t_final_meep_start
    total_run_time = time.time() - start_total
    time_records = {
        "total_time_seconds": total_run_time,
        "total_meep_time_seconds": total_meep_time,  # [新增] 總 MEEP 執行時間
        "avg_fm_train_time": float(np.mean(history['timing_metrics']['fm_train_time'])) if history['timing_metrics']['fm_train_time'] else 0,
        "avg_new_data_sim_time": float(np.mean(history['timing_metrics']['new_data_sim_time'])) if history['timing_metrics']['new_data_sim_time'] else 0
    }
    
    experiment_log = {
        "timestamp": timestamp,
        "hyperparameters": {
            "RESOLUTION": RESOLUTION,
            "MDM_LC": MDM_LC,
            "WG_LENGTH": WG_LENGTH,
            "WG_WIDTH": WG_WIDTH,
            "N_SIO2": N_SIO2,
            "N_SI": N_SI,
            "wl_cen": wl_cen,
            "FCEN": FCEN,
            "DF": DF,
            "NFREQ": NFREQ,
            "gaussian_kernel_size": KERNEL_SIZE,     
            "gaussian_kernel_sigma": KERNEL_SIGMA,   
            "tanh_beta": TANH_BETA,                  
            "tanh_eta": TANH_ETA,                    
            "smooth_threshold": SMOOTH_THRESHOLD,    
            "GRID_ROWS": PARAMS['GRID_ROWS'],
            "GRID_COLS": PARAMS['GRID_COLS'],
            "SAMPLER_TYPE": PARAMS['SAMPLER_TYPE'],
            "INIT_SIM_COUNT": PARAMS['INIT_DATASET_SIZE'],
            "ITERATIONS": PARAMS['ITERATIONS'],
            "SAMPLES_PER_ITER": PARAMS['ADDING_NUM'],
            "FM_EPOCHS": PARAMS['NUM_EPOCHS'],
            "FM_LR": PARAMS['LEARNING_RATE'],
            "FM_K": PARAMS['K_FACTOR'],
            "NUM_READS": PARAMS['NUM_READS']
        },
        "time_records": time_records,
        "results": {
            "best_fom": float(-best_fom), 
            "best_latent_flat": best_config.tolist(), 
            "fom_evolution_history": best_fom_history,
            "all_evaluated_foms": np.abs(foms).tolist(), 
            "final_analysis": final_res
        }
    }
    
    with open(os.path.join(output_folder, "final_result.json"), 'w') as f:
        json.dump(experiment_log, f, indent=4)
        
    print(f"Done. Results in {output_folder}")

if __name__ == "__main__":
    if size > 1 and rank != 0: worker_node()
    else: master_node()