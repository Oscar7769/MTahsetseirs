import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter

# MPI & PyTorch
from mpi4py import MPI
import torch
import torch.distributions as dist  # 【升級重點 1】引入官方分佈庫計算 KL
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# D-Wave / Dimod (For Annealing)
import dimod
from dwave.samplers import SimulatedAnnealingSampler
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

# 載入您的 Factorization Machine 模型
import sys
import os
# 自動定位到當前檔案的上一層(專案根目錄)，並加入 shared_modules 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.abspath(os.path.join(current_dir, "..", "shared_modules"))

if shared_path not in sys.path:
    sys.path.append(shared_path)

from factorization_machine import FactorizationMachine

# MEEP
import meep as mp
from shared_modules.meep_utils import (
    evaluate_fom_from_geometry,
    run_bend_simulation,
    RESOLUTION, DPML, MDM_LC, WG_LENGTH, WG_WIDTH,
    SX, SY, CELL, N_SIO2, N_SI, SIO2_MEDIUM, SI_MEDIUM,
    WL_CEN, FCEN, DF, NFREQ
)

# ==============================================================================
# 0. 全域設定與參數
# ==============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

WORK_TAG = 1
STOP_TAG = 2

# 【升級重點 2】智能硬體指派：Master 用 GPU 訓練，Worker 用 CPU 跑 MEEP
if rank == 0 and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- BVAE 參數 ---
IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE
HIDDEN_LAYERS_VAE = [512, 256]
LATENT_N = 100     # 10 * 10 = 100
LATENT_K = 2     
BVAE_BATCH_SIZE = 64
BVAE_LR = 1e-3
BVAE_EPOCHS = 30
BVAE_DATA_SAMPLES = 10000 

# --- MEEP 參數 (彎曲波導: 左進 -> 上出) — 已從 shared_modules.meep_utils 匯入 ---
# RESOLUTION, DPML, MDM_LC, WG_WIDTH, WG_LENGTH, SX, SY, CELL
# N_SIO2, N_SI, SIO2_MEDIUM, SI_MEDIUM, WL_CEN, FCEN, DF, NFREQ

# --- FMQA 優化參數 ---
INIT_SIM_COUNT = 2000  # 初始樣本數
ITERATIONS = 200       # 迭代次數
SAMPLES_PER_ITER = 30
FM_EPOCHS = 1000      
FM_LR = 1e-3
FM_K = 10               
NUM_READS = 2000
NUM_SWEEPS = 1000

# ==============================================================================
# 1. 模型類別定義
# ==============================================================================

class GaussianSmoothing(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1.0):
        super(GaussianSmoothing, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        return F.conv2d(x_padded, self.weight, padding=0, groups=self.channels)

class TanhProjection(nn.Module):
    def __init__(self, beta=50.0, eta=0.5):
        super(TanhProjection, self).__init__()
        self.beta = beta
        self.eta = eta

    def forward(self, x):
        beta_t = torch.tensor(self.beta, device=x.device, dtype=torch.float32)
        eta_t = torch.tensor(self.eta, device=x.device, dtype=torch.float32)
        
        num = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (x - eta_t))
        den = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (1.0 - eta_t))
        return torch.clamp(num / den, 0.0, 1.0)

class CategoricalVAE(nn.Module):
    def __init__(self, input_dim, hidden_layers, N, K, img_size=64):
        super(CategoricalVAE, self).__init__()
        self.N = N 
        self.K = K 
        self.input_dim = input_dim
        self.img_size = img_size
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )
        self.encoder_linear = nn.Linear(64 * 8 * 8, N * K)

        self.init_channels = 32
        self.init_size = 8      
        linear_out_dim = self.init_channels * self.init_size * self.init_size
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(N * K, 512),
            nn.ReLU(),
            nn.Linear(512, linear_out_dim),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.init_channels, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Sigmoid() 
        )
        
        self.gaussian_smooth = GaussianSmoothing(channels=1, kernel_size=5, sigma=1.0)
        self.tanh_project = TanhProjection(beta=30.0, eta=0.5)

    def forward(self, x, temperature=1.0, hard=False):
        x_2d = x.view(-1, 1, self.img_size, self.img_size) 
        
        conv_out = self.encoder_conv(x_2d)
        conv_out_flat = conv_out.view(conv_out.size(0), -1)
        
        logits = self.encoder_linear(conv_out_flat)
        logits = logits.view(-1, self.N, self.K)
        
        z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        z_flat = z.view(-1, self.N * self.K)
        
        x_recon = self.decode(z_flat)
        return x_recon, logits, z
    
    def decode(self, z_flat):
        x = self.decoder_linear(z_flat)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.decoder_conv(x)
        x = self.gaussian_smooth(x)
        x = self.tanh_project(x)
        x_recon_flat = x.view(-1, self.input_dim)
        return x_recon_flat

    def decode_from_indices(self, indices):
        batch_size = indices.size(0)
        z = torch.zeros(batch_size, self.N, self.K).to(indices.device)
        z.scatter_(2, indices.unsqueeze(2).long(), 1.0)
        z_flat = z.view(-1, self.N * self.K)
        return self.decode(z_flat)

# ==============================================================================
# 2. 輔助函數與 BVAE 訓練
# ==============================================================================

def cat_kl_div(logits, N, K, device):
    ''' 【升級重點 3】使用 PyTorch distributions 穩定計算 KL 散度 '''
    batch_size = logits.size(0)
    logits_flat = logits.view(batch_size * N, K)
    q = dist.Categorical(logits=logits_flat)
    p = dist.Categorical(probs=torch.full((batch_size * N, K), 1.0 / K, device=device))
    kl = dist.kl.kl_divergence(q, p)
    return torch.mean(kl)

def generate_wg_dataset(num_samples, img_size=64, feature_size=4.0):
    data = []
    if rank == 0:
        print(f"[Master] Generating {num_samples} waveguide samples...")
        for _ in range(num_samples):
            noise = np.random.normal(0, 1, (img_size, img_size))
            sigma = np.random.uniform(feature_size - 1.0, feature_size + 1.0)
            smooth = gaussian_filter(noise, sigma=sigma)
            smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
            binary = (smooth > 0).astype(np.float32)
            data.append(binary.flatten())
    return np.array(data, dtype=np.float32)

def train_bvae(model, data_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=BVAE_LR)
    model.train()
    
    init_temp = 1.0
    anneal_rate = 0.0005
    min_temp = 0.1
    global_step = 0
    beta_kl = 0.01  # 【升級重點 4】配合 mean reduction 的 Beta 權重
    
    print(f"[Master] Training BVAE on {DEVICE}...")
    for epoch in range(epochs):
        total_loss, total_recon, total_kl = 0, 0, 0
        for batch_data in data_loader:
            batch_data = batch_data[0].to(DEVICE)
            temp = np.maximum(init_temp * np.exp(-anneal_rate * global_step), min_temp)
            
            optimizer.zero_grad()
            recon, logits, _ = model(batch_data, temperature=temp)
            
            # Reconstruction loss (使用 mean 計算)
            recon_loss = F.binary_cross_entropy(recon, batch_data, reduction='mean')
            # KL loss
            kl_loss = cat_kl_div(logits, model.N, model.K, DEVICE)
            
            loss = recon_loss + beta_kl * kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            global_step += 1
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            n_batches = len(data_loader)
            print(f"  [BVAE] Epoch {epoch+1:02d}/{epochs}, "
                  f"Loss: {total_loss/n_batches:.4f}, "
                  f"Recon: {total_recon/n_batches:.4f}, "
                  f"KL: {total_kl/n_batches:.4f}, "
                  f"Temp: {temp:.4f}")
    return model

def train_fm(model, X, y, epochs):
    optimizer = optim.Adam(model.parameters(), lr=FM_LR)
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            optimizer.step()
    return model

# ==============================================================================
# 3. MEEP 幾何與模擬 (僅在 Worker CPU 執行)
# ==============================================================================

def get_meep_geometry(design_grid_64x64):
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

def simulate_structure(latent_vector_np, bvae_decoder):
    """
    bVAE 專屬包裝函式：
      1. 透過 bVAE decoder 將 latent index 向量解碼為 64x64 二值結構圖像。
      2. 呼叫 get_meep_geometry 將圖像轉換為 MEEP geometry。
      3. 呼叫 shared_modules.meep_utils.run_bend_simulation 執行 FDTD 模擬。
    回傳 transmission (float, 0~1)。
    """
    with torch.no_grad():
        idx_tensor = torch.tensor(latent_vector_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        recon_img = bvae_decoder.decode_from_indices(idx_tensor)
        img_grid = recon_img.view(IMG_SIZE, IMG_SIZE).cpu().numpy()
        binary_grid = (img_grid > 0.5).astype(np.float64)

    mp.verbosity(0)
    try:
        geometry = get_meep_geometry(binary_grid)
        transmission = run_bend_simulation(geometry)  # → shared_modules.meep_utils
    except Exception as e:
        print(f"[Worker {rank}] MEEP Error: {e}")
        transmission = 0.0

    return transmission

# ==============================================================================
# 4. Phase 3: Final Analysis & Plotting
# ==============================================================================

def final_analysis_plot(best_latent, bvae_decoder, output_dir, fom_history):
    print("[Phase 3] Running Final High-Res Analysis...")
    with torch.no_grad():
        idx_tensor = torch.tensor(best_latent, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        img_grid = bvae_decoder.decode_from_indices(idx_tensor).view(IMG_SIZE, IMG_SIZE).cpu().numpy()
        binary_grid = (img_grid > 0.5).astype(np.float64)

    np.save(os.path.join(output_dir, "best_structure_64x64.npy"), binary_grid)
    
    latent_dim_size = int(np.sqrt(LATENT_N))
    latent_2d = np.array(best_latent).reshape(latent_dim_size, latent_dim_size)
    np.save(os.path.join(output_dir, f"best_latent_{latent_dim_size}x{latent_dim_size}.npy"), latent_2d)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(latent_2d, cmap='gray', interpolation='none')
    plt.title(f"Best Latent Representation ({latent_dim_size}x{latent_dim_size})")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"best_latent_{latent_dim_size}x{latent_dim_size}.png"), dpi=300)
    plt.close()

    geometry = get_meep_geometry(binary_grid)
    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF), center=mp.Vector3(-SX / 2 + DPML + 0.2, 0), size=mp.Vector3(0, WG_WIDTH), direction=mp.X, eig_band=1, eig_parity=mp.EVEN_Y)]
    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], geometry=geometry, sources=sources, resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
    dft_obj = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(0,0), size=mp.Vector3(SX, SY))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, MDM_LC/2 + WG_LENGTH/2), 1e-4))
    
    eps_data = sim.get_epsilon()
    ez_data = sim.get_dft_array(dft_obj, mp.Ez, 0)

    plt.figure(figsize=(8, 8))
    im = plt.imshow(np.abs(ez_data).T, interpolation='spline36', cmap='inferno', origin='lower', extent=[-SX/2, SX/2, -SY/2, SY/2])
    plt.contour(eps_data.T, levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.3, extent=[-SX/2, SX/2, -SY/2, SY/2])
    colorbar = plt.colorbar(im)
    colorbar.set_label('Intensity |Ez|^2')
    plt.title("Optimized Structure Field (Ez)")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.savefig(os.path.join(output_dir, "final_field.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best FOM (Transmission)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

def plot_optimization_trajectory(foms, split_index, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = foms 
    
    plt.scatter(indices[:split_index], transmissions[:split_index], s=5, c='red', alpha=0.6, label='Initial Random Samples')
    if len(foms) > split_index:
        plt.scatter(indices[split_index:], transmissions[split_index:], s=5, c='blue', alpha=0.6, label='FMQA Optimized Samples')
    
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
# 5. 主流程
# ==============================================================================

def worker_process():
    bvae_state = comm.bcast(None, root=0)
    bvae = CategoricalVAE(INPUT_DIM, HIDDEN_LAYERS_VAE, LATENT_N, LATENT_K).to(DEVICE)
    bvae.load_state_dict(bvae_state)
    bvae.eval()
    
    while True:
        status = MPI.Status()
        task_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == STOP_TAG:
            break
        
        if tag == WORK_TAG:
            task_idx, latent_vec = task_data
            try:
                transmission = simulate_structure(latent_vec, bvae)
            except Exception as e:
                print(f"[Worker {rank}] Simulation Error: {e}")
                transmission = 0.0
            comm.send((task_idx, transmission), dest=0, tag=WORK_TAG)

def master_process():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"bVAE_GU_FMQA_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    time_records = {
        "bvae_train_time_sec": 0, "initial_sampling_time_sec": 0,
        "total_fm_train_time_sec": 0, "loop_total_time_sec": 0, "best_fom": 0.0
    }

    # === Phase 1: BVAE ===
    raw_data = generate_wg_dataset(BVAE_DATA_SAMPLES, IMG_SIZE)
    dataset = TensorDataset(torch.tensor(raw_data))
    loader = DataLoader(dataset, batch_size=BVAE_BATCH_SIZE, shuffle=True)
    
    bvae = CategoricalVAE(INPUT_DIM, HIDDEN_LAYERS_VAE, LATENT_N, LATENT_K).to(DEVICE)
    
    t_start_bvae = time.time()
    bvae = train_bvae(bvae, loader, BVAE_EPOCHS)
    time_records["bvae_train_time_sec"] = time.time() - t_start_bvae
    print(f"[Timer] BVAE Training took {time_records['bvae_train_time_sec']:.2f}s")
    
    print("[Master] Broadcasting BVAE model to Workers (via CPU)...")
    # 將模型移至 CPU 再廣播，確保 Worker CPU 環境不報錯
    bvae.to("cpu")
    bvae_state = bvae.state_dict()
    comm.bcast(bvae_state, root=0)
    bvae.to(DEVICE) # Master 放回 GPU 繼續用
    
    t_start_init = time.time()
    print(f"[Master] Generating initial population ({INIT_SIM_COUNT})...")
    X_latent = np.random.randint(0, 2, size=(INIT_SIM_COUNT, LATENT_N))
    y_fom = parallel_evaluate(X_latent)
    time_records["initial_sampling_time_sec"] = time.time() - t_start_init
    
    current_best = np.max(y_fom)
    print(f"[Phase 1] Best Initial T: {current_best:.4f}")

    # === Phase 2: Loop ===
    best_history = [current_best]
    history_set = set(map(tuple, X_latent))
    t_start_loop = time.time()
    
    fm_model = FactorizationMachine(LATENT_N, FM_K).to(DEVICE)
    
    for i in range(ITERATIONS):
        print(f"\n--- Iteration {i+1}/{ITERATIONS} ---")
        
        t_start_fm = time.time()
        train_fm(fm_model, X_latent, y_fom, FM_EPOCHS)
        time_records["total_fm_train_time_sec"] += (time.time() - t_start_fm)
        
        bias, h, Q = fm_model.get_bhQ()
        
        # 【升級重點 5】安全剝離 Tensor 轉為純 numpy Float，防止 BQM 崩潰
        h_np = h.detach().cpu().numpy() if torch.is_tensor(h) else np.array(h)
        Q_np = Q.detach().cpu().numpy() if torch.is_tensor(Q) else np.array(Q)
        bias_val = float(bias.detach().cpu().numpy()) if torch.is_tensor(bias) else float(bias)
        
        linear_terms = {idx: -float(h_np[idx]) for idx in range(LATENT_N)}
        quadratic_terms = {(r, c): -float(Q_np[r, c]) for r in range(LATENT_N) for c in range(r+1, LATENT_N)}
        
        bqm = dimod.BinaryQuadraticModel(linear_terms, quadratic_terms, -bias_val, dimod.BINARY)
        sampler = SimulatedAnnealingSampler()

        candidates = []
        candidate_set = set()
        attempts = 0
        max_attempts = 10
        
        while len(candidates) < SAMPLES_PER_ITER and attempts < max_attempts:
            sampleset = sampler.sample(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS)
            for sample in sampleset.data(['sample'], sorted_by='energy'):
                vec_tuple = tuple(int(sample.sample[k]) for k in range(LATENT_N))
                if vec_tuple not in history_set and vec_tuple not in candidate_set:
                    candidate_set.add(vec_tuple)
                    candidates.append(np.array(vec_tuple))
                if len(candidates) >= SAMPLES_PER_ITER:
                    break
            attempts += 1
            if len(candidates) < SAMPLES_PER_ITER:
                print(f"  [SA] Attempt {attempts}: Only found {len(candidates)}/{SAMPLES_PER_ITER} unique samples. Retrying...")

        if len(candidates) < SAMPLES_PER_ITER:
            missing_count = SAMPLES_PER_ITER - len(candidates)
            print(f"  [Warning] SA reached max attempts. Filling remaining {missing_count} spots with random samples...")
            while len(candidates) < SAMPLES_PER_ITER:
                vec = np.random.randint(0, 2, size=LATENT_N)
                vec_tuple = tuple(vec)
                if vec_tuple not in history_set and vec_tuple not in candidate_set:
                    candidate_set.add(vec_tuple)
                    candidates.append(vec)
        
        candidates = np.array(candidates)
        print(f"  Evaluating {len(candidates)} new candidates...")
        new_foms = parallel_evaluate(candidates)
        
        X_latent = np.vstack([X_latent, candidates])
        y_fom = np.concatenate([y_fom, new_foms])
        history_set.update(map(tuple, candidates))
        
        current_best = np.max(y_fom)
        best_history.append(current_best)
        print(f"  Added {len(candidates)} samples. Global Best T: {current_best:.4f}")

    time_records["loop_total_time_sec"] = time.time() - t_start_loop
    time_records["best_fom"] = float(np.max(y_fom))

    # === Phase 3: Final ===
    for w in range(1, size):
        comm.send(None, dest=w, tag=STOP_TAG)
        
    best_idx = np.argmax(y_fom)
    best_latent = X_latent[best_idx]
    
    final_analysis_plot(best_latent, bvae, out_dir, best_history)
    plot_optimization_trajectory(y_fom, INIT_SIM_COUNT, out_dir)
    
    experiment_log = {
        "timestamp": timestamp,
        "hyperparameters": {
            "IMG_SIZE": IMG_SIZE, "HIDDEN_LAYERS_VAE": HIDDEN_LAYERS_VAE,
            "LATENT_N": LATENT_N, "LATENT_K": LATENT_K, "BVAE_BATCH_SIZE": BVAE_BATCH_SIZE,
            "BVAE_LR": BVAE_LR, "BVAE_EPOCHS": BVAE_EPOCHS, "BVAE_DATA_SAMPLES": BVAE_DATA_SAMPLES,
            "RESOLUTION": RESOLUTION, "MDM_LC": MDM_LC, "WG_LENGTH": WG_LENGTH, "WG_WIDTH": WG_WIDTH,
            "N_SIO2": N_SIO2, "N_SI": N_SI, "WL_CEN": WL_CEN, "FCEN": FCEN, "DF": DF, "NFREQ": NFREQ,
            "INIT_SIM_COUNT": INIT_SIM_COUNT, "ITERATIONS": ITERATIONS, "SAMPLES_PER_ITER": SAMPLES_PER_ITER,
            "FM_EPOCHS": FM_EPOCHS, "FM_LR": FM_LR, "FM_K": FM_K, "NUM_READS": NUM_READS, "NUM_SWEEPS": NUM_SWEEPS
        },
        "time_records": time_records,
        "results": {
            "best_fom": time_records["best_fom"],
            "best_latent_flat": best_latent.tolist(), 
            "fom_evolution_history": best_history,
            "all_evaluated_foms": y_fom.tolist() 
        }
    }
    
    json_path = os.path.join(out_dir, "training_log.json")
    with open(json_path, 'w') as f:
        json.dump(experiment_log, f, indent=4)
        
    print(f"Done. Log saved to {json_path}")

def parallel_evaluate(latent_vectors):
    num_tasks = len(latent_vectors)
    results = np.zeros(num_tasks)
    
    if size == 1:
        print("Warning: Running in serial mode. Please use mpirun.")
        return np.zeros(num_tasks) 

    tasks_sent = 0
    tasks_done = 0
    
    for w in range(1, size):
        if tasks_sent < num_tasks:
            comm.send((tasks_sent, latent_vectors[tasks_sent]), dest=w, tag=WORK_TAG)
            tasks_sent += 1
            
    while tasks_done < num_tasks:
        status = MPI.Status()
        idx, fom = comm.recv(source=MPI.ANY_SOURCE, tag=WORK_TAG, status=status)
        sender = status.Get_source()
        results[idx] = fom
        tasks_done += 1
        
        if tasks_sent < num_tasks:
            comm.send((tasks_sent, latent_vectors[tasks_sent]), dest=sender, tag=WORK_TAG)
            tasks_sent += 1
            
    return results

if __name__ == "__main__":
    if size > 1:
        if rank == 0:
            master_process()
        else:
            worker_process()
    else:
        print("Please run with: mpirun -np 4 python script.py")