import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage 
import meep as mp

# ==============================================================================
# SECTION 0: GLOBAL PARAMETERS
# ==============================================================================

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
    grid_matrix = np.array(binary_vector).reshape((grid_rows, grid_cols))
    
    pixel_height = MDM_LC / grid_rows
    pixel_width = MDM_LC / grid_cols
    
    pad = 1 

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

# ==============================================================================
# SECTION 2: VISUALIZATION & ANALYSIS
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

    x = np.linspace(-SX/2, SX/2, intensity.shape[0])
    y = np.linspace(-SY/2, SY/2, intensity.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                   cmap='jet', origin='lower')
    
    ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()], 
               levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Intensity $|E_z|^2$', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    
    ax.set_title(f"Bend Waveguide DFT Field Distribution", fontsize=18)
    ax.set_xlabel("x ($\\mu$m)", fontsize=16)
    ax.set_ylabel("y ($\\mu$m)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "eval_TE0_field.png"), dpi=150)
    plt.close(fig)

    # 畫出單純的結構圖 (二值化)
    eps_threshold = (N_SI**2 + N_SIO2**2) / 2
    eps_binary = (eps_data > eps_threshold).astype(int)
    
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    im2 = ax2.imshow(eps_binary.T, extent=[x.min(), x.max(), y.min(), y.max()], 
                     cmap='Greys', origin='lower')
                     
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cb2 = fig2.colorbar(im2, cax=cax2, ticks=[0, 1])
    cb2.set_label('Material (0: SiO2, 1: Si)', fontsize=16)
    cb2.ax.tick_params(labelsize=14)
    
    ax2.set_title("Binarized Device Structure", fontsize=18)
    ax2.set_xlabel("x ($\\mu$m)", fontsize=16)
    ax2.set_ylabel("y ($\\mu$m)", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "eval_TE0_structure_binary.png"), dpi=150)
    plt.close(fig2)

    # 畫出原始 Config 格子圖
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    config_matrix = best_config.reshape((grid_rows, grid_cols))
    
    im3 = ax3.imshow(config_matrix, cmap='Greys', origin='upper')
    
    # 加上網格線
    ax3.set_xticks(np.arange(-.5, grid_cols, 1), minor=True)
    ax3.set_yticks(np.arange(-.5, grid_rows, 1), minor=True)
    ax3.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax3.tick_params(which="minor", size=0)
    
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cb3 = fig3.colorbar(im3, cax=cax3, ticks=[0, 1])
    cb3.set_label('Binary Config', fontsize=16)
    cb3.ax.tick_params(labelsize=14)
    
    ax3.set_title("Best Binary Config", fontsize=18)
    ax3.set_xlabel("Grid X", fontsize=16)
    ax3.set_ylabel("Grid Y", fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "eval_TE0_config_grid.png"), dpi=150)
    plt.close(fig3)

    detailed_results = {
        "Transmission": float(trans),
        "Transmission_dB": float(trans_db),
        "Reflection": float(refl),
        "Loss": float(loss)
    }
    return detailed_results

if __name__ == "__main__":
    # ==============================================================================
    # 使用者自訂區域：請在下方 MY_BEST_CONFIG 陣列中填入你想測試的 0/1 結構
    # (目前預設為 8x8，共 64 個數字)
    # ==============================================================================
    MY_BEST_CONFIG = [
            1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0
    ]
    
    GRID_ROWS = 8
    GRID_COLS = 8
    OUTPUT_FOLDER = "."
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    best_config = np.array(MY_BEST_CONFIG)
    expected_len = GRID_ROWS * GRID_COLS
    if len(best_config) != expected_len:
        raise ValueError(f"Config length is {len(best_config)}, but expected {expected_len} ({GRID_ROWS}x{GRID_COLS})")
        
    print(f"Evaluating config of length {len(best_config)}...")
    results = perform_detailed_final_analysis(best_config, GRID_ROWS, GRID_COLS, OUTPUT_FOLDER)
    
    print("\n===============================")
    print("--- Evaluation Complete ---")
    print(f"Transmission (fom): {results['Transmission']:.4f} ({results['Transmission_dB']:.2f} dB)")
    print(f"Plots saved to:")
    print(f"  - Field Plot:     {os.path.join(OUTPUT_FOLDER, 'eval_TE0_field.png')}")
    print(f"  - Structure Plot: {os.path.join(OUTPUT_FOLDER, 'eval_TE0_structure_binary.png')}")
    print(f"  - Config Grid:    {os.path.join(OUTPUT_FOLDER, 'eval_TE0_config_grid.png')}")
    print("===============================\n")
