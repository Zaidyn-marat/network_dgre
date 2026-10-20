import pandas as pd
import numpy as np
import os
import shutil
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================
METADATA_INPUT = 'hunt24.csv'           # Input metadata from Hunt et al. (2024)
MEMBER_DATA_DIR = 'hunt24'             # Directory containing 7000+ member CSVs
FILTERED_DATA_DIR = 'tau_le1_samples'  # Directory for filtered samples
MST_RESULTS_FILE = 'MST_Results_Raw.csv'
FINAL_COMBINED_DATA = 'NEW_data_for_fig4_plot.csv'

# Physical Constants & Parameters
G_CONST = 0.00449                      # pc^3 / M_sun / Myr^2
N_MASSIVE_LIST = [5, 10, 15, 20]       # N for Mass Segregation Ratio
MC_TRIALS = 200                        # Monte Carlo iterations
PROB_LIMIT = 0.8                       # Membership probability threshold
SIZE_RANGE = (25, 1500)                # Cluster size filtering (star count)

# Plotting Configuration
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12})
if not os.path.exists('figures'): os.makedirs('figures')
if not os.path.exists(FILTERED_DATA_DIR): os.makedirs(FILTERED_DATA_DIR)

# =============================================================================
# 2. STAGE 1: DYNAMICAL FILTERING
# =============================================================================
def stage_1_filtering():
    print("\n[Stage 1] Loading metadata and calculating dynamical age (tau)...")
    try:
        df = pd.read_csv(METADATA_INPUT)
        # Standardize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]
    except Exception as e:
        print(f"Error loading metadata: {e}"); return None

    # Relaxation time calculation (t_rh)
    def get_trh(row):
        n, r_h, m = row.get('n', 0), row.get('r50pc', 0) * 1.33, row.get('masstot', 0)
        if n <= 10 or r_h <= 0 or m <= 0: return np.nan
        ln_lambda = np.log(0.4 * n)
        return (0.138 * n / ln_lambda) * np.sqrt(r_h**3 / (G_CONST * m))

    df['t_rh'] = df.apply(get_trh, axis=1)
    df['age_myr'] = 10**(df['logage50']) / 1e6
    df['tau'] = df['age_myr'] / df['t_rh']

    # Filter: Open Clusters ('o') and Dynamically Young (tau <= 1)
    mask = (df['type'].astype(str).str.strip() == 'o') & (df['tau'] <= 1.0) & (df['tau'] > 0)
    filtered_df = df[mask].copy()
    
    # Process member file copying
    if os.path.exists(MEMBER_DATA_DIR):
        source_files = {f.lower().replace(' ', ''): f for f in os.listdir(MEMBER_DATA_DIR)}
        copied = 0
        for name in filtered_df['name']:
            target = f"{str(name).strip().lower().replace(' ', '')}.csv"
            if target in source_files:
                shutil.copy2(os.path.join(MEMBER_DATA_DIR, source_files[target]), 
                             os.path.join(FILTERED_DATA_DIR, source_files[target]))
                copied += 1
    
    print(f"- Filtering complete: {len(filtered_df)} clusters selected (tau <= 1)")
    filtered_df.to_csv('metadata_filtered.csv', index=False)
    return filtered_df

# =============================================================================
# 3. STAGE 2: MST ANALYSIS
# =============================================================================
def stage_2_mst(meta_df):
    print("\n[Stage 2] Running MST Engine (3D Bayesian vs 2D Projected)...")
    # Standardize join key for robust matching
    meta_df['join_key'] = meta_df['name'].astype(str).str.lower().str.replace(' ', '')
    hunt_lookup = meta_df.set_index('join_key').to_dict('index')

    def get_mst_len(pts):
        if len(pts) < 2: return np.nan
        return nx.minimum_spanning_tree(nx.from_numpy_array(squareform(pdist(pts)))).size(weight='weight')

    def get_coords(df, info, mode_3d=True):
        p_c, s_c = info['plx'], max(info.get('e_plx', 0.05), 0.01)
        ra, dec = np.radians(df['radeg']), np.radians(df['dedeg'])
        if mode_3d:
            # 3D Bayesian distance reconstruction
            plx_corr = (df['plx'] * s_c**2 + p_c * df['e_plx']**2) / (s_c**2 + df['e_plx']**2)
            d = 1000.0 / plx_corr
            return np.vstack([d*np.cos(dec)*np.cos(ra), d*np.cos(dec)*np.sin(ra), d*np.sin(dec)]).T
        else:
            # 2D Projected coordinates
            d = 1000.0 / p_c
            return np.vstack([d*np.cos(dec)*np.cos(ra), d*np.cos(dec)*np.sin(ra)]).T

    results = []
    member_files = [f for f in os.listdir(FILTERED_DATA_DIR) if f.endswith('.csv')]

    for f in tqdm(member_files, desc="Calculating MSR"):
        key = f.replace('.csv', '').lower().replace(' ', '')
        if key not in hunt_lookup: continue
        
        try:
            m_df = pd.read_csv(os.path.join(FILTERED_DATA_DIR, f))
            m_df.columns = [c.lower().strip() for c in m_df.columns]
            m_df = m_df[m_df['prob'] >= PROB_LIMIT].dropna(subset=['mass50','radeg','dedeg','plx','e_plx'])
            if not (SIZE_RANGE[0] <= len(m_df) <= SIZE_RANGE[1]): continue

            info = hunt_lookup[key]
            res_row = {'Name': info['name']} # Key for Stage 3 merging
            
            mass_vals = m_df['mass50'].values
            idx_sorted = np.argsort(mass_vals)
            c3d = get_coords(m_df, info, True)
            c2d = get_coords(m_df, info, False)

            for n in N_MASSIVE_LIST:
                if len(m_df) < n + 2: continue
                # 3D MSR
                l_m3 = get_mst_len(c3d[idx_sorted[-n:]])
                l_r3 = [get_mst_len(c3d[np.random.choice(len(m_df), n, replace=False)]) for _ in range(MC_TRIALS)]
                # 2D MSR
                l_m2 = get_mst_len(c2d[idx_sorted[-n:]])
                l_r2 = [get_mst_len(c2d[np.random.choice(len(m_df), n, replace=False)]) for _ in range(MC_TRIALS)]
                
                res_row.update({
                    f'Lambda3D_{n}': np.mean(l_r3)/l_m3, f'Sig3D_{n}': int((np.mean(l_r3)+np.std(l_r3))/l_m3 < 1),
                    f'Lambda2D_{n}': np.mean(l_r2)/l_m2, f'Sig2D_{n}': int((np.mean(l_r2)+np.std(l_r2))/l_m2 < 1)
                })
            results.append(res_row)
        except: continue

    mst_df = pd.DataFrame(results)
    mst_df.to_csv(MST_RESULTS_FILE, index=False)
    return mst_df

# =============================================================================
# 4. STAGE 3 & 4: DATA MERGING & VISUALIZATION
# =============================================================================
def run_stage_3_4(mst_df, meta_df):
    print("\n[Stage 3] Merging MST results with dynamical parameters...")
    
    # Explicitly join Left 'Name' with Right 'name'
    final_df = pd.merge(
        mst_df, 
        meta_df[['name', 'age_myr', 't_rh', 'tau']], 
        left_on='Name', 
        right_on='name', 
        how='left'
    )
    
    # Diagnostics
    valid_count = final_df['tau'].notnull().sum()
    print(f"- Merge Diagnostics: {len(final_df)} records total, {valid_count} matched with Tau data.")
    
    final_df.to_csv(FINAL_COMBINED_DATA, index=False)

    print("\n[Stage 4] Generating Publication-Quality Figures...")
    
    # Figure 1: Lambda_MSR Distribution
    plt.figure(figsize=(6, 5))
    plt.hist(final_df['Lambda3D_5'], bins=35, range=(0, 2), alpha=0.5, label='3D Bayesian', color='blue', density=True)
    plt.hist(final_df['Lambda2D_5'], bins=35, range=(0, 2), alpha=0.5, label='2D Projected', color='orange', density=True)
    plt.axvline(1.0, color='k', ls='--')
    plt.xlabel(r'$\Lambda_{MSR}$ (N=5)'); plt.ylabel('Normalized Frequency'); plt.legend(); plt.tight_layout()
    plt.savefig('figures/Figure1_Lambda_Dist.pdf')

    # Figure 2: Inverse Fraction vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_MASSIVE_LIST, [final_df[f'Sig3D_{n}'].mean() for n in N_MASSIVE_LIST], 'o-', label='3D Bayesian', color='blue')
    plt.plot(N_MASSIVE_LIST, [final_df[f'Sig2D_{n}'].mean() for n in N_MASSIVE_LIST], 's--', label='2D Projected', color='orange')
    plt.xlabel('Number of Massive Stars (N)'); plt.ylabel('Inverse Fraction'); plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('figures/Figure2_Sig_vs_N.pdf')

    # Figure 3: 3D vs 2D Scatter Comparison
    plt.figure(figsize=(6, 5))
    plt.scatter(final_df['Lambda3D_5'], final_df['Lambda2D_5'], s=10, alpha=0.4, color='teal')
    plt.plot([0, 3], [0, 3], 'r--', alpha=0.5)
    plt.xlabel(r'$\Lambda_{MSR, 3D}$ (N=5)'); plt.ylabel(r'$\Lambda_{MSR, 2D}$ (N=5)'); plt.tight_layout()
    plt.savefig('figures/Figure3_Comparison.pdf')

    # Figure 4: Evolution with Dynamical Age (tau)
    plot_data = final_df.dropna(subset=['tau', 'Lambda3D_5'])
    if not plot_data.empty:
        plt.figure(figsize=(6, 5))
        log_tau = np.log10(plot_data['tau'])
        plt.scatter(log_tau, plot_data['Lambda3D_5'], s=8, color='gray', alpha=0.3, label='Individual Clusters')
        
        # Median Trend Line
        bins = np.linspace(log_tau.min(), log_tau.max(), 8)
        bc = (bins[:-1] + bins[1:]) / 2
        mv = [plot_data[(log_tau >= bins[i]) & (log_tau < bins[i+1])]['Lambda3D_5'].median() for i in range(len(bins)-1)]
        plt.plot(bc, mv, 'r-o', lw=2, label='Median Trend')
        
        plt.axvline(0, color='gray', ls=':', alpha=0.5) # Tau = 1 marker at log10(tau)=0
        plt.axhline(1.0, color='k', ls='--')
        plt.xlabel(r'$\log_{10}(\tau)$'); plt.ylabel(r'$\Lambda_{MSR}$ (3D, N=5)'); plt.legend(); plt.tight_layout()
        plt.savefig('figures/Figure4_Evolution.pdf')
        print("- Figure 4 generated successfully.")
    else:
        print("- Figure 4 skipped: Insufficient data pairs.")

    print(f"\n[Pipeline Finished] Results saved to {FINAL_COMBINED_DATA} and figures/ directory.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()
    # Execute stages
    meta_filtered = stage_1_filtering()
    if meta_filtered is not None:
        mst_raw = stage_2_mst(meta_filtered)
        run_stage_3_4(mst_raw, meta_filtered)
    
    print(f"\nTotal execution time: {(time.time() - start_time)/60:.2f} minutes.")