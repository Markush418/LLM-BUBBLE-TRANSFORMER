"""
Attention Maps Visualization: L9 vs L10 vs Softmax
====================================================
Creates publication-quality figures comparing attention patterns.
Output: PDF vectorial + PNG, 'hot' colormap, all 16 heads, includes softmax baseline.
"""

import os, sys, json
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from qwen3_focus_bubble_wrapper import Qwen3FocusBubbleWrapper

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
WINDOW = 256
EPSILON = 0.001
TAU_ITERS = 1

OUT_DIR = Path("results_real/attention_viz_l9_l10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_model_and_tokenizer():
    """Load Qwen3-0.6B model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def prepare_input(tokenizer):
    """Prepare test input (same as verification scripts)."""
    text = "Hello world this is a test " * 20
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :WINDOW].cuda()
    return input_ids


def extract_softmax_attention(model, input_ids):
    """Extract softmax attention from all layers."""
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    # outputs.attentions is tuple of [B, H, N, N] per layer
    attns = {}
    for i, attn in enumerate(outputs.attentions):
        attns[i] = attn.float().cpu().numpy()  # [B, H, N, N]
    return attns


def extract_focus_attention(model, tokenizer, input_ids, layer_idx, epsilon=EPSILON, tau_iters=TAU_ITERS, use_delta=False, lam=0.0):
    """Extract Focus Bubble attention for a specific layer."""
    orig_attn = model.model.layers[layer_idx].self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn,
        epsilon=epsilon,
        tau_iters=tau_iters,
        use_psi=True,
        use_delta=use_delta,
        lam=lam,
    ).cuda()
    model.model.layers[layer_idx].self_attn = wrapper

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Restore original
    model.model.layers[layer_idx].self_attn = orig_attn
    
    # Extract attention for the swapped layer
    attn = outputs.attentions[layer_idx].float().cpu().numpy()
    return attn  # [B, H, N, N]


def compute_attention_metrics(attn_head):
    """Compute metrics for a single head attention matrix [N, N]."""
    # Entropy (peakedness)
    eps = 1e-10
    entropy = -np.sum(attn_head * np.log(attn_head + eps), axis=-1).mean()
    
    # Diagonal mass ratio
    N = attn_head.shape[0]
    diag_mask = np.eye(N, dtype=bool)
    diag_mass = attn_head[diag_mask].sum()
    offdiag_mass = attn_head[~diag_mask].sum()
    diag_ratio = diag_mass / (diag_mass + offdiag_mass + 1e-10)
    
    # Concentration ratio (fraction > 1e-3)
    concentration = (attn_head > 1e-3).mean()
    
    return {
        'entropy': entropy,
        'diag_ratio': diag_ratio,
        'concentration': concentration,
    }


def plot_figure_1a_avg_heatmap(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1A: Average attention heatmap (softmax vs focus)"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Average across heads and batch
    softmax_avg = softmax_attns[layer_idx].mean(axis=(0, 1))  # [N, N]
    focus_avg = focus_attns[layer_idx].mean(axis=(0, 1))
    
    vmax = max(softmax_avg.max(), focus_avg.max())
    
    im1 = axes[0].imshow(softmax_avg, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Softmax L{layer_idx} (avg 16 heads)', fontweight='bold')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(focus_avg, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Focus Bubble L{layer_idx} (avg 16 heads)', fontweight='bold')
    axes[1].set_xlabel('Key position')
    axes[1].set_ylabel('Query position')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Figure 1A: Average Attention Heatmap (L{layer_idx})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig1a_avg_heatmap_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1a_avg_heatmap_L{layer_idx}.png', dpi=300)
    plt.close()


def plot_figure_1b_per_head_diff(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1B: Per-head L2 difference bar plot"""
    softmax_l = softmax_attns[layer_idx][0]  # [H, N, N]
    focus_l = focus_attns[layer_idx][0]
    
    H = softmax_l.shape[0]
    l2_diffs = []
    l2_ratios = []
    
    for h in range(H):
        s = softmax_l[h]
        f = focus_l[h]
        l2_diff = np.linalg.norm(f - s)
        l2_base = np.linalg.norm(s)
        l2_diffs.append(l2_diff)
        l2_ratios.append(l2_diff / (l2_base + 1e-10) * 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    heads = np.arange(16)
    bars1 = ax1.bar(heads, l2_diffs, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Head')
    ax1.set_ylabel('L2 Difference')
    ax1.set_title('L2 Difference (Focus - Softmax) per Head')
    ax1.set_xticks(heads)
    ax1.grid(axis='y', alpha=0.3)
    
    bars2 = ax2.bar(heads, l2_ratios, color='coral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Head')
    ax2.set_ylabel('L2 Ratio (%)')
    ax2.set_title('L2 Ratio % per Head')
    ax2.set_xticks(heads)
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight outlier heads (> mean + 1.5*std)
    mean_r = np.mean(l2_ratios)
    std_r = np.std(l2_ratios)
    threshold = mean_r + 1.5 * std_r
    for i, ratio in enumerate(l2_ratios):
        if ratio > threshold:
            ax2.bar(i, ratio, color='red', edgecolor='black', linewidth=1.5)
    
    plt.suptitle(f'Figure 1B: Per-Head L2 Difference (Focus vs Softmax)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig1b_per_head_diff_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1b_per_head_diff_L{layer_idx}.png', dpi=300)
    plt.close()
    
    return l2_ratios


def plot_figure_1c_all_heads_heatmaps(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1C: All 16 heads heatmaps (4x4 grid)"""
    softmax_l = softmax_attns[layer_idx][0]  # [H, N, N]
    focus_l = focus_attns[layer_idx][0]
    
    H = softmax_l.shape[0]
    ncols = 4
    nrows = 4
    
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(16, 14))
    
    vmax = max(softmax_l.max(), focus_l.max())
    
    for h in range(H):
        row = h // ncols
        col = (h % ncols) * 2
        
        # Softmax
        im1 = axes[row, col].imshow(softmax_l[h], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[row, col].set_title(f'Softmax H{h}', fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Focus
        im2 = axes[row, col+1].imshow(focus_l[h], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[row, col+1].set_title(f'Focus H{h}', fontsize=9)
        axes[row, col+1].set_xticks([])
        axes[row, col+1].set_yticks([])
    
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    
    plt.suptitle(f'Figure 1C: All 16 Heads Heatmaps - Softmax (left) vs Focus (right) L{layer_idx}', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(out_dir / f'fig1c_all_heads_heatmaps_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1c_all_heads_heatmaps_L{layer_idx}.png', dpi=300)
    plt.close()


def plot_figure_1d_diag_offdiag(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1D: Diagonal vs Off-diagonal mass scatter"""
    softmax_l = softmax_attns[layer_idx][0]  # [H, N, N]
    focus_l = focus_attns[layer_idx][0]
    
    H = softmax_l.shape[0]
    softmax_diag = []
    softmax_offdiag = []
    focus_diag = []
    focus_offdiag = []
    
    for h in range(H):
        s = softmax_l[h]
        f = focus_l[h]
        N = s.shape[0]
        diag_mask = np.eye(N, dtype=bool)
        
        softmax_diag.append(s[diag_mask].sum())
        softmax_offdiag.append(s[~diag_mask].sum())
        focus_diag.append(f[diag_mask].sum())
        focus_offdiag.append(f[~diag_mask].sum())
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    ax.scatter(softmax_diag, softmax_offdiag, c='steelblue', s=80, alpha=0.7, label='Softmax', edgecolor='black', linewidth=0.5)
    ax.scatter(focus_diag, focus_offdiag, c='coral', s=80, alpha=0.7, label='Focus Bubble', edgecolor='black', linewidth=0.5)
    
    # Add head labels
    for h in range(H):
        ax.annotate(f'H{h}', (softmax_diag[h], softmax_offdiag[h]), fontsize=7, alpha=0.6)
        ax.annotate(f'H{h}', (focus_diag[h], focus_offdiag[h]), fontsize=7, alpha=0.6)
    
    ax.set_xlabel('Diagonal Mass')
    ax.set_ylabel('Off-diagonal Mass')
    ax.set_title(f'Figure 1D: Diagonal vs Off-diagonal Mass (L{layer_idx})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(out_dir / f'fig1d_diag_offdiag_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1d_diag_offdiag_L{layer_idx}.png', dpi=300)
    plt.close()


def plot_figure_1e_entropy_boxplot(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1E: Entropy per head boxplot"""
    softmax_l = softmax_attns[layer_idx][0]  # [H, N, N]
    focus_l = focus_attns[layer_idx][0]
    
    H = softmax_l.shape[0]
    softmax_entropies = []
    focus_entropies = []
    softmax_diag_ratios = []
    focus_diag_ratios = []
    softmax_concentrations = []
    focus_concentrations = []
    
    for h in range(H):
        s_metrics = compute_attention_metrics(softmax_l[h])
        f_metrics = compute_attention_metrics(focus_l[h])
        
        softmax_entropies.append(s_metrics['entropy'])
        focus_entropies.append(f_metrics['entropy'])
        softmax_diag_ratios.append(s_metrics['diag_ratio'])
        focus_diag_ratios.append(f_metrics['diag_ratio'])
        softmax_concentrations.append(s_metrics['concentration'])
        focus_concentrations.append(f_metrics['concentration'])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    data_entropy = [softmax_entropies, focus_entropies]
    data_diag = [softmax_diag_ratios, focus_diag_ratios]
    data_conc = [softmax_concentrations, focus_concentrations]
    
    labels = ['Softmax', 'Focus Bubble']
    
    for ax, data, title in zip(axes, [data_entropy, data_diag, data_conc], 
                                ['Entropy (Peakedness)', 'Diagonal Mass Ratio', 'Concentration Ratio (>1e-3)']):
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=0.3)
        # Add individual points
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.04, size=len(d))
            ax.scatter(x, d, c=colors[i], alpha=0.5, s=30, edgecolor='black', linewidth=0.3)
    
    plt.suptitle(f'Figure 1E: Per-Head Metrics Distribution (L{layer_idx})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig1e_entropy_boxplot_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1e_entropy_boxplot_L{layer_idx}.png', dpi=300)
    plt.close()


def plot_figure_1f_softmax_vs_focus_delta(softmax_attns, focus_attns, layer_idx, out_dir):
    """Fig 1F: Difference heatmap (Focus - Softmax) average"""
    softmax_avg = softmax_attns[layer_idx].mean(axis=(0, 1))
    focus_avg = focus_attns[layer_idx].mean(axis=(0, 1))
    
    diff = focus_avg - softmax_avg
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    vmax = max(softmax_avg.max(), focus_avg.max())
    
    im1 = axes[0].imshow(softmax_avg, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0].set_title('Softmax (avg)', fontweight='bold')
    axes[0].set_xlabel('Key'); axes[0].set_ylabel('Query')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(focus_avg, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[1].set_title('Focus Bubble (avg)', fontweight='bold')
    axes[1].set_xlabel('Key'); axes[1].set_ylabel('Query')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference with RdBu_r (diverging)
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    axes[2].set_title('Focus - Softmax (difference)', fontweight='bold')
    axes[2].set_xlabel('Key'); axes[2].set_ylabel('Query')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Figure 1F: Softmax vs Focus Bubble vs Difference (L{layer_idx})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f'fig1f_diff_heatmap_L{layer_idx}.pdf')
    plt.savefig(out_dir / f'fig1f_diff_heatmap_L{layer_idx}.png', dpi=300)
    plt.close()


def save_metrics_csv(softmax_attns, focus_attns, layer_idx, out_dir):
    """Save per-head metrics to CSV for supplementary table."""
    softmax_l = softmax_attns[layer_idx][0]
    focus_l = focus_attns[layer_idx][0]
    H = softmax_l.shape[0]
    
    rows = []
    for h in range(H):
        s_metrics = compute_attention_metrics(softmax_l[h])
        f_metrics = compute_attention_metrics(focus_l[h])
        l2_diff = np.linalg.norm(focus_l[h] - softmax_l[h])
        l2_ratio = l2_diff / (np.linalg.norm(softmax_l[h]) + 1e-10) * 100
        
        rows.append({
            'layer': layer_idx,
            'head': h,
            'softmax_entropy': s_metrics['entropy'],
            'focus_entropy': f_metrics['entropy'],
            'softmax_diag_ratio': s_metrics['diag_ratio'],
            'focus_diag_ratio': f_metrics['diag_ratio'],
            'softmax_concentration': s_metrics['concentration'],
            'focus_concentration': f_metrics['concentration'],
            'l2_diff': l2_diff,
            'l2_ratio_pct': l2_ratio,
        })
    
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'metrics_summary_L{layer_idx}.csv', index=False)
    return df


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("=" * 70)
    print("  ATTENTION MAPS VISUALIZATION: L9 vs L10 vs Softmax")
    print("=" * 70)
    
    # Load model
    print("\n[1/5] Loading model...")
    tokenizer, model = load_model_and_tokenizer()
    input_ids = prepare_input(tokenizer)
    print(f"  Input shape: {input_ids.shape}")
    
    # Extract softmax attention (baseline)
    print("\n[2/5] Extracting softmax attention...")
    softmax_attns = extract_softmax_attention(model, input_ids)
    
    # Extract Focus attention for L9 and L10
    print("\n[3/5] Extracting Focus attention (L9)...")
    focus_l9 = extract_focus_attention(model, tokenizer, input_ids, 9, use_delta=False, lam=0.0)
    
    print("\n[4/5] Extracting Focus attention (L10)...")
    focus_l10 = extract_focus_attention(model, tokenizer, input_ids, 10, use_delta=False, lam=0.0)
    
    focus_attns = {9: focus_l9, 10: focus_l10}
    
    # Generate figures for both layers
    print("\n[5/5] Generating figures...")
    for layer_idx in [9, 10]:
        print(f"  Processing L{layer_idx}...")
        focus_attn = focus_attns[layer_idx]
        
        plot_figure_1a_avg_heatmap(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        l2_ratios = plot_figure_1b_per_head_diff(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        plot_figure_1c_all_heads_heatmaps(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        plot_figure_1d_diag_offdiag(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        plot_figure_1e_entropy_boxplot(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        plot_figure_1f_softmax_vs_focus_delta(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        df = save_metrics_csv(softmax_attns, {layer_idx: focus_attn}, layer_idx, OUT_DIR)
        
        print(f"  L{layer_idx} metrics summary:")
        print(f"    Mean L2 ratio: {df['l2_ratio_pct'].mean():.1f}%")
        print(f"    Mean entropy (Softmax/Focus): {df['softmax_entropy'].mean():.3f} / {df['focus_entropy'].mean():.3f}")
        print(f"    Mean diag ratio (Softmax/Focus): {df['softmax_diag_ratio'].mean():.3f} / {df['focus_diag_ratio'].mean():.3f}")
        print(f"    Mean concentration (Softmax/Focus): {df['softmax_concentration'].mean():.3f} / {df['focus_concentration'].mean():.3f}")
    
    print(f"\n{'='*70}")
    print(f"  ALL FIGURES SAVED TO: {OUT_DIR}")
    print(f"{'='*70}")
    for f in sorted(OUT_DIR.glob('*')):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()