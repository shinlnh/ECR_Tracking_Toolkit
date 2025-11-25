#!/usr/bin/env python3
"""
Plot OTB100 Evaluation Results
Visualize CSRT baseline performance metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Parse results file
def parse_results(file_path):
    """Parse OTB100 results from text file"""
    sequences = []
    frames = []
    iou_scores = []
    success_rates = []
    fps_values = []
    
    with open(file_path, 'r', encoding='utf-16-le', errors='ignore') as f:
        for line in f:
            # Match sequence result lines (multiple spaces between columns)
            match = re.match(r'^(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if match:
                seq_name, frame_count, iou, success, fps = match.groups()
                # Skip Overall and duplicate Woman entries
                if seq_name == 'Overall':
                    continue
                if seq_name == 'Woman' and float(iou) == 0.0:
                    continue
                    
                sequences.append(seq_name)
                frames.append(int(frame_count))
                iou_scores.append(float(iou))
                success_rates.append(float(success))
                fps_values.append(float(fps))
    
    return sequences, frames, iou_scores, success_rates, fps_values


def plot_iou_distribution(iou_scores, sequences):
    """Plot IoU score distribution histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(iou_scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(iou_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(iou_scores):.3f}')
    plt.axvline(np.median(iou_scores), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(iou_scores):.3f}')
    plt.xlabel('Average IoU', fontsize=12)
    plt.ylabel('Number of Sequences', fontsize=12)
    plt.title('CSRT Baseline - IoU Distribution on OTB100', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_success_vs_iou(iou_scores, success_rates, sequences):
    """Scatter plot: Success Rate vs IoU"""
    plt.figure(figsize=(10, 6))
    plt.scatter(iou_scores, success_rates, c=iou_scores, cmap='RdYlGn', 
                s=50, alpha=0.6, edgecolors='black')
    plt.colorbar(label='IoU Score')
    
    # Add diagonal reference line
    max_val = max(max(iou_scores), max(success_rates))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='IoU = Success')
    
    plt.xlabel('Average IoU', fontsize=12)
    plt.ylabel('Success Rate @0.50', fontsize=12)
    plt.title('CSRT Baseline - Success Rate vs IoU', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_top_bottom_sequences(sequences, iou_scores, n=10):
    """Bar plot of top and bottom performing sequences"""
    # Sort by IoU
    sorted_indices = np.argsort(iou_scores)
    
    # Get top and bottom n sequences
    top_indices = sorted_indices[-n:][::-1]  # Reverse for descending
    bottom_indices = sorted_indices[:n]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top performers
    top_seqs = [sequences[i] for i in top_indices]
    top_ious = [iou_scores[i] for i in top_indices]
    colors_top = ['green' if iou >= 0.7 else 'yellowgreen' for iou in top_ious]
    
    ax1.barh(range(n), top_ious, color=colors_top, edgecolor='black')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(top_seqs)
    ax1.set_xlabel('Average IoU', fontsize=12)
    ax1.set_title(f'Top {n} Sequences', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_ious):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Bottom performers
    bottom_seqs = [sequences[i] for i in bottom_indices]
    bottom_ious = [iou_scores[i] for i in bottom_indices]
    colors_bottom = ['red' if iou < 0.2 else 'orange' for iou in bottom_ious]
    
    ax2.barh(range(n), bottom_ious, color=colors_bottom, edgecolor='black')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(bottom_seqs)
    ax2.set_xlabel('Average IoU', fontsize=12)
    ax2.set_title(f'Bottom {n} Sequences', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(bottom_ious):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_fps_vs_frames(frames, fps_values, sequences):
    """Scatter plot: FPS vs Number of Frames"""
    plt.figure(figsize=(10, 6))
    plt.scatter(frames, fps_values, c=fps_values, cmap='plasma', 
                s=50, alpha=0.6, edgecolors='black')
    plt.colorbar(label='FPS')
    
    plt.xlabel('Number of Frames', fontsize=12)
    plt.ylabel('Average FPS', fontsize=12)
    plt.title('CSRT Baseline - Processing Speed vs Sequence Length', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add mean FPS line
    mean_fps = np.mean(fps_values)
    plt.axhline(mean_fps, color='red', linestyle='--', linewidth=2, 
                label=f'Mean FPS: {mean_fps:.2f}')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def plot_performance_categories(iou_scores, success_rates):
    """Pie chart: Performance categories"""
    # Categorize sequences
    excellent = sum(1 for iou in iou_scores if iou >= 0.7)
    good = sum(1 for iou in iou_scores if 0.5 <= iou < 0.7)
    moderate = sum(1 for iou in iou_scores if 0.3 <= iou < 0.5)
    poor = sum(1 for iou in iou_scores if iou < 0.3)
    
    categories = ['Excellent\n(IoU ≥ 0.7)', 'Good\n(0.5 ≤ IoU < 0.7)', 
                  'Moderate\n(0.3 ≤ IoU < 0.5)', 'Poor\n(IoU < 0.3)']
    counts = [excellent, good, moderate, poor]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoU categories
    ax1.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 11})
    ax1.set_title('Performance Distribution by IoU', fontsize=14, fontweight='bold')
    
    # Success rate categories
    high_success = sum(1 for sr in success_rates if sr >= 0.7)
    medium_success = sum(1 for sr in success_rates if 0.4 <= sr < 0.7)
    low_success = sum(1 for sr in success_rates if sr < 0.4)
    
    success_categories = ['High\n(≥ 70%)', 'Medium\n(40-70%)', 'Low\n(< 40%)']
    success_counts = [high_success, medium_success, low_success]
    success_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax2.pie(success_counts, labels=success_categories, autopct='%1.1f%%', 
            startangle=90, colors=success_colors, textprops={'fontsize': 11})
    ax2.set_title('Success Rate Distribution (@0.50 threshold)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def generate_summary_stats(sequences, frames, iou_scores, success_rates, fps_values):
    """Generate summary statistics text"""
    stats = f"""
╔════════════════════════════════════════════════════════════════╗
║           OTB100 CSRT BASELINE - SUMMARY STATISTICS            ║
╚════════════════════════════════════════════════════════════════╝

Total Sequences:        {len(sequences)}
Total Frames:           {sum(frames):,}

─────────────────────── IoU Metrics ────────────────────────────
Mean IoU:               {np.mean(iou_scores):.3f}
Median IoU:             {np.median(iou_scores):.3f}
Std Dev:                {np.std(iou_scores):.3f}
Min IoU:                {np.min(iou_scores):.3f} ({sequences[np.argmin(iou_scores)]})
Max IoU:                {np.max(iou_scores):.3f} ({sequences[np.argmax(iou_scores)]})

─────────────────── Success Rate Metrics ───────────────────────
Mean Success:           {np.mean(success_rates):.3f}
Median Success:         {np.median(success_rates):.3f}
Sequences ≥ 0.7:        {sum(1 for sr in success_rates if sr >= 0.7)} ({100*sum(1 for sr in success_rates if sr >= 0.7)/len(sequences):.1f}%)

──────────────────────── FPS Metrics ───────────────────────────
Mean FPS:               {np.mean(fps_values):.2f}
Median FPS:             {np.median(fps_values):.2f}
Min FPS:                {np.min(fps_values):.2f} ({sequences[np.argmin(fps_values)]})
Max FPS:                {np.max(fps_values):.2f} ({sequences[np.argmax(fps_values)]})

════════════════════════════════════════════════════════════════
"""
    return stats


def main():
    # Parse results
    results_file = Path(__file__).parent / 'otb100_csrt_results.txt'
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    print("Parsing results...")
    sequences, frames, iou_scores, success_rates, fps_values = parse_results(results_file)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Print summary statistics
    stats = generate_summary_stats(sequences, frames, iou_scores, success_rates, fps_values)
    print(stats)
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / 'otb100_plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Generate all plots
    print("  1. IoU Distribution...")
    fig1 = plot_iou_distribution(iou_scores, sequences)
    fig1.savefig(output_dir / '1_iou_distribution.png', dpi=150, bbox_inches='tight')
    
    print("  2. Success vs IoU...")
    fig2 = plot_success_vs_iou(iou_scores, success_rates, sequences)
    fig2.savefig(output_dir / '2_success_vs_iou.png', dpi=150, bbox_inches='tight')
    
    print("  3. Top/Bottom Sequences...")
    fig3 = plot_top_bottom_sequences(sequences, iou_scores, n=10)
    fig3.savefig(output_dir / '3_top_bottom_sequences.png', dpi=150, bbox_inches='tight')
    
    print("  4. FPS vs Frames...")
    fig4 = plot_fps_vs_frames(frames, fps_values, sequences)
    fig4.savefig(output_dir / '4_fps_vs_frames.png', dpi=150, bbox_inches='tight')
    
    print("  5. Performance Categories...")
    fig5 = plot_performance_categories(iou_scores, success_rates)
    fig5.savefig(output_dir / '5_performance_categories.png', dpi=150, bbox_inches='tight')
    
    # Save summary stats to file
    stats_file = output_dir / 'summary_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats)
    
    print(f"\n✅ Complete! Plots saved to: {output_dir}")
    print(f"   - 5 visualization plots (.png)")
    print(f"   - Summary statistics (summary_statistics.txt)")
    
    # Show all plots
    print("\nDisplaying plots...")
    plt.show()


if __name__ == '__main__':
    main()
