"""
Advanced WiFi AP Visualization System
Provides detailed individual AP analysis and comprehensive combined metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from scipy import stats
from sklearn.metrics import silhouette_score
import logging

class AdvancedWiFiVisualizer:
    """Advanced visualization system for WiFi AP placement analysis"""
    
    def __init__(self, building_width: float, building_height: float, resolution: float = 0.2):
        self.building_width = building_width
        self.building_height = building_height
        self.resolution = resolution
        self.setup_style()
        
    def setup_style(self):
        """Setup professional plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    def create_individual_ap_analysis(self, ap_locations: Dict, rssi_grids: List[np.ndarray], 
                                    points: List[Tuple], collector, output_dir: str):
        """Create detailed individual AP analysis plots"""
        logging.info("Creating individual AP analysis plots...")
        
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            if i >= len(rssi_grids):
                continue
                
            # Extract AP information
            x, y = ap_coords[0], ap_coords[1]
            z = ap_coords[2] if len(ap_coords) > 2 else 0
            tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            
            # Create comprehensive individual AP plot
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Signal Coverage Map
            ax1 = plt.subplot(2, 3, 1)
            self._plot_ap_coverage_map(ax1, ap_name, ap_coords, rssi_grids[i], points)
            
            # 2. Signal Strength Distribution
            ax2 = plt.subplot(2, 3, 2)
            self._plot_signal_distribution(ax2, ap_name, rssi_grids[i])
            
            # 3. Coverage Statistics
            ax3 = plt.subplot(2, 3, 3)
            self._plot_coverage_statistics(ax3, ap_name, rssi_grids[i])
            
            # 4. Distance vs Signal Strength
            ax4 = plt.subplot(2, 3, 4)
            self._plot_distance_vs_signal(ax4, ap_name, ap_coords, points, collector)
            
            # 5. Coverage Quality Analysis
            ax5 = plt.subplot(2, 3, 5)
            self._plot_coverage_quality(ax5, ap_name, rssi_grids[i])
            
            # 6. AP Performance Metrics
            ax6 = plt.subplot(2, 3, 6)
            self._plot_performance_metrics(ax6, ap_name, ap_coords, rssi_grids[i])
            
            plt.suptitle(f'Advanced Analysis: {ap_name} (z={z:.1f}m, {tx_power:.0f}dBm)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save individual AP plot
            output_path = os.path.join(output_dir, f'individual_analysis_{ap_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Created individual analysis for {ap_name}")
    
    def _plot_ap_coverage_map(self, ax, ap_name: str, ap_coords: Tuple, rssi_grid: np.ndarray, points: List[Tuple]):
        """Plot detailed coverage map for individual AP"""
        x, y = ap_coords[0], ap_coords[1]
        
        # Create coverage heatmap
        x_coords = np.array([pt[0] for pt in points])
        y_coords = np.array([pt[1] for pt in points])
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)
        
        # Reshape RSSI grid for plotting
        if len(rssi_grid.shape) == 1:
            rssi_grid_2d = rssi_grid.reshape((len(y_unique), len(x_unique)))
        else:
            rssi_grid_2d = rssi_grid
            
        # Plot heatmap
        im = ax.imshow(rssi_grid_2d, extent=[0, self.building_width, 0, self.building_height], 
                      origin='lower', cmap='RdYlBu_r', aspect='auto')
        
        # Add AP location
        ax.scatter(x, y, s=300, c='red', marker='^', edgecolors='black', linewidth=3, zorder=10)
        ax.annotate(ap_name, (x, y), xytext=(10, 10), textcoords='offset points', 
                   fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="white", alpha=0.8))
        
        # Add coverage contours
        levels = [-67, -50, -40, -30]
        colors = ['red', 'orange', 'yellow', 'green']
        for level, color in zip(levels, colors):
            if np.min(rssi_grid_2d) <= level <= np.max(rssi_grid_2d):
                contour = ax.contour(rssi_grid_2d, levels=[level], colors=color, 
                                   linewidths=2, alpha=0.8, linestyles='--')
                ax.clabel(contour, inline=True, fontsize=8, fmt=f'{level} dBm')
        
        ax.set_title(f'{ap_name} Coverage Map')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        plt.colorbar(im, ax=ax, label='Signal Strength (dBm)')
        
    def _plot_signal_distribution(self, ax, ap_name: str, rssi_grid: np.ndarray):
        """Plot signal strength distribution"""
        rssi_values = rssi_grid.flatten()
        
        # Create histogram with KDE
        ax.hist(rssi_values, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(rssi_values)
        x_range = np.linspace(rssi_values.min(), rssi_values.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add statistics
        mean_signal = np.mean(rssi_values)
        std_signal = np.std(rssi_values)
        ax.axvline(mean_signal, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_signal:.1f} dBm')
        ax.axvline(mean_signal + std_signal, color='orange', linestyle=':', linewidth=2,
                  label=f'+1σ: {mean_signal + std_signal:.1f} dBm')
        ax.axvline(mean_signal - std_signal, color='orange', linestyle=':', linewidth=2,
                  label=f'-1σ: {mean_signal - std_signal:.1f} dBm')
        
        ax.set_title(f'{ap_name} Signal Distribution')
        ax.set_xlabel('Signal Strength (dBm)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_coverage_statistics(self, ax, ap_name: str, rssi_grid: np.ndarray):
        """Plot coverage statistics"""
        rssi_values = rssi_grid.flatten()
        
        # Calculate coverage metrics
        excellent_coverage = np.sum(rssi_values >= -40) / len(rssi_values) * 100
        good_coverage = np.sum((rssi_values >= -50) & (rssi_values < -40)) / len(rssi_values) * 100
        acceptable_coverage = np.sum((rssi_values >= -67) & (rssi_values < -50)) / len(rssi_values) * 100
        poor_coverage = np.sum(rssi_values < -67) / len(rssi_values) * 100
        
        # Create stacked bar chart
        categories = ['Excellent\n(≥-40 dBm)', 'Good\n(-50 to -40 dBm)', 
                     'Acceptable\n(-67 to -50 dBm)', 'Poor\n(<-67 dBm)']
        values = [excellent_coverage, good_coverage, acceptable_coverage, poor_coverage]
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{ap_name} Coverage Statistics')
        ax.set_ylabel('Coverage Percentage (%)')
        ax.set_ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
    def _plot_distance_vs_signal(self, ax, ap_name: str, ap_coords: Tuple, points: List[Tuple], collector):
        """Plot distance vs signal strength relationship"""
        x, y = ap_coords[0], ap_coords[1]
        
        distances = []
        signals = []
        
        for pt in points:
            distance = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
            signal = collector.calculate_rssi(distance, None)
            distances.append(distance)
            signals.append(signal)
        
        # Create scatter plot
        ax.scatter(distances, signals, alpha=0.6, s=20, c='blue')
        
        # Add theoretical path loss curve
        max_dist = max(distances)
        dist_range = np.linspace(0, max_dist, 100)
        theoretical_signals = [collector.calculate_rssi(d, None) for d in dist_range]
        ax.plot(dist_range, theoretical_signals, 'r--', linewidth=2, label='Theoretical Path Loss')
        
        # Add coverage thresholds
        ax.axhline(y=-40, color='green', linestyle='-', alpha=0.7, label='Excellent (-40 dBm)')
        ax.axhline(y=-50, color='yellow', linestyle='-', alpha=0.7, label='Good (-50 dBm)')
        ax.axhline(y=-67, color='orange', linestyle='-', alpha=0.7, label='Acceptable (-67 dBm)')
        
        ax.set_title(f'{ap_name} Distance vs Signal Strength')
        ax.set_xlabel('Distance (meters)')
        ax.set_ylabel('Signal Strength (dBm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_coverage_quality(self, ax, ap_name: str, rssi_grid: np.ndarray):
        """Plot coverage quality analysis"""
        rssi_values = rssi_grid.flatten()
        
        # Calculate quality metrics
        mean_signal = np.mean(rssi_values)
        std_signal = np.std(rssi_values)
        min_signal = np.min(rssi_values)
        max_signal = np.max(rssi_values)
        
        # Create radar chart-like visualization
        metrics = ['Mean Signal', 'Signal Stability', 'Coverage Range', 'Quality Score']
        values = [
            (mean_signal + 100) / 100,  # Normalize to 0-1
            1 - (std_signal / 50),      # Lower std is better
            (max_signal - min_signal) / 100,  # Coverage range
            np.sum(rssi_values >= -50) / len(rssi_values)  # Quality score
        ]
        
        # Ensure values are in [0, 1]
        values = [max(0, min(1, v)) for v in values]
        
        # Create bar chart
        bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], 
                     alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{ap_name} Coverage Quality Analysis')
        ax.set_ylabel('Normalized Score (0-1)')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_metrics(self, ax, ap_name: str, ap_coords: Tuple, rssi_grid: np.ndarray):
        """Plot performance metrics dashboard"""
        x, y = ap_coords[0], ap_coords[1]
        z = ap_coords[2] if len(ap_coords) > 2 else 0
        tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
        
        rssi_values = rssi_grid.flatten()
        
        # Calculate performance metrics
        mean_signal = np.mean(rssi_values)
        coverage_area = np.sum(rssi_values >= -67) / len(rssi_values) * 100
        signal_variance = np.var(rssi_values)
        efficiency = (mean_signal + 100) / tx_power  # Signal per dBm of power
        
        # Create metrics display
        metrics_text = f"""
        AP Performance Metrics
        
        Location: ({x:.1f}, {y:.1f}, {z:.1f})
        TX Power: {tx_power:.1f} dBm
        
        Mean Signal: {mean_signal:.1f} dBm
        Coverage Area: {coverage_area:.1f}%
        Signal Variance: {signal_variance:.1f} dB²
        Power Efficiency: {efficiency:.2f} dBm/dBm
        
        Signal Range: {np.min(rssi_values):.1f} to {np.max(rssi_values):.1f} dBm
        Coverage Quality: {'Excellent' if coverage_area > 90 else 'Good' if coverage_area > 70 else 'Fair'}
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="lightblue", alpha=0.8))
        
        ax.set_title(f'{ap_name} Performance Dashboard')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def create_combined_analysis(self, ap_locations: Dict, rssi_grids: List[np.ndarray], 
                               points: List[Tuple], output_dir: str):
        """Create comprehensive combined analysis"""
        logging.info("Creating combined AP analysis...")
        
        # Create large comprehensive plot
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Combined Coverage Heatmap
        ax1 = plt.subplot(3, 4, 1)
        self._plot_combined_coverage_heatmap(ax1, ap_locations, rssi_grids, points)
        
        # 2. AP Performance Comparison
        ax2 = plt.subplot(3, 4, 2)
        self._plot_ap_performance_comparison(ax2, ap_locations, rssi_grids)
        
        # 3. Coverage Overlap Analysis
        ax3 = plt.subplot(3, 4, 3)
        self._plot_coverage_overlap(ax3, ap_locations, rssi_grids)
        
        # 4. Signal Quality Distribution
        ax4 = plt.subplot(3, 4, 4)
        self._plot_combined_signal_quality(ax4, rssi_grids)
        
        # 5. AP Placement Analysis
        ax5 = plt.subplot(3, 4, 5)
        self._plot_ap_placement_analysis(ax5, ap_locations)
        
        # 6. Interference Analysis
        ax6 = plt.subplot(3, 4, 6)
        self._plot_interference_analysis(ax6, ap_locations, rssi_grids)
        
        # 7. Coverage Efficiency
        ax7 = plt.subplot(3, 4, 7)
        self._plot_coverage_efficiency(ax7, ap_locations, rssi_grids)
        
        # 8. Signal Strength Statistics
        ax8 = plt.subplot(3, 4, 8)
        self._plot_signal_statistics(ax8, rssi_grids)
        
        # 9. AP Load Distribution
        ax9 = plt.subplot(3, 4, 9)
        self._plot_ap_load_distribution(ax9, ap_locations, rssi_grids, points)
        
        # 10. Coverage Gaps Analysis
        ax10 = plt.subplot(3, 4, 10)
        self._plot_coverage_gaps(ax10, ap_locations, rssi_grids, points)
        
        # 11. Power Efficiency Analysis
        ax11 = plt.subplot(3, 4, 11)
        self._plot_power_efficiency(ax11, ap_locations, rssi_grids)
        
        # 12. Overall System Metrics
        ax12 = plt.subplot(3, 4, 12)
        self._plot_system_metrics(ax12, ap_locations, rssi_grids)
        
        plt.suptitle('Advanced WiFi AP System Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save combined analysis
        output_path = os.path.join(output_dir, 'advanced_combined_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Created advanced combined analysis")
        
    def _plot_combined_coverage_heatmap(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray], points: List[Tuple]):
        """Plot combined coverage heatmap"""
        # Combine all RSSI grids
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        
        # Create heatmap
        im = ax.imshow(combined_grid, extent=[0, self.building_width, 0, self.building_height], 
                      origin='lower', cmap='RdYlBu_r', aspect='auto')
        
        # Add AP locations
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:len(ap_locations)]
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            x, y = ap_coords[0], ap_coords[1]
            ax.scatter(x, y, s=200, c=[colors[i]], marker='^', edgecolors='black', 
                      linewidth=2, zorder=10, label=ap_name)
            ax.annotate(ap_name, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
        
        ax.set_title('Combined Coverage Heatmap')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        plt.colorbar(im, ax=ax, label='Signal Strength (dBm)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    def _plot_ap_performance_comparison(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot AP performance comparison"""
        ap_names = list(ap_locations.keys())
        mean_signals = []
        coverage_areas = []
        
        for rssi_grid in rssi_grids:
            rssi_values = rssi_grid.flatten()
            mean_signals.append(np.mean(rssi_values))
            coverage_areas.append(np.sum(rssi_values >= -67) / len(rssi_values) * 100)
        
        x = np.arange(len(ap_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mean_signals, width, label='Mean Signal (dBm)', alpha=0.8)
        bars2 = ax.bar(x + width/2, coverage_areas, width, label='Coverage Area (%)', alpha=0.8)
        
        ax.set_title('AP Performance Comparison')
        ax.set_xlabel('Access Points')
        ax.set_ylabel('Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(ap_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
    def _plot_coverage_overlap(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot coverage overlap analysis"""
        # Calculate overlap matrix
        n_aps = len(ap_locations)
        overlap_matrix = np.zeros((n_aps, n_aps))
        
        for i in range(n_aps):
            for j in range(n_aps):
                if i != j:
                    # Calculate overlap between AP i and AP j
                    coverage_i = rssi_grids[i] >= -67
                    coverage_j = rssi_grids[j] >= -67
                    overlap = np.sum(coverage_i & coverage_j) / np.sum(coverage_i | coverage_j)
                    overlap_matrix[i, j] = overlap
        
        # Plot overlap heatmap
        im = ax.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_title('Coverage Overlap Analysis')
        ax.set_xlabel('AP Index')
        ax.set_ylabel('AP Index')
        
        # Add text annotations
        for i in range(n_aps):
            for j in range(n_aps):
                if i != j:
                    text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Overlap Ratio')
        
    def _plot_combined_signal_quality(self, ax, rssi_grids: List[np.ndarray]):
        """Plot combined signal quality distribution"""
        all_signals = []
        for rssi_grid in rssi_grids:
            all_signals.extend(rssi_grid.flatten())
        
        # Create quality categories
        excellent = np.sum(np.array(all_signals) >= -40)
        good = np.sum((np.array(all_signals) >= -50) & (np.array(all_signals) < -40))
        acceptable = np.sum((np.array(all_signals) >= -67) & (np.array(all_signals) < -50))
        poor = np.sum(np.array(all_signals) < -67)
        
        categories = ['Excellent\n(≥-40 dBm)', 'Good\n(-50 to -40 dBm)', 
                     'Acceptable\n(-67 to -50 dBm)', 'Poor\n(<-67 dBm)']
        values = [excellent, good, acceptable, poor]
        colors = ['green', 'yellow', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        ax.set_title('Combined Signal Quality Distribution')
        
    def _plot_ap_placement_analysis(self, ax, ap_locations: Dict):
        """Plot AP placement analysis"""
        x_coords = [ap_coords[0] for ap_coords in ap_locations.values()]
        y_coords = [ap_coords[1] for ap_coords in ap_locations.values()]
        z_coords = [ap_coords[2] if len(ap_coords) > 2 else 0 for ap_coords in ap_locations.values()]
        
        # Create 3D-like visualization
        scatter = ax.scatter(x_coords, y_coords, s=[100 + z*5 for z in z_coords], 
                           c=z_coords, cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Add AP labels
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            ax.annotate(ap_name, (ap_coords[0], ap_coords[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax.set_title('AP Placement Analysis')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_xlim(0, self.building_width)
        ax.set_ylim(0, self.building_height)
        plt.colorbar(scatter, ax=ax, label='Z-coordinate (m)')
        ax.grid(True, alpha=0.3)
        
    def _plot_interference_analysis(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot interference analysis"""
        # Calculate interference at each point
        interference_levels = []
        
        for i in range(len(rssi_grids[0].flatten())):
            signals = [grid.flatten()[i] for grid in rssi_grids]
            if len(signals) > 1:
                # Calculate interference as sum of all signals except the strongest
                sorted_signals = sorted(signals, reverse=True)
                interference = 10 * np.log10(sum(10**(s/10) for s in sorted_signals[1:]))
                interference_levels.append(interference)
        
        # Plot interference distribution
        ax.hist(interference_levels, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(np.mean(interference_levels), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(interference_levels):.1f} dBm')
        
        ax.set_title('Interference Analysis')
        ax.set_xlabel('Interference Level (dBm)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_coverage_efficiency(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot coverage efficiency analysis"""
        ap_names = list(ap_locations.keys())
        efficiencies = []
        
        for i, rssi_grid in enumerate(rssi_grids):
            rssi_values = rssi_grid.flatten()
            coverage_area = np.sum(rssi_values >= -67) / len(rssi_values)
            tx_power = ap_locations[ap_names[i]][3] if len(ap_locations[ap_names[i]]) > 3 else 20.0
            efficiency = coverage_area / tx_power  # Coverage per dBm
            efficiencies.append(efficiency)
        
        bars = ax.bar(ap_names, efficiencies, color='lightgreen', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, efficiency in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{efficiency:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Coverage Efficiency (Coverage/Dbm)')
        ax.set_ylabel('Efficiency')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
    def _plot_signal_statistics(self, ax, rssi_grids: List[np.ndarray]):
        """Plot signal statistics"""
        all_signals = []
        for rssi_grid in rssi_grids:
            all_signals.extend(rssi_grid.flatten())
        
        # Calculate statistics
        mean_signal = np.mean(all_signals)
        std_signal = np.std(all_signals)
        min_signal = np.min(all_signals)
        max_signal = np.max(all_signals)
        
        # Create statistics display
        stats_text = f"""
        Overall Signal Statistics
        
        Mean Signal: {mean_signal:.1f} dBm
        Std Deviation: {std_signal:.1f} dBm
        Min Signal: {min_signal:.1f} dBm
        Max Signal: {max_signal:.1f} dBm
        Signal Range: {max_signal - min_signal:.1f} dBm
        
        Coverage Quality:
        • Excellent (≥-40 dBm): {np.sum(np.array(all_signals) >= -40) / len(all_signals) * 100:.1f}%
        • Good (-50 to -40 dBm): {np.sum((np.array(all_signals) >= -50) & (np.array(all_signals) < -40)) / len(all_signals) * 100:.1f}%
        • Acceptable (-67 to -50 dBm): {np.sum((np.array(all_signals) >= -67) & (np.array(all_signals) < -50)) / len(all_signals) * 100:.1f}%
        • Poor (<-67 dBm): {np.sum(np.array(all_signals) < -67) / len(all_signals) * 100:.1f}%
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="lightblue", alpha=0.8))
        
        ax.set_title('Signal Statistics Summary')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def _plot_ap_load_distribution(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray], points: List[Tuple]):
        """Plot AP load distribution"""
        ap_names = list(ap_locations.keys())
        load_distribution = []
        
        # Calculate load for each AP (number of points where it's the strongest)
        for i, rssi_grid in enumerate(rssi_grids):
            load = 0
            for j, rssi_grid_other in enumerate(rssi_grids):
                if i != j:
                    # Count points where this AP is stronger
                    stronger_points = np.sum(rssi_grid > rssi_grid_other)
                    load += stronger_points
            load_distribution.append(load)
        
        # Normalize load
        total_load = sum(load_distribution)
        load_percentages = [load/total_load*100 for load in load_distribution]
        
        bars = ax.bar(ap_names, load_percentages, color='lightcoral', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, percentage in zip(bars, load_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('AP Load Distribution')
        ax.set_ylabel('Load Percentage (%)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
    def _plot_coverage_gaps(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray], points: List[Tuple]):
        """Plot coverage gaps analysis"""
        # Find coverage gaps
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        coverage_gaps = combined_grid < -67
        
        # Create gap visualization
        gap_im = ax.imshow(coverage_gaps, extent=[0, self.building_width, 0, self.building_height], 
                          origin='lower', cmap='Reds', aspect='auto')
        
        # Add AP locations
        for ap_name, ap_coords in ap_locations.items():
            x, y = ap_coords[0], ap_coords[1]
            ax.scatter(x, y, s=100, c='blue', marker='^', edgecolors='white', 
                      linewidth=2, zorder=10)
            ax.annotate(ap_name, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold', color='white')
        
        gap_percentage = np.sum(coverage_gaps) / coverage_gaps.size * 100
        ax.set_title(f'Coverage Gaps Analysis\n({gap_percentage:.1f}% gaps)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        plt.colorbar(gap_im, ax=ax, label='Coverage Gap')
        
    def _plot_power_efficiency(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot power efficiency analysis"""
        ap_names = list(ap_locations.keys())
        power_efficiencies = []
        
        for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
            tx_power = ap_coords[3] if len(ap_coords) > 3 else 20.0
            rssi_values = rssi_grids[i].flatten()
            mean_signal = np.mean(rssi_values)
            efficiency = (mean_signal + 100) / tx_power  # Signal per dBm
            power_efficiencies.append(efficiency)
        
        bars = ax.bar(ap_names, power_efficiencies, color='gold', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, efficiency in zip(bars, power_efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{efficiency:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Power Efficiency (Signal/Dbm)')
        ax.set_ylabel('Efficiency')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
    def _plot_system_metrics(self, ax, ap_locations: Dict, rssi_grids: List[np.ndarray]):
        """Plot overall system metrics"""
        # Calculate system-wide metrics
        combined_grid = np.max(np.stack(rssi_grids), axis=0)
        all_signals = combined_grid.flatten()
        
        total_coverage = np.sum(all_signals >= -67) / len(all_signals) * 100
        mean_signal = np.mean(all_signals)
        signal_variance = np.var(all_signals)
        total_power = sum(ap_coords[3] if len(ap_coords) > 3 else 20.0 for ap_coords in ap_locations.values())
        
        # Create metrics display
        metrics_text = f"""
        System Performance Summary
        
        Total APs: {len(ap_locations)}
        Total Coverage: {total_coverage:.1f}%
        Mean Signal: {mean_signal:.1f} dBm
        Signal Variance: {signal_variance:.1f} dB²
        Total Power: {total_power:.1f} dBm
        
        Coverage Quality:
        • Excellent: {np.sum(all_signals >= -40) / len(all_signals) * 100:.1f}%
        • Good: {np.sum((all_signals >= -50) & (all_signals < -40)) / len(all_signals) * 100:.1f}%
        • Acceptable: {np.sum((all_signals >= -67) & (all_signals < -50)) / len(all_signals) * 100:.1f}%
        • Poor: {np.sum(all_signals < -67) / len(all_signals) * 100:.1f}%
        
        System Efficiency: {total_coverage / total_power:.2f}%/dBm
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="lightgreen", alpha=0.8))
        
        ax.set_title('System Performance Metrics')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off') 