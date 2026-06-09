

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, PathPatch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from typing import Dict, List, Tuple, Optional, Any, cast
import matplotlib.colors as mcolors
import scipy.ndimage
import matplotlib.image as mpimg


def get_sharp_green_pink_cmap():
    # Pink for bad (-90 to -65), green for good (-65 to 0)
    colors = ["#ff69b4", "#00ff00"]  # pink, green
    cmap = mcolors.ListedColormap(colors)
    bounds = [-90, -65, 0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

class AdvancedHeatmapVisualizer:
    """High-quality heatmap visualizer for WiFi signal strength analysis."""
    
    def __init__(self, building_width: float, building_height: float):
        """
        Initialize the visualizer.
        
        Args:
            building_width: Width of the building in meters
            building_height: Height of the building in meters
        """
        self.building_width = building_width
        self.building_height = building_height
        
        # Set high-quality plotting style
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Use custom green-pink colormap
        self.custom_cmap = get_sharp_green_pink_cmap()
        self.norm = mcolors.Normalize(vmin=-100, vmax=0)
    
    def create_comprehensive_visualizations(self, ap_locations: Dict[str, Any], 
                                          materials_grid: Any, collector: Any, 
                                          points: List[Tuple[float, float, float]], 
                                          output_dir: str, engine: Any = None, regions: Optional[list] = None, roi_polygon: Optional[list] = None, background_image: Optional[str] = None, image_extent: Optional[list] = None) -> None:
        
        if regions is None:
            regions = []
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate signal strength grids
        ap_signal_grids, combined_signal_grid, x_unique, y_unique = self._calculate_signal_grids(
            ap_locations, collector, points
        )
        
        # 1. Create Individual AP Heatmaps
        print("Creating individual AP heatmaps...")
        for ap_name, signal_grid in ap_signal_grids.items():
            self.create_individual_ap_heatmap(
                ap_name, signal_grid, ap_locations[ap_name], 
                x_unique, y_unique, output_dir, materials_grid, regions=regions, roi_polygon=roi_polygon, background_image=background_image, image_extent=image_extent
            )
        
        # 2. Create Combined Coverage Heatmap
        print("Creating combined coverage heatmap...")
        # Pass roi_polygon explicitly
        self.create_combined_coverage_heatmap(
            combined_signal_grid, ap_locations, x_unique, y_unique, output_dir, materials_grid, regions=regions, roi_polygon=roi_polygon, background_image=background_image, image_extent=image_extent
        )
        if background_image:
            self.create_combined_coverage_heatmap(
                combined_signal_grid, ap_locations, x_unique, y_unique, output_dir, materials_grid, regions=regions, roi_polygon=roi_polygon, background_image=background_image, image_extent=image_extent, suffix='_with_bg'
            )
        
        # 3. Create Interactive Visualization
        print("Creating interactive visualization...")
        self.create_interactive_visualization(
            ap_signal_grids, combined_signal_grid, ap_locations,
            x_unique, y_unique, output_dir
        )
        
        # 4. Create Signal Quality Analysis
        print("Creating signal quality analysis...")
        self.create_signal_quality_analysis(
            ap_signal_grids, combined_signal_grid, ap_locations, output_dir
        )
        
        print(f"All visualizations saved to: {output_dir}")
    
    def _calculate_signal_grids(self, ap_locations: Dict[str, Any], collector: Any, 
                               points: List[Tuple[float, float, float]]) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate signal strength grids for each AP and combined coverage."""
        # Extract coordinates
        x_coords = np.array([x for (x, y, z) in points])
        y_coords = np.array([y for (x, y, z) in points])
        
        # --- HARD CAP: Downsample to a fixed grid size for plotting ---
        MAX_GRID_SIZE = 200
        MIN_GRID_SIZE = 50
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        # --- Degenerate grid check ---
        if x_min == x_max or y_min == y_max:
            raise ValueError(f"Cannot plot: all x or y values are the same (x: {x_min}–{x_max}, y: {y_min}–{y_max}). Check your input data, ROI, and region definitions.")
        n_x = max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, len(np.unique(x_coords))))
        n_y = max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, len(np.unique(y_coords))))
        x_unique = np.linspace(x_min, x_max, n_x)
        y_unique = np.linspace(y_min, y_max, n_y)
        grid_shape = (len(y_unique), len(x_unique))
        print(f"[DEBUG] Plotting grid shape: {grid_shape}")
        
        # Calculate individual AP signal grids
        ap_signal_grids = {}
        for ap_name, ap_coords in ap_locations.items():
            signal_grid = np.zeros(grid_shape)
            ap_x, ap_y = ap_coords[:2]
            for i, y in enumerate(y_unique):
                for j, x in enumerate(x_unique):
                    distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
                    signal = collector.calculate_rssi(distance, None)
                    signal_grid[i, j] = signal
            ap_signal_grids[ap_name] = signal_grid
        
        # Calculate combined signal grid (maximum signal at each point)
        combined_signal_grid = np.zeros(grid_shape)
        for i, y in enumerate(y_unique):
            for j, x in enumerate(x_unique):
                max_signal = -100
                for ap_name, ap_coords in ap_locations.items():
                    ap_x, ap_y = ap_coords[:2]
                    distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
                    signal = collector.calculate_rssi(distance, None)
                    max_signal = max(max_signal, signal)
                combined_signal_grid[i, j] = max_signal
        
        return ap_signal_grids, combined_signal_grid, x_unique, y_unique
    
    def create_individual_ap_heatmap(self, ap_name: str, signal_grid: np.ndarray, 
                                   ap_coords: Tuple[float, float, float], 
                                   x_unique: np.ndarray, y_unique: np.ndarray,
                                   output_dir: str, materials_grid: Any, regions: Optional[list]=None, roi_polygon: Optional[list]=None, background_image: Optional[str] = None, image_extent: Optional[list] = None) -> None:
        """Create high-quality individual AP heatmap with green-pink colormap and region overlays."""
        masked_grid = np.ma.masked_less(signal_grid, -90)
        smooth_grid = scipy.ndimage.gaussian_filter(masked_grid, sigma=1.0)
        cmap = self.get_green_to_pink_cmap()
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
        # Set extent to ROI bounding box if available
        if roi_polygon is not None and len(roi_polygon) >= 3:
            xs = [p[0] for p in roi_polygon]
            ys = [p[1] for p in roi_polygon]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            extent = (x0, x1, y0, y1)
        else:
            x0, x1 = float(x_unique[0]), float(x_unique[-1])
            y0, y1 = float(y_unique[0]), float(y_unique[-1])
            extent = (x0, x1, y0, y1)
        im = ax.imshow(
            smooth_grid.T,
            extent=extent,
            cmap=cmap,
            vmin=-90,
            vmax=0,
            interpolation='nearest',
            aspect='auto',
            alpha=0.95,
            zorder=2,
            origin='lower'
        )
        cbar = plt.colorbar(im, ax=ax, ticks=[0, -65, -90])
        cbar.ax.set_yticklabels(['0 (Strong)', '-65 (Good/Threshold)', '-90 (Weak)'])
        cbar.set_label('Signal Strength (dBm)', fontsize=12, fontweight='bold')
        # Do NOT invert y-axis so 0 is at the top, -90 at the bottom
        # Draw ROI boundary if provided
        if roi_polygon is not None and len(roi_polygon) >= 3:
            roi_patch = Polygon(roi_polygon, closed=True, fill=False, edgecolor='black', linewidth=4, linestyle='-', zorder=10)
            ax.add_patch(roi_patch)
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
        # Draw building regions (polygons) if available
        if regions is not None:
            palette = plt.get_cmap('tab20')
            for i, region in enumerate(regions):
                # Support both dict and object (BuildingRegion)
                if isinstance(region, dict):
                    name = region.get('name', f'Region {i+1}')
                    polygon = region.get('polygon')
                elif hasattr(region, 'name'):
                    name = getattr(region, 'name', f'Region {i+1}')
                    polygon = getattr(region, 'polygon', None)
                else:
                    name = f'Region {i+1}'
                    polygon = None
                # Draw polygons from 'polygon' key or attribute
                if polygon and isinstance(polygon, list) and len(polygon) >= 3:
                    poly = Polygon(polygon, closed=True, fill=True, alpha=0.35, edgecolor='black', linewidth=1, facecolor=palette(i % 20), zorder=5)
                    ax.add_patch(poly)
                    centroid = np.mean(np.array(polygon), axis=0)
                    ax.text(centroid[0], centroid[1], name, ha='center', va='center', fontsize=10, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'), zorder=6)
                elif region.get('shape') == 'circle' and all(k in region for k in ('cx', 'cy', 'r')):
                    cx, cy, r = region['cx'], region['cy'], region['r']
                    circ = Circle((cx, cy), r, fill=True, alpha=0.35, edgecolor='black', linewidth=1, facecolor=palette(i % 20), zorder=5)
                    ax.add_patch(circ)
                    ax.text(cx, cy, name, fontsize=16, fontweight='bold', color='black', ha='center', va='center', zorder=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor='none', boxshadow=True))
        # Modern AP markers with drop shadow (smaller size)
        ap_x, ap_y = ap_coords[:2]
        shadow = Circle((ap_x+0.3, ap_y-0.3), 0.7, facecolor='gray', edgecolor='none', alpha=0.3, zorder=9)
        ax.add_patch(shadow)
        ap_circle = Circle((ap_x, ap_y), 0.6, facecolor='white', edgecolor='black', linewidth=3, alpha=0.95, zorder=10)
        ax.add_patch(ap_circle)
        color = plt.get_cmap('tab10')(0)
        ap_inner = Circle((ap_x, ap_y), 0.4, facecolor=color, edgecolor='none', alpha=0.95, zorder=11)
        ax.add_patch(ap_inner)
        ax.text(ap_x, ap_y, f'{ap_name}', fontsize=13, fontweight='bold', ha='center', va='center', color='white', zorder=12, bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, alpha=0.8, edgecolor='none'))
        ax.text(ap_x, ap_y-2.1, f'({ap_x:.1f}, {ap_y:.1f})', fontsize=11, ha='center', va='top', color='black', alpha=0.7, zorder=12)
        ax.set_xlabel('X (meters)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=15, fontweight='bold')
        ax.set_title(f'AP {ap_name} Coverage Heatmap', fontsize=18, fontweight='bold', pad=18)
        ax.grid(False)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{ap_name}_heatmap.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Individual AP heatmap saved: {output_path}")

    def get_green_to_pink_cmap(self):
        # Custom colormap: 0 to -65 dBm = shades of green, -65 to -90 dBm = shades of pink
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            (0.0, '#008000'),    # 0 dBm, dark green (top)
            (0.72, '#adffb0'),   # -65 dBm, light green (middle)
            (0.72, '#ffd1e6'),   # -65 dBm, light pink (boundary)
            (1.0, '#ff69b4')     # -90 dBm, strong pink (bottom)
        ]
        return LinearSegmentedColormap.from_list("green_to_pink", colors, N=256)

    def create_combined_coverage_heatmap(self, combined_signal_grid: np.ndarray, 
                                       ap_locations: Dict[str, Any],
                                       x_unique: np.ndarray, y_unique: np.ndarray,
                                       output_dir: str, materials_grid: Any, regions: Optional[list]=None, roi_polygon: Optional[list]=None, background_image: Optional[str] = None, image_extent: Optional[list] = None, suffix: str = '') -> None:
        # Mask out areas with signal below -90 dBm (no coverage)
        masked_grid = np.ma.masked_less(combined_signal_grid, -90)
        # Mask out areas outside the ROI polygon if provided
        if roi_polygon is not None and len(roi_polygon) >= 3:
            from matplotlib.path import Path
            roi_path = Path(roi_polygon)
            X, Y = np.meshgrid(x_unique, y_unique, indexing='ij')
            mask = np.zeros(X.shape, dtype=bool)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    mask[i, j] = not roi_path.contains_point((X[i, j], Y[i, j]))
            masked_grid = np.ma.masked_where(mask.T, masked_grid)
        # Use green-to-pink colormap
        cmap = self.get_green_to_pink_cmap()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        # Set extent to ROI bounding box if available
        if roi_polygon is not None and len(roi_polygon) >= 3:
            xs = [p[0] for p in roi_polygon]
            ys = [p[1] for p in roi_polygon]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            extent = (x0, x1, y0, y1)
        else:
            x0, x1 = float(x_unique[0]), float(x_unique[-1])
            y0, y1 = float(y_unique[0]), float(y_unique[-1])
            extent = (x0, x1, y0, y1)
        im = ax.imshow(
            masked_grid.T,
            extent=extent,
            cmap=cmap,
            vmin=-90,
            vmax=0,
            interpolation='bilinear',
            aspect='auto',
            alpha=1.0,
            zorder=2,
            origin='lower'
        )
        # Colorbar outside plot
        cbar = plt.colorbar(im, ax=ax, pad=0.03, aspect=30, shrink=0.85, location='right', ticks=[0, -65, -90])
        cbar.set_label('Combined Signal Strength (dBm)', fontsize=16, fontweight='bold', labelpad=18)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks([0, -65, -90])
        cbar.set_ticklabels(['0 (Strong)', '-65 (Good/Threshold)', '-90 (Weak)'])
        # Do NOT invert y-axis so 0 is at the top, -90 at the bottom
        # Axes labels and ticks
        ax.set_xlabel('X (meters)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y (meters)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_xticks(np.linspace(x0, x1, 6))
        ax.set_yticks(np.linspace(y0, y1, 6))
        ax.tick_params(axis='both', which='major', labelsize=14, length=0)
        # Title
        ax.set_title('Combined WiFi Coverage Heatmap', fontsize=26, fontweight='bold', pad=30)
        # Tight layout, white background
        plt.tight_layout(pad=2.0)
        fig.patch.set_facecolor('white')
        # Save
        output_path = os.path.join(output_dir, f'combined_coverage_heatmap{suffix}.png')
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Combined coverage heatmap saved: {output_path}")
    
    def _draw_building_regions(self, ax, materials_grid: Any) -> None:
        """Draw building regions and materials on the plot."""
        if materials_grid is None:
            return
        
        # Draw building outline
        building_rect = Rectangle((0, 0), self.building_width, self.building_height, 
                                fill=False, edgecolor='black', linewidth=3, alpha=0.8)
        ax.add_patch(building_rect)
        
        # Draw material regions if available
        try:
            # This is a simplified version - you may need to adapt based on your materials_grid structure
            if hasattr(materials_grid, 'shape') and len(materials_grid.shape) >= 2:
                # Draw walls or material boundaries
                wall_rect = Rectangle((5, 5), self.building_width-10, self.building_height-10, 
                                    fill=False, edgecolor='gray', linewidth=2, alpha=0.6)
                ax.add_patch(wall_rect)
        except Exception as e:
            # If materials_grid structure is different, just draw basic building outline
            pass
    
    def create_interactive_visualization(self, ap_signal_grids: Dict[str, np.ndarray], 
                                       combined_signal_grid: np.ndarray,
                                       ap_locations: Dict[str, Any],
                                       x_unique: np.ndarray, y_unique: np.ndarray,
                                       output_dir: str) -> None:
        """Create interactive Plotly visualization."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots for individual APs and combined
            n_aps = len(ap_locations)
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Combined Coverage'] + list(ap_locations.keys())[:3],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Custom colorscale for signal strength
            colorscale = [
                [0, '#FF69B4'],      # Pink for weak signal
                [0.35, '#FFB6C1'],   # Light pink
                [0.65, '#00FF00'],   # Green for good signal
                [1, '#008000']       # Dark green
            ]
            
            # Combined coverage heatmap
            fig.add_trace(
                go.Heatmap(
                    z=combined_signal_grid,
                    x=x_unique,
                    y=y_unique,
                    colorscale=colorscale,
                    zmin=-100,
                    zmax=0,
                    name='Combined Coverage',
                    showscale=True,
                    colorbar=dict(title="Signal Strength (dBm)")
                ),
                row=1, col=1
            )
            
            # Individual AP heatmaps
            for i, (ap_name, signal_grid) in enumerate(list(ap_signal_grids.items())[:3]):
                row = (i + 1) // 2 + 1
                col = (i + 1) % 2 + 1
                
                fig.add_trace(
                    go.Heatmap(
                        z=signal_grid,
                        x=x_unique,
                        y=y_unique,
                        colorscale=colorscale,
                        zmin=-100,
                        zmax=0,
                        name=f'{ap_name} Coverage',
                        showscale=False
                    ),
                    row=row, col=col
                )
            
            # Add AP markers
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, (ap_name, ap_coords) in enumerate(ap_locations.items()):
                ap_x, ap_y = ap_coords[:2]
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=[ap_x],
                        y=[ap_y],
                        mode='markers+text',
                        marker=dict(size=15, color=color, symbol='circle'),
                        text=[ap_name],
                        textposition="top center",
                        name=f'{ap_name} Location',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive WiFi Coverage Analysis",
                title_x=0.5,
                width=1200,
                height=800,
                showlegend=False
            )
            
            # Save interactive HTML
            output_path = os.path.join(output_dir, 'interactive_coverage_analysis.html')
            fig.write_html(output_path)
            
            print(f"Interactive visualization saved: {output_path}")
            
        except ImportError:
            print("Plotly not available, skipping interactive visualization")
    
    def create_signal_quality_analysis(self, ap_signal_grids: Dict[str, np.ndarray], 
                                     combined_signal_grid: np.ndarray,
                                     ap_locations: Dict[str, Any], output_dir: str) -> None:
        """Create signal quality analysis plots."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
        
        # 1. Signal Quality Distribution
        ax1 = axes[0, 0]
        all_signals = combined_signal_grid.flatten()
        
        # Create histogram with custom bins
        bins = np.linspace(-100, 0, 50)
        n, bins, patches = ax1.hist(all_signals, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Color bins based on signal quality
        for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
            if bin_center >= -65:
                patch.set_facecolor('green')
            else:
                patch.set_facecolor('pink')
        
        ax1.axvline(x=-65, color='black', linestyle='--', linewidth=2, label='Good Signal Threshold (-65 dBm)')
        ax1.set_xlabel('Signal Strength (dBm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Signal Quality Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. AP Performance Comparison
        ax2 = axes[0, 1]
        ap_names = list(ap_locations.keys())
        avg_signals = [np.mean(grid) for grid in ap_signal_grids.values()]
        good_coverage_percent = [np.sum(grid >= -65) / grid.size * 100 for grid in ap_signal_grids.values()]
        
        x = np.arange(len(ap_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, avg_signals, width, label='Average Signal (dBm)', alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, good_coverage_percent, width, label='Good Coverage (%)', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Access Points', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Signal (dBm)', fontsize=12, fontweight='bold')
        ax2_twin.set_ylabel('Good Coverage (%)', fontsize=12, fontweight='bold')
        ax2.set_title('AP Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ap_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_signals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        for bar, value in zip(bars2, good_coverage_percent):
            height = bar.get_height()
            ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Coverage Quality Map
        ax3 = axes[1, 0]
        coverage_quality = np.where(combined_signal_grid >= -65, 1, 0)  # Binary: good/bad coverage
        
        im = ax3.imshow(coverage_quality, extent=(0, self.building_width, 0, self.building_height), 
                       origin='lower', cmap='RdYlGn', aspect='equal', alpha=0.8)
        
        # Add AP locations
        for ap_name, ap_coords in ap_locations.items():
            ap_x, ap_y = ap_coords[:2]
            ax3.scatter(ap_x, ap_y, s=200, c='red', marker='^', edgecolors='black', linewidth=2, zorder=10)
            ax3.annotate(ap_name, (ap_x, ap_y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))
        
        ax3.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax3.set_title('Coverage Quality Map\nGreen: Good Signal (≥-65 dBm), Red: Weak Signal (<-65 dBm)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Coverage Quality', fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Weak Signal', 'Good Signal'])
        
        # 4. Signal Strength Statistics
        ax4 = axes[1, 1]
        
        # Calculate statistics
        stats_data = {
            'Metric': ['Min Signal', 'Max Signal', 'Mean Signal', 'Std Signal', 'Good Coverage %', 'Weak Coverage %'],
            'Value': [
                np.min(combined_signal_grid),
                np.max(combined_signal_grid),
                np.mean(combined_signal_grid),
                np.std(combined_signal_grid),
                np.sum(combined_signal_grid >= -65) / combined_signal_grid.size * 100,
                np.sum(combined_signal_grid < -65) / combined_signal_grid.size * 100
            ]
        }
        
        # Create table
        table_data = [[stats_data['Metric'][i], f"{stats_data['Value'][i]:.2f}"] 
                      for i in range(len(stats_data['Metric']))]
        
        table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(table_data)):
            for j in range(2):
                cell = table[(i+1, j)]
                if i < 4:  # Signal statistics
                    cell.set_facecolor('#E6F3FF')
                else:  # Coverage statistics
                    cell.set_facecolor('#E6FFE6' if 'Good' in table_data[i][0] else '#FFE6E6')
        
        ax4.set_title('Signal Strength Statistics', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'signal_quality_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Signal quality analysis saved: {output_path}")


# Convenience function for backward compatibility
def create_visualization_plots(ap_locations, building_width, building_height, materials_grid, collector, points, output_dir, engine=None, regions: Optional[list] = None, roi_polygon: Optional[list] = None, background_image: Optional[str] = None, image_extent: Optional[list] = None):
    """
    Create comprehensive high-quality heatmap visualizations for AP placement analysis.
    
    This is a convenience function that creates an AdvancedHeatmapVisualizer instance
    and calls the comprehensive visualization method.
    """
    if regions is None:
        regions = []
    visualizer = AdvancedHeatmapVisualizer(building_width, building_height)
    visualizer.create_comprehensive_visualizations(
        ap_locations, materials_grid, collector, points, output_dir, engine, regions=regions, roi_polygon=roi_polygon, background_image=background_image, image_extent=image_extent
    ) 