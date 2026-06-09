"""
Enhanced Floor Plan Processor for WiFi Signal Prediction

This module extends the original floor plan processor to support custom building boundaries
(polygon shapes) instead of forcing rectangular dimensions. This allows for more realistic
building layouts with irregular shapes.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import os
from typing import Dict, List, Tuple, Optional
from src.physics.materials import MATERIALS
from src.visualization.building_visualizer import BuildingVisualizer
import json

class EnhancedFloorPlanProcessor:
    def __init__(self):
        self.image = None
        self.image_path = None
        self.width_meters = None
        self.height_meters = None
        self.materials_grid = None
        self.visualizer = None
        self.regions = []  # List of (x, y, w, h, material) tuples
        self.resolution = 0.2  # 20 cm resolution
        self.building_boundary = None  # List of (x, y) tuples defining building perimeter
        self.use_custom_boundary = False  # Flag to use custom boundary instead of rectangular
        
    def load_image(self, image_path: str) -> bool:
        """
        Load a floor plan image (JPEG/PNG).
        
        Args:
            image_path: Path to the floor plan image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False
            
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
            
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_path = image_path
        
        print(f"Successfully loaded image: {image_path}")
        print(f"Image dimensions: {self.image.shape[1]} x {self.image.shape[0]} pixels")
        return True
    
    def set_building_dimensions(self, width_meters: float, height_meters: float):
        """
        Set the real-world dimensions of the building (for rectangular boundaries).
        
        Args:
            width_meters: Building width in meters
            height_meters: Building height in meters
        """
        self.width_meters = width_meters
        self.height_meters = height_meters
        self.use_custom_boundary = False
        print(f"Building dimensions set to: {width_meters}m x {height_meters}m (rectangular)")
    
    def set_custom_building_boundary(self, boundary_points: List[Tuple[float, float]]):
        """
        Set a custom building boundary using polygon points.
        
        Args:
            boundary_points: List of (x, y) tuples defining the building perimeter in meters
        """
        if len(boundary_points) < 3:
            print("Error: Boundary must have at least 3 points to form a polygon")
            return False
            
        self.building_boundary = boundary_points
        self.use_custom_boundary = True
        
        # Calculate bounding box for the custom boundary
        x_coords = [p[0] for p in boundary_points]
        y_coords = [p[1] for p in boundary_points]
        self.width_meters = max(x_coords) - min(x_coords)
        self.height_meters = max(y_coords) - min(y_coords)
        
        print(f"Custom building boundary set with {len(boundary_points)} points")
        print(f"Bounding box: {self.width_meters:.1f}m x {self.height_meters:.1f}m")
        return True
    
    def add_boundary_point_interactive(self):
        """
        Interactively add points to define the building boundary.
        """
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
            
        print("\n=== Adding Building Boundary Point ===")
        print("Enter coordinates in pixels (use grid as reference):")
        
        try:
            x_pixels = int(input("X coordinate: "))
            y_pixels = int(input("Y coordinate: "))
        except ValueError:
            print("Invalid coordinates. Please enter numbers only.")
            return
        
        # Convert to meters
        x_m, y_m = self.pixel_to_meters(x_pixels, y_pixels)
        
        # Initialize boundary if not exists
        if self.building_boundary is None:
            self.building_boundary = []
        
        self.building_boundary.append((x_m, y_m))
        self.use_custom_boundary = True
        
        print(f"Added boundary point: ({x_m:.1f}m, {y_m:.1f}m)")
        print(f"Total boundary points: {len(self.building_boundary)}")
        
        # Update bounding box
        if len(self.building_boundary) >= 2:
            x_coords = [p[0] for p in self.building_boundary]
            y_coords = [p[1] for p in self.building_boundary]
            self.width_meters = max(x_coords) - min(x_coords)
            self.height_meters = max(y_coords) - min(y_coords)
        
        # Automatically refresh the display
        self.display_image_with_grid()
    
    def define_boundary_by_coordinates(self):
        """
        Define custom polygon boundary by entering coordinates directly.
        Professional-grade: ensures at least 3 points, closes polygon, and validates input.
        """
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
        
        print("\n=== Define Custom Polygon Boundary by Coordinates ===")
        print("Enter coordinates in meters (not pixels). Example: 0,0 or 10.5,15.2")
        print("Type 'done' when finished to close the polygon. Minimum 3 points required.")
        
        self.building_boundary = []
        self.use_custom_boundary = True
        point_num = 1
        
        while True:
            print(f"\n--- Point {point_num} ---")
            coord_input = input("Enter coordinates (x,y) or 'done': ").strip().lower()
            if coord_input == 'done':
                break
            try:
                if ',' in coord_input:
                    x_str, y_str = coord_input.split(',')
                    x_m = float(x_str.strip())
                    y_m = float(y_str.strip())
                else:
                    print("Invalid format. Use 'x,y' format (e.g., 10.5,15.2)")
                    continue
                self.building_boundary.append((x_m, y_m))
                print(f"Added point {point_num}: ({x_m:.2f}m, {y_m:.2f}m)")
                point_num += 1
                # Optionally show preview after each point
                if len(self.building_boundary) >= 2:
                    self.display_image_with_grid()
            except ValueError:
                print("Invalid coordinates. Please enter numbers in 'x,y' format.")
                continue
        # Validation
        if len(self.building_boundary) < 3:
            print("Error: Need at least 3 points to form a polygon boundary.")
            self.building_boundary = []
            return
        # Close the polygon if not already closed
        if self.building_boundary[0] != self.building_boundary[-1]:
            self.building_boundary.append(self.building_boundary[0])
        print(f"Custom polygon boundary defined with {len(self.building_boundary)-1} sides.")
        self.display_image_with_grid()
    
    def define_boundary_by_grid_clicking(self):
        """
        Define custom polygon boundary by clicking on grid coordinates.
        Professional-grade: ensures at least 3 points, closes polygon, and validates input.
        """
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
        print("\n=== Define Custom Polygon Boundary by Grid Clicking ===")
        print("Look at the grid overlay in 'floor_plan_current_state.png'")
        print("Enter pixel coordinates from the grid (e.g., 100,50)")
        print("The system will convert pixels to meters automatically.")
        print("Type 'done' when finished to close the polygon. Minimum 3 points required.")
        self.building_boundary = []
        self.use_custom_boundary = True
        point_num = 1
        while True:
            print(f"\n--- Point {point_num} ---")
            coord_input = input("Enter pixel coordinates (x,y) or 'done': ").strip().lower()
            if coord_input == 'done':
                break
            try:
                if ',' in coord_input:
                    x_str, y_str = coord_input.split(',')
                    x_pixels = int(x_str.strip())
                    y_pixels = int(y_str.strip())
                else:
                    print("Invalid format. Use 'x,y' format (e.g., 100,50)")
                    continue
                x_m, y_m = self.pixel_to_meters(x_pixels, y_pixels)
                self.building_boundary.append((x_m, y_m))
                print(f"Added point {point_num}: Pixel ({x_pixels},{y_pixels}) → Meter ({x_m:.2f}m, {y_m:.2f}m)")
                point_num += 1
                if len(self.building_boundary) >= 2:
                    self.display_image_with_grid()
            except ValueError:
                print("Invalid coordinates. Please enter numbers in 'x,y' format.")
                continue
        if len(self.building_boundary) < 3:
            print("Error: Need at least 3 points to form a polygon boundary.")
            self.building_boundary = []
            return
        if self.building_boundary[0] != self.building_boundary[-1]:
            self.building_boundary.append(self.building_boundary[0])
        print(f"Custom polygon boundary defined with {len(self.building_boundary)-1} sides.")
        self.display_image_with_grid()
    
    def finish_boundary(self):
        """
        Finish defining the building boundary and close the polygon.
        """
        if self.building_boundary is None or len(self.building_boundary) < 3:
            print("Error: Need at least 3 points to form a building boundary")
            return False
        
        # Close the polygon by adding the first point at the end if not already closed
        if self.building_boundary[0] != self.building_boundary[-1]:
            self.building_boundary.append(self.building_boundary[0])
        
        print(f"Building boundary completed with {len(self.building_boundary)} points")
        
        # Automatically refresh the display
        self.display_image_with_grid()
        
        return True
    
    def clear_boundary(self):
        """Clear the current building boundary."""
        self.building_boundary = None
        self.use_custom_boundary = False
        print("Building boundary cleared")
    
    def get_building_perimeter_polygon(self):
        """
        Get the building perimeter polygon for AP placement optimization.
        
        Returns:
            List of (x, y) tuples defining the building perimeter, or None if not available
        """
        if self.use_custom_boundary and self.building_boundary:
            return self.building_boundary
        elif not self.use_custom_boundary and self.width_meters and self.height_meters:
            # Return rectangular boundary
            return [
                (0, 0),
                (self.width_meters, 0),
                (self.width_meters, self.height_meters),
                (0, self.height_meters),
                (0, 0)
            ]
        return None
    
    def is_point_inside_building(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the building boundary.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            bool: True if point is inside building boundary
        """
        if self.use_custom_boundary and self.building_boundary:
            # Use custom polygon boundary
            path = Path(self.building_boundary)
            return path.contains_point((x, y))
        elif not self.use_custom_boundary and self.width_meters and self.height_meters:
            # Use rectangular boundary
            return 0 <= x <= self.width_meters and 0 <= y <= self.height_meters
        return False
    
    def pixel_to_meters(self, x_pixels: int, y_pixels: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to real-world meters.
        
        Args:
            x_pixels: X coordinate in pixels
            y_pixels: Y coordinate in pixels
            
        Returns:
            Tuple of (x_meters, y_meters)
        """
        if self.image is None or self.width_meters is None or self.height_meters is None:
            return (0, 0)
            
        img_height, img_width = self.image.shape[:2]
        x_meters = (x_pixels / img_width) * self.width_meters
        y_meters = (y_pixels / img_height) * self.height_meters
        return (x_meters, y_meters)
    
    def meters_to_pixels(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        """
        Convert real-world meters to pixel coordinates.
        
        Args:
            x_meters: X coordinate in meters
            y_meters: Y coordinate in meters
            
        Returns:
            Tuple of (x_pixels, y_pixels)
        """
        if self.image is None or self.width_meters is None or self.height_meters is None:
            return (0, 0)
            
        img_height, img_width = self.image.shape[:2]
        x_pixels = int((x_meters / self.width_meters) * img_width)
        y_pixels = int((y_meters / self.height_meters) * img_height)
        return (x_pixels, y_pixels)
    
    def display_image_with_grid(self):
        """Display the floor plan image with a grid overlay and current boundary/regions."""
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
            
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(self.image)
        
        # Add dense grid overlay with markings every 10 units
        img_height, img_width = self.image.shape[:2]
        grid_spacing = 10  # pixels - dense grid every 10 units
        
        # Vertical lines
        for x in range(0, img_width, grid_spacing):
            alpha = 0.6 if x % 50 == 0 else 0.4  # Thicker lines every 50 pixels
            linewidth = 1.5 if x % 50 == 0 else 0.8
            ax.axvline(x=x, color='darkred', alpha=alpha, linewidth=linewidth)
            
        # Horizontal lines
        for y in range(0, img_height, grid_spacing):
            alpha = 0.6 if y % 50 == 0 else 0.4  # Thicker lines every 50 pixels
            linewidth = 1.5 if y % 50 == 0 else 0.8
            ax.axhline(y=y, color='darkred', alpha=alpha, linewidth=linewidth)
        
        # Add coordinate labels every 50 pixels (major grid lines)
        for x in range(0, img_width, 50):
            ax.text(x, 10, f'{x}', color='darkred', fontsize=8, ha='center', weight='bold')
        for y in range(0, img_height, 50):
            ax.text(10, y, f'{y}', color='darkred', fontsize=8, va='center', weight='bold')
        
        # Draw building boundary if defined
        if self.building_boundary:
            boundary_pixels = [self.meters_to_pixels(x, y) for x, y in self.building_boundary]
            boundary_pixels = [(x, img_height - y) for x, y in boundary_pixels]  # Flip Y for image coordinates
            
            # Draw boundary line
            boundary_x = [p[0] for p in boundary_pixels]
            boundary_y = [p[1] for p in boundary_pixels]
            ax.plot(boundary_x, boundary_y, 'b-', linewidth=3, label='Building Boundary')
            
            # Draw prominent boundary points with dots and labels
            for i, (x, y) in enumerate(boundary_pixels):
                # Large, prominent dot
                ax.scatter(x, y, c='red', s=100, zorder=10, edgecolors='black', linewidth=2)
                
                # Point number label
                ax.text(x + 5, y + 5, f'P{i+1}', fontsize=12, fontweight='bold', 
                       color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Fill boundary area
            ax.fill(boundary_x, boundary_y, alpha=0.1, color='blue')
        
        # Draw regions if any
        for x, y, w, h, material in self.regions:
            x_pix, y_pix = self.meters_to_pixels(x, y)
            w_pix, h_pix = self.meters_to_pixels(w, h)
            
            # Get material color
            color = self.get_material_color(material)
            
            rect = Rectangle((x_pix, y_pix), w_pix, h_pix, 
                           facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x_pix + w_pix/2, y_pix + h_pix/2, material, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Set title based on current state
        title = "Floor Plan with Grid Overlay"
        if self.building_boundary:
            title += " and Building Boundary"
        if self.regions:
            title += " and Regions"
        title += "\nRed grid: 10-unit spacing, Thicker lines: 50-unit spacing"
        
        ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.tight_layout()
        
        # Save the image instead of showing it
        output_path = 'floor_plan_current_state.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Current floor plan state saved to: {output_path}")
        print("This file updates automatically with all your changes.")
        print("Use this image as reference for entering coordinates.")
    
    def add_region_interactive(self):
        """
        Interactively add a region to the floor plan.
        User clicks to define a rectangle and selects material.
        """
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
            
        print("\n=== Adding Region ===")
        print("Available materials:")
        for i, material_name in enumerate(MATERIALS.keys(), 1):
            print(f"{i:2d}. {material_name}")
        
        # Get material selection
        while True:
            try:
                material_choice = int(input("\nSelect material number: ")) - 1
                if 0 <= material_choice < len(MATERIALS):
                    material_name = list(MATERIALS.keys())[material_choice]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get region coordinates
        print(f"\nSelected material: {material_name}")
        print("Enter region coordinates (in pixels, use grid as reference):")
        
        try:
            x = int(input("X coordinate (left edge): "))
            y = int(input("Y coordinate (bottom edge): "))
            width = int(input("Width (in pixels): "))
            height = int(input("Height (in pixels): "))
        except ValueError:
            print("Invalid coordinates. Please enter numbers only.")
            return
        
        # Convert to meters
        x_m, y_m = self.pixel_to_meters(x, y)
        w_m, h_m = self.pixel_to_meters(width, height)
        
        # Check if region is inside building boundary
        if self.use_custom_boundary and self.building_boundary:
            # Check if the region corners are inside the boundary
            corners = [(x_m, y_m), (x_m + w_m, y_m), (x_m, y_m + h_m), (x_m + w_m, y_m + h_m)]
            inside_count = sum(1 for corner in corners if self.is_point_inside_building(*corner))
            
            if inside_count < 2:  # At least half the corners should be inside
                print("Warning: Region appears to be mostly outside the building boundary")
                proceed = input("Continue anyway? (y/n): ").lower()
                if proceed != 'y':
                    return
        
        # Add region
        self.regions.append((x_m, y_m, w_m, h_m, material_name))
        print(f"Added region: {material_name} at ({x_m:.1f}m, {y_m:.1f}m) with size {w_m:.1f}m x {h_m:.1f}m")
        
        # Automatically refresh the display
        self.display_image_with_grid()
    
    def remove_region(self):
        """Remove the last added region."""
        if self.regions:
            removed = self.regions.pop()
            print(f"Removed region: {removed[4]} at ({removed[0]:.1f}m, {removed[1]:.1f}m)")
            # Automatically refresh the display
            self.display_image_with_grid()
        else:
            print("No regions to remove.")
    
    def list_regions(self):
        """List all defined regions."""
        if not self.regions:
            print("No regions defined.")
            return
            
        print("\n=== Defined Regions ===")
        for i, (x, y, w, h, material) in enumerate(self.regions, 1):
            print(f"{i}. {material}: ({x:.1f}m, {y:.1f}m) - {w:.1f}m x {h:.1f}m")
    
    def preview_regions(self):
        """Display the floor plan with all defined regions overlaid - same as display_image_with_grid."""
        # Use the same unified display method
        self.display_image_with_grid()
    
    def get_material_color(self, material_name: str) -> str:
        """Get the color for a material."""
        material_colors = {
            'concrete': '#808080', 'glass': '#ADD8E6', 'wood': '#8B4513', 
            'drywall': '#F5F5F5', 'metal': '#C0C0C0', 'brick': "#A52929", 
            'plaster': '#FFFACD', 'tile': '#D3D3D3', 'stone': '#A9A9A9', 
            'asphalt': '#696969', 'carpet': '#B22222', 'plastic': '#FFB6C1',
            'foam': '#F0E68C', 'fabric': '#DDA0DD', 'paper': '#FFF0F5', 
            'ceramic': '#FAFAD2', 'rubber': '#FF6347', 'air': '#FFFFFF'
        }
        return material_colors.get(material_name.lower(), '#FFFFFF')
    
    def generate_materials_grid(self) -> bool:
        """
        Generate the materials grid from defined regions, respecting building boundary.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.regions:
            print("No regions defined. Please add regions first.")
            return False
            
        if self.width_meters is None or self.height_meters is None:
            print("Building dimensions not set. Please set dimensions first.")
            return False
        
        # Create visualizer with bounding box dimensions
        self.visualizer = BuildingVisualizer(
            width=self.width_meters, 
            height=self.height_meters, 
            resolution=self.resolution
        )
        
        # Add user-defined regions
        for x, y, w, h, material_name in self.regions:
            if material_name in MATERIALS:
                self.visualizer.add_material(MATERIALS[material_name], x, y, w, h)
            else:
                print(f"Warning: Unknown material '{material_name}', using air instead.")
                self.visualizer.add_material(MATERIALS['air'], x, y, w, h)
        
        # If using custom boundary, mask areas outside the boundary
        if self.use_custom_boundary and self.building_boundary:
            self._apply_boundary_mask()
        
        self.materials_grid = self.visualizer.materials_grid
        print(f"Generated materials grid: {len(self.materials_grid)} x {len(self.materials_grid[0])} cells")
        return True
    
    def _apply_boundary_mask(self):
        """
        Apply building boundary mask to the materials grid.
        Areas outside the boundary will be set to None (no material).
        """
        if not self.building_boundary or self.materials_grid is None:
            return
            
        # Create boundary path
        boundary_path = Path(self.building_boundary)
        
        # Apply mask to materials grid
        for i in range(len(self.materials_grid)):
            for j in range(len(self.materials_grid[0])):
                # Convert grid coordinates to real coordinates
                x = j * self.resolution
                y = i * self.resolution
                
                # Check if point is inside boundary
                if not boundary_path.contains_point((x, y)):
                    self.materials_grid[i][j] = MATERIALS['air']  # Use air instead of None
    
    def save_configuration(self, output_path: str):
        """
        Save the floor plan configuration to a JSON file.
        
        Args:
            output_path: Path to save the configuration file
        """
        config = {
            'image_path': self.image_path,
            'width_meters': self.width_meters,
            'height_meters': self.height_meters,
            'resolution': self.resolution,
            'use_custom_boundary': self.use_custom_boundary,
            'building_boundary': self.building_boundary,
            'regions': [
                {
                    'x': x, 'y': y, 'width': w, 'height': h, 'material': material
                }
                for x, y, w, h, material in self.regions
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {output_path}")
    
    def load_configuration(self, config_path: str) -> bool:
        """
        Load a floor plan configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load image (optional - don't fail if image is missing)
            if 'image_path' in config:
                if os.path.exists(config['image_path']):
                    if not self.load_image(config['image_path']):
                        print(f"Warning: Could not load image from {config['image_path']}, but continuing with configuration")
                else:
                    print(f"Warning: Image file not found at {config['image_path']}, but continuing with configuration")
            
            # Set dimensions
            if 'width_meters' in config and 'height_meters' in config:
                self.width_meters = config['width_meters']
                self.height_meters = config['height_meters']
            else:
                print("Error: Building dimensions not found in configuration")
                return False
            
            # Load custom boundary if present
            if 'use_custom_boundary' in config and config['use_custom_boundary']:
                if 'building_boundary' in config:
                    self.building_boundary = config['building_boundary']
                    self.use_custom_boundary = True
                    print(f"Loaded custom building boundary with {len(self.building_boundary)} points")
                else:
                    print("Warning: Custom boundary flag set but no boundary data found")
            
            # Load regions
            self.regions = []
            if 'regions' in config:
                for region in config['regions']:
                    self.regions.append((
                        region['x'], region['y'], region['width'], region['height'], region['material']
                    ))
            
            print(f"Configuration loaded from: {config_path}")
            print(f"  Building dimensions: {self.width_meters}m x {self.height_meters}m")
            print(f"  Custom boundary: {'Yes' if self.use_custom_boundary else 'No'}")
            print(f"  Number of regions: {len(self.regions)}")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def interactive_setup(self):
        """
        Run an interactive setup session for the floor plan with custom boundary support.
        Professional-grade: robust dimension and boundary handling, clear user guidance, and validation.
        """
        print("=== Enhanced Floor Plan Processor Interactive Setup ===")
        print("This setup supports custom building boundaries (polygon shapes)")
        
        # --- Load image ---
        while True:
            image_path = input("\nEnter path to floor plan image (JPEG/PNG): ").strip()
            if self.load_image(image_path):
                break
            print("Could not load image. Please try again.")
        
        # --- Set dimensions (must be > 0) ---
        while True:
            try:
                width = float(input("Enter building width in meters: "))
                height = float(input("Enter building height in meters: "))
                if width > 0 and height > 0:
                    self.set_building_dimensions(width, height)
                    break
                else:
                    print("Width and height must be positive numbers.")
            except ValueError:
                print("Please enter valid numbers.")
        
        # --- Clear any existing boundary or regions ---
        self.building_boundary = None
        self.regions = []
        self.use_custom_boundary = False
        
        # --- Create grid overlay ---
        print("\nCreating grid overlay for your floor plan...")
        self.display_image_with_grid()
        print("✓ Grid overlay created! Check 'floor_plan_current_state.png'")
        print("Use this image as reference for entering coordinates.")
        
        # --- Choose boundary type ---
        print("\n=== Building Boundary Setup ===")
        print("1. Use rectangular boundary (traditional)")
        print("2. Define custom polygon boundary by coordinates (recommended)")
        print("3. Define custom polygon boundary by clicking grid coordinates")
        
        while True:
            try:
                choice = int(input("Choose boundary type (1, 2, or 3): "))
                if choice == 1:
                    # Rectangular boundary: set as 4-corner polygon
                    print("\nDefining rectangular boundary...")
                    # Confirm dimensions
                    print(f"Current dimensions: width={self.width_meters}m, height={self.height_meters}m")
                    confirm = input("Use these dimensions? (y/n): ").strip().lower()
                    if confirm != 'y':
                        while True:
                            try:
                                width = float(input("Enter building width in meters: "))
                                height = float(input("Enter building height in meters: "))
                                if width > 0 and height > 0:
                                    self.set_building_dimensions(width, height)
                                    break
                                else:
                                    print("Width and height must be positive numbers.")
                            except ValueError:
                                print("Please enter valid numbers.")
                    # Set rectangular boundary as polygon
                    self.building_boundary = [
                        (0, 0),
                        (self.width_meters, 0),
                        (self.width_meters, self.height_meters),
                        (0, self.height_meters),
                        (0, 0)
                    ]
                    self.use_custom_boundary = False
                    print(f"Rectangular boundary set: {self.building_boundary}")
                    break
                elif choice == 2:
                    # Custom polygon boundary by coordinates
                    self.define_boundary_by_coordinates()
                    break
                elif choice == 3:
                    # Custom polygon boundary by clicking grid coordinates
                    self.define_boundary_by_grid_clicking()
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        # --- Display image with grid and boundary ---
        self.display_image_with_grid()
        
        # --- Interactive region definition ---
        while True:
            print("\n=== Region Management ===")
            print("1. Add region")
            print("2. Remove last region")
            print("3. List regions")
            print("4. Show current state (refresh display)")
            print("5. Generate materials grid")
            print("6. Save configuration")
            print("7. Exit")
            
            try:
                choice = int(input("Choose option (1-7): "))
                
                if choice == 1:
                    self.add_region_interactive()
                elif choice == 2:
                    self.remove_region()
                elif choice == 3:
                    self.list_regions()
                elif choice == 4:
                    self.display_image_with_grid()
                elif choice == 5:
                    if self.generate_materials_grid():
                        print("Materials grid generated successfully!")
                    else:
                        print("Failed to generate materials grid.")
                elif choice == 6:
                    output_path = input("Enter output file path (e.g., my_floor_plan.json): ").strip()
                    self.save_configuration(output_path)
                elif choice == 7:
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
                    
            except ValueError:
                print("Please enter a valid number.")
        
        print("Setup completed!")
    
    def get_materials_grid(self):
        """Get the generated materials grid."""
        return self.materials_grid
    
    def get_visualizer(self):
        """Get the building visualizer."""
        return self.visualizer
    
    def generate_ap_placement_visualization(self, ap_locations: dict, rssi_grids: Optional[List] = None, 
                                          output_path: str = "ap_placement_floor_plan.png"):
        """
        Generate AP placement visualization on the floor plan image.
        
        Args:
            ap_locations: Dictionary of AP locations
            rssi_grids: Optional list of RSSI grids for coverage visualization
            output_path: Path to save the visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.visualizer is None:
            print("No visualizer available. Please generate materials grid first.")
            return False
        
        # Set floor plan image if available
        if self.image_path and os.path.exists(self.image_path):
            self.visualizer.set_floor_plan_image(self.image_path)
        
        # Generate visualization
        if rssi_grids:
            return self.visualizer.plot_coverage_on_floor_plan_image(
                rssi_grids, ap_locations, output_path, show_regions=True
            )
        else:
            # Just show AP placement without coverage
            return self.visualizer.plot_ap_placement_on_floor_plan(
                ap_locations, None, output_path
            ) 