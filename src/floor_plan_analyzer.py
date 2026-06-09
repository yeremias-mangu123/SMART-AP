#!/usr/bin/env python3
"""
Comprehensive Floor Plan Analyzer
Maps building regions with coordinates, materials, and boundaries for AP placement and interference analysis.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import os

class MaterialType(Enum):
    """Material types for building regions."""
    AIR = "air"
    BRICK = "brick"
    CONCRETE = "concrete"
    DRYWALL = "drywall"
    GLASS = "glass"
    CARPET = "carpet"
    TILE = "tile"
    METAL = "metal"
    WOOD = "wood"
    PLASTIC = "plastic"
    FABRIC = "fabric"
    STONE = "stone"

@dataclass
class RegionBoundary:
    """Defines the boundary of a building region."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 3.0  # Default ceiling height
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def depth(self) -> float:
        return self.z_max - self.z_min
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def volume(self) -> float:
        return self.area * self.depth
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )
    
    def contains_point(self, x: float, y: float, z: float = 1.5) -> bool:
        """Check if a point is inside this region."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)
    
    def intersects(self, other: 'RegionBoundary') -> bool:
        """Check if this region intersects with another."""
        return not (self.x_max < other.x_min or self.x_min > other.x_max or
                   self.y_max < other.y_min or self.y_min > other.y_max or
                   self.z_max < other.z_min or self.z_min > other.z_max)

@dataclass
class BuildingRegion:
    """Represents a region in the building with full metadata."""
    id: str
    name: str
    region_type: str  # 'room', 'corridor', 'wall', 'open_space', 'facility'
    boundary: RegionBoundary
    material: MaterialType
    material_properties: Dict[str, Any]
    usage: str = "general"
    priority: int = 1  # Higher priority regions get APs first
    user_density: float = 0.1  # Users per square meter
    device_density: float = 0.15  # Devices per square meter
    interference_sensitivity: float = 1.0  # How sensitive to interference
    coverage_requirement: float = 0.9  # Required coverage percentage
    polygon: Optional[List[Tuple[float, float]]] = None  # Polygonal boundary points
    is_polygonal: bool = False  # Whether this region uses polygon instead of bounding box
    
    def __post_init__(self):
        """Set default material properties based on material type."""
        if not self.material_properties:
            self.material_properties = self._get_default_material_properties()
        
        # Determine if this is a polygonal region
        if self.polygon is not None and len(self.polygon) >= 3:
            self.is_polygonal = True
            # Update boundary to encompass the polygon
            xs = [pt[0] for pt in self.polygon]
            ys = [pt[1] for pt in self.polygon]
            self.boundary = RegionBoundary(
                x_min=min(xs), y_min=min(ys),
                x_max=max(xs), y_max=max(ys),
                z_min=self.boundary.z_min, z_max=self.boundary.z_max
            )
    
    def _get_default_material_properties(self) -> Dict[str, Any]:
        """Get default properties for the material type."""
        defaults = {
            MaterialType.AIR: {
                'attenuation_db': 0.0,
                'reflection_coefficient': 0.0,
                'transmission_coefficient': 1.0,
                'frequency_dependent': False
            },
            MaterialType.BRICK: {
                'attenuation_db': 8.0,
                'reflection_coefficient': 0.3,
                'transmission_coefficient': 0.1,
                'frequency_dependent': True
            },
            MaterialType.CONCRETE: {
                'attenuation_db': 12.0,
                'reflection_coefficient': 0.4,
                'transmission_coefficient': 0.05,
                'frequency_dependent': True
            },
            MaterialType.DRYWALL: {
                'attenuation_db': 3.0,
                'reflection_coefficient': 0.2,
                'transmission_coefficient': 0.3,
                'frequency_dependent': True
            },
            MaterialType.GLASS: {
                'attenuation_db': 2.0,
                'reflection_coefficient': 0.1,
                'transmission_coefficient': 0.8,
                'frequency_dependent': True
            },
            MaterialType.CARPET: {
                'attenuation_db': 1.0,
                'reflection_coefficient': 0.1,
                'transmission_coefficient': 0.9,
                'frequency_dependent': False
            },
            MaterialType.TILE: {
                'attenuation_db': 1.5,
                'reflection_coefficient': 0.2,
                'transmission_coefficient': 0.8,
                'frequency_dependent': False
            }
        }
        return defaults.get(self.material, defaults[MaterialType.AIR])

    def contains_point(self, x: float, y: float, z: float = 1.5) -> bool:
        """Check if a point is inside this region (supports both bounding box and polygon)."""
        # First check if point is within bounding box (quick rejection)
        if not self.boundary.contains_point(x, y, z):
            return False
        
        # If it's a polygonal region, do detailed polygon test
        if self.is_polygonal and self.polygon:
            return point_in_polygon(x, y, self.polygon)
        
        # Otherwise, use bounding box
        return True
    
    def get_centroid(self) -> Tuple[float, float, float]:
        """Get the centroid of the region (center of mass for polygons)."""
        if self.is_polygonal and self.polygon:
            # Calculate centroid of polygon
            n = len(self.polygon)
            if n == 0:
                return self.boundary.center
            
            # Shoelace formula for polygon centroid
            cx = cy = 0.0
            area = 0.0
            
            for i in range(n):
                j = (i + 1) % n
                xi, yi = self.polygon[i]
                xj, yj = self.polygon[j]
                
                cross = xi * yj - xj * yi
                cx += (xi + xj) * cross
                cy += (yi + yj) * cross
                area += cross
            
            if abs(area) < 1e-10:  # Degenerate polygon
                return self.boundary.center
            
            area /= 2.0
            cx /= (6.0 * area)
            cy /= (6.0 * area)
            
            return (cx, cy, (self.boundary.z_min + self.boundary.z_max) / 2)
        else:
            return self.boundary.center
    
    def get_area(self) -> float:
        """Calculate the area of the region."""
        if self.is_polygonal and self.polygon:
            # Calculate polygon area using shoelace formula
            n = len(self.polygon)
            if n < 3:
                return 0.0
            
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                xi, yi = self.polygon[i]
                xj, yj = self.polygon[j]
                area += xi * yj - xj * yi
            
            return abs(area) / 2.0
        else:
            return self.boundary.area
    
    def get_perimeter(self) -> float:
        """Calculate the perimeter of the region."""
        if self.is_polygonal and self.polygon:
            # Calculate polygon perimeter
            n = len(self.polygon)
            if n < 2:
                return 0.0
            
            perimeter = 0.0
            for i in range(n):
                j = (i + 1) % n
                xi, yi = self.polygon[i]
                xj, yj = self.polygon[j]
                perimeter += np.sqrt((xj - xi)**2 + (yj - yi)**2)
            
            return perimeter
        else:
            return 2 * (self.boundary.width + self.boundary.height)
    
    def get_optimal_ap_positions(self, num_aps: int = 1) -> List[Tuple[float, float, float]]:
        """Get optimal AP positions within this region."""
        if num_aps <= 0:
            return []
        
        if self.is_polygonal and self.polygon:
            # For polygonal regions, use centroid and distribute around it
            centroid = self.get_centroid()
            if num_aps == 1:
                return [centroid]
            
            # For multiple APs, distribute them within the polygon
            positions = []
            area = self.get_area()
            radius = np.sqrt(area / (np.pi * num_aps)) * 0.7  # 70% of theoretical radius
            
            # Place first AP at centroid
            positions.append(centroid)
            
            # Place remaining APs in a pattern within the polygon
            for i in range(1, num_aps):
                angle = 2 * np.pi * i / num_aps
                distance = radius * (0.5 + 0.5 * (i % 2))  # Vary distance
                
                x = centroid[0] + distance * np.cos(angle)
                y = centroid[1] + distance * np.sin(angle)
                z = centroid[2]
                
                # Ensure point is within polygon
                if self.contains_point(x, y, z):
                    positions.append((x, y, z))
                else:
                    # Fallback to centroid
                    positions.append(centroid)
            
            return positions
        else:
            # For rectangular regions, use grid placement
            if num_aps == 1:
                return [self.boundary.center]
            
            # Calculate grid dimensions
            cols = int(np.ceil(np.sqrt(num_aps)))
            rows = int(np.ceil(num_aps / cols))
            
            positions = []
            for i in range(num_aps):
                col = i % cols
                row = i // cols
                
                x = self.boundary.x_min + (col + 0.5) * self.boundary.width / cols
                y = self.boundary.y_min + (row + 0.5) * self.boundary.height / rows
                z = (self.boundary.z_min + self.boundary.z_max) / 2
                
                positions.append((x, y, z))
            
            return positions

class FloorPlanAnalyzer:
    """Comprehensive floor plan analyzer for building regions and materials."""
    
    def __init__(self, building_width: float, building_length: float, building_height: float):
        self.building_width = building_width
        self.building_length = building_length
        self.building_height = building_height
        self.regions: List[BuildingRegion] = []
        self.materials_grid = None
        self.resolution = 0.2  # meters per grid cell
        
    def analyze_complex_office_layout(self) -> List[BuildingRegion]:
        """Analyze and create a comprehensive office building layout."""
        logging.info("Creating comprehensive office building layout analysis...")
        
        regions = []
        region_id = 1
        
        # Define building perimeter
        wall_thickness = 0.3
        perimeter = BuildingRegion(
            id=f"region_{region_id}",
            name="Building Perimeter",
            region_type="wall",
            boundary=RegionBoundary(0, 0, self.building_width, self.building_length),
            material=MaterialType.BRICK,
            material_properties={},
            usage="structural",
            priority=1
        )
        regions.append(perimeter)
        region_id += 1
        
        # Define internal regions based on typical office layout
        internal_regions = self._define_internal_regions(region_id)
        regions.extend(internal_regions)
        
        self.regions = regions
        logging.info(f"Created {len(regions)} building regions")
        
        # Generate materials grid
        self._generate_materials_grid()
        
        return regions
    
    def _define_internal_regions(self, start_id: int) -> List[BuildingRegion]:
        """Define internal building regions with realistic office layout."""
        regions = []
        region_id = start_id
        wall_thickness = 0.3
        
        # Lobby and Reception Area
        lobby = BuildingRegion(
            id=f"region_{region_id}",
            name="Lobby",
            region_type="room",
            boundary=RegionBoundary(wall_thickness, wall_thickness, 8.0, 6.0),
            material=MaterialType.TILE,
            material_properties={},
            usage="reception",
            priority=3,
            user_density=0.05,
            device_density=0.1
        )
        regions.append(lobby)
        region_id += 1
        
        # Conference Rooms
        conf_rooms = [
            {"name": "Conference Room 1", "x": wall_thickness + 10, "y": self.building_length - 8, "w": 8, "h": 6},
            {"name": "Conference Room 2", "x": wall_thickness + 20, "y": self.building_length - 6, "w": 6, "h": 4},
            {"name": "Conference Room 3", "x": wall_thickness + 28, "y": self.building_length - 5, "w": 4, "h": 3}
        ]
        
        for conf in conf_rooms:
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=conf["name"],
                region_type="room",
                boundary=RegionBoundary(conf["x"], conf["y"], conf["x"] + conf["w"], conf["y"] + conf["h"]),
                material=MaterialType.GLASS,
                material_properties={},
                usage="meeting",
                priority=4,
                user_density=0.3,
                device_density=0.4,
                interference_sensitivity=1.2
            )
            regions.append(room)
            region_id += 1
        
        # Executive Offices
        exec_offices = [
            {"name": "CEO Office", "x": self.building_width - 8, "y": self.building_length - 10, "w": 8, "h": 10},
            {"name": "CFO Office", "x": self.building_width - 6, "y": self.building_length - 6, "w": 6, "h": 6}
        ]
        
        for office in exec_offices:
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=office["name"],
                region_type="room",
                boundary=RegionBoundary(office["x"], office["y"], office["x"] + office["w"], office["y"] + office["h"]),
                material=MaterialType.CARPET,
                material_properties={},
                usage="executive",
                priority=5,
                user_density=0.1,
                device_density=0.2,
                interference_sensitivity=1.5
            )
            regions.append(room)
            region_id += 1
        
        # Department Areas
        dept_areas = [
            {"name": "IT Department", "x": wall_thickness + 2, "y": wall_thickness + 8, "w": 12, "h": 8},
            {"name": "Marketing Department", "x": wall_thickness + 16, "y": wall_thickness + 8, "w": 10, "h": 8},
            {"name": "Sales Department", "x": wall_thickness + 28, "y": wall_thickness + 8, "w": 10, "h": 8}
        ]
        
        for dept in dept_areas:
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=dept["name"],
                region_type="room",
                boundary=RegionBoundary(dept["x"], dept["y"], dept["x"] + dept["w"], dept["y"] + dept["h"]),
                material=MaterialType.CARPET,
                material_properties={},
                usage="department",
                priority=4,
                user_density=0.2,
                device_density=0.3
            )
            regions.append(room)
            region_id += 1
        
        # Individual Offices
        office_width, office_height = 4.0, 5.0
        office_spacing = 0.5
        
        for row in range(3):
            for col in range(3):
                x = wall_thickness + 2 + col * (office_width + office_spacing)
                y = wall_thickness + 18 + row * (office_height + 0.5)
                
                office = BuildingRegion(
                    id=f"region_{region_id}",
                    name=f"Office {row*3 + col + 1}",
                    region_type="room",
                    boundary=RegionBoundary(x, y, x + office_width, y + office_height),
                    material=MaterialType.DRYWALL,
                    material_properties={},
                    usage="individual",
                    priority=3,
                    user_density=0.1,
                    device_density=0.15
                )
                regions.append(office)
                region_id += 1
        
        # Facilities
        facilities = [
            {"name": "Break Room", "x": wall_thickness + 16, "y": wall_thickness + 18, "w": 6, "h": 4, "material": MaterialType.TILE},
            {"name": "Kitchen", "x": wall_thickness + 16, "y": wall_thickness + 24, "w": 6, "h": 3, "material": MaterialType.TILE},
            {"name": "Server Room", "x": wall_thickness + 2, "y": wall_thickness + 40, "w": 4, "h": 6, "material": MaterialType.CONCRETE},
            {"name": "Storage", "x": wall_thickness + 8, "y": wall_thickness + 40, "w": 4, "h": 6, "material": MaterialType.DRYWALL},
            {"name": "Men's Restroom", "x": wall_thickness + 30, "y": wall_thickness + 18, "w": 3, "h": 4, "material": MaterialType.TILE},
            {"name": "Women's Restroom", "x": wall_thickness + 35, "y": wall_thickness + 18, "w": 3, "h": 4, "material": MaterialType.TILE},
            {"name": "Print Room", "x": wall_thickness + 30, "y": wall_thickness + 24, "w": 4, "h": 3, "material": MaterialType.DRYWALL}
        ]
        
        for facility in facilities:
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=facility["name"],
                region_type="facility",
                boundary=RegionBoundary(facility["x"], facility["y"], 
                                      facility["x"] + facility["w"], facility["y"] + facility["h"]),
                material=facility["material"],
                material_properties={},
                usage="facility",
                priority=2,
                user_density=0.05,
                device_density=0.1
            )
            regions.append(room)
            region_id += 1
        
        # Phone Booths
        booths = [
            {"x": wall_thickness + 36, "y": wall_thickness + 8, "w": 2, "h": 2},
            {"x": wall_thickness + 36, "y": wall_thickness + 12, "w": 2, "h": 2}
        ]
        
        for i, booth in enumerate(booths):
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=f"Phone Booth {i+1}",
                region_type="room",
                boundary=RegionBoundary(booth["x"], booth["y"], 
                                      booth["x"] + booth["w"], booth["y"] + booth["h"]),
                material=MaterialType.GLASS,
                material_properties={},
                usage="private",
                priority=2,
                user_density=0.1,
                device_density=0.1
            )
            regions.append(room)
            region_id += 1
        
        # Collaboration Space
        collab = BuildingRegion(
            id=f"region_{region_id}",
            name="Collaboration Space",
            region_type="room",
            boundary=RegionBoundary(wall_thickness + 16, wall_thickness + 28, 
                                  wall_thickness + 28, wall_thickness + 36),
            material=MaterialType.CARPET,
            material_properties={},
            usage="collaboration",
            priority=4,
            user_density=0.15,
            device_density=0.25,
            interference_sensitivity=1.1
        )
        regions.append(collab)
        region_id += 1
        
        # Corridors
        corridors = [
            {"name": "Main Corridor", "x": wall_thickness + 2, "y": wall_thickness + 16, "w": self.building_width - 2*wall_thickness - 4, "h": 1.5},
            {"name": "Vertical Corridor", "x": wall_thickness + 15, "y": wall_thickness + 8, "w": 1.5, "h": 8}
        ]
        
        for corridor in corridors:
            room = BuildingRegion(
                id=f"region_{region_id}",
                name=corridor["name"],
                region_type="corridor",
                boundary=RegionBoundary(corridor["x"], corridor["y"], 
                                      corridor["x"] + corridor["w"], corridor["y"] + corridor["h"]),
                material=MaterialType.TILE,
                material_properties={},
                usage="circulation",
                priority=1,
                user_density=0.02,
                device_density=0.05
            )
            regions.append(room)
            region_id += 1
        
        return regions
    
    def _generate_materials_grid(self):
        """Generate a 3D materials grid based on the regions."""
        grid_width = int(self.building_width / self.resolution)
        grid_height = int(self.building_length / self.resolution)
        grid_depth = int(self.building_height / self.resolution)
        
        # Initialize with air
        self.materials_grid = [[[MaterialType.AIR for _ in range(grid_width)] 
                               for _ in range(grid_height)] 
                              for _ in range(grid_depth)]
        
        # Fill in materials based on regions
        for region in self.regions:
            self._fill_region_in_grid(region)
        
        logging.info(f"Generated 3D materials grid: {grid_depth}x{grid_height}x{grid_width}")
    
    def _fill_region_in_grid(self, region: BuildingRegion):
        """Fill a region's material into the 3D grid."""
        boundary = region.boundary
        if self.materials_grid is None:
            return
        # Convert to grid coordinates
        x_dim = len(self.materials_grid[0][0]) if self.materials_grid and self.materials_grid[0] and self.materials_grid[0][0] else 0
        y_dim = len(self.materials_grid[0]) if self.materials_grid and self.materials_grid[0] else 0
        z_dim = len(self.materials_grid) if self.materials_grid else 0
        x_min = max(0, int(boundary.x_min / self.resolution))
        x_max = min(x_dim, int(boundary.x_max / self.resolution))
        y_min = max(0, int(boundary.y_min / self.resolution))
        y_max = min(y_dim, int(boundary.y_max / self.resolution))
        z_min = max(0, int(boundary.z_min / self.resolution))
        z_max = min(z_dim, int(boundary.z_max / self.resolution))
        # Fill the region
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Convert grid coordinates back to world coordinates
                    world_x = x * self.resolution
                    world_y = y * self.resolution
                    world_z = z * self.resolution
                    
                    # Check if this grid cell is inside the region
                    if region.contains_point(world_x, world_y, world_z):
                        self.materials_grid[z][y][x] = region.material
    
    def get_region_at_point(self, x: float, y: float, z: float = 1.5) -> Optional[BuildingRegion]:
        """Get the region that contains a given point."""
        for region in self.regions:
            if region.contains_point(x, y, z):
                return region
        return None
    
    def get_high_priority_regions(self) -> List[BuildingRegion]:
        """Get regions that should have APs placed in them."""
        return [r for r in self.regions if r.region_type == "room" and r.priority >= 3]
    
    def get_interference_sensitive_regions(self) -> List[BuildingRegion]:
        """Get regions that are sensitive to interference."""
        return [r for r in self.regions if r.interference_sensitivity > 1.0]
    
    def calculate_total_user_load(self) -> float:
        """Calculate total user load across all regions."""
        total_load = 0.0
        for region in self.regions:
            if region.region_type == "room":
                total_load += region.boundary.area * region.user_density
        return total_load
    
    def calculate_total_device_load(self) -> float:
        """Calculate total device load across all regions."""
        total_load = 0.0
        for region in self.regions:
            if region.region_type == "room":
                total_load += region.boundary.area * region.device_density
        return total_load
    
    def export_analysis(self, filepath: str):
        """Export the floor plan analysis to JSON."""
        analysis_data = {
            "building_dimensions": {
                "width": self.building_width,
                "length": self.building_length,
                "height": self.building_height
            },
            "regions": []
        }
        
        for region in self.regions:
            region_data = {
                "id": region.id,
                "name": region.name,
                "type": region.region_type,
                "boundary": {
                    "x_min": region.boundary.x_min,
                    "y_min": region.boundary.y_min,
                    "x_max": region.boundary.x_max,
                    "y_max": region.boundary.y_max,
                    "z_min": region.boundary.z_min,
                    "z_max": region.boundary.z_max
                },
                "material": region.material.value,
                "material_properties": region.material_properties,
                "usage": region.usage,
                "priority": region.priority,
                "user_density": region.user_density,
                "device_density": region.device_density,
                "interference_sensitivity": region.interference_sensitivity,
                "coverage_requirement": region.coverage_requirement
            }
            analysis_data["regions"].append(region_data)
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logging.info(f"Floor plan analysis exported to {filepath}")
    
    def get_ap_placement_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for AP placement based on region analysis."""
        high_priority_regions = self.get_high_priority_regions()
        total_user_load = self.calculate_total_user_load()
        total_device_load = self.calculate_total_device_load()
        
        # Calculate recommended AP count based on user/device load
        recommended_aps = max(
            len(high_priority_regions),  # At least one AP per high-priority room
            int(total_user_load / 10),   # One AP per 10 users
            int(total_device_load / 25)  # One AP per 25 devices
        )
        
        # Get optimal AP locations (room centers)
        ap_locations = []
        for region in high_priority_regions:
            center = region.boundary.center
            ap_locations.append({
                "region_id": region.id,
                "region_name": region.name,
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "priority": region.priority,
                "user_density": region.user_density,
                "device_density": region.device_density
            })
        
        return {
            "recommended_ap_count": recommended_aps,
            "ap_locations": ap_locations,
            "total_user_load": total_user_load,
            "total_device_load": total_device_load,
            "high_priority_regions": len(high_priority_regions),
            "interference_sensitive_regions": len(self.get_interference_sensitive_regions())
        } 

    def create_complex_polygonal_layout(self) -> List[BuildingRegion]:
        """Create a complex office layout with polygonal regions for testing."""
        logging.info("Creating complex polygonal office layout...")
        
        regions = []
        region_id = 1
        
        # L-shaped office area
        l_office_polygon = [
            (2.0, 2.0), (15.0, 2.0), (15.0, 8.0), (10.0, 8.0), (10.0, 12.0), (2.0, 12.0)
        ]
        l_office = BuildingRegion(
            id=f"region_{region_id}",
            name="L-Shaped Office",
            region_type="room",
            boundary=RegionBoundary(2, 2, 15, 12),
            material=MaterialType.CARPET,
            material_properties={},
            usage="office",
            priority=4,
            user_density=0.15,
            device_density=0.25,
            polygon=l_office_polygon
        )
        regions.append(l_office)
        region_id += 1
        
        # Circular conference room
        center_x, center_y = 25, 10
        radius = 6
        conference_polygon = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            conference_polygon.append((x, y))
        
        conference = BuildingRegion(
            id=f"region_{region_id}",
            name="Circular Conference Room",
            region_type="room",
            boundary=RegionBoundary(center_x - radius, center_y - radius, 
                                  center_x + radius, center_y + radius),
            material=MaterialType.GLASS,
            material_properties={},
            usage="meeting",
            priority=5,
            user_density=0.3,
            device_density=0.4,
            interference_sensitivity=1.3,
            polygon=conference_polygon
        )
        regions.append(conference)
        region_id += 1
        
        # Irregular open space
        open_space_polygon = [
            (18.0, 2.0), (35.0, 2.0), (35.0, 6.0), (30.0, 6.0), (30.0, 10.0), (25.0, 10.0), (25.0, 15.0), (18.0, 15.0)
        ]
        open_space = BuildingRegion(
            id=f"region_{region_id}",
            name="Irregular Open Space",
            region_type="room",
            boundary=RegionBoundary(18, 2, 35, 15),
            material=MaterialType.CARPET,
            material_properties={},
            usage="collaboration",
            priority=3,
            user_density=0.2,
            device_density=0.3,
            polygon=open_space_polygon
        )
        regions.append(open_space)
        region_id += 1
        
        # Triangular storage area
        storage_polygon = [
            (2.0, 15.0), (8.0, 15.0), (5.0, 20.0)
        ]
        storage = BuildingRegion(
            id=f"region_{region_id}",
            name="Triangular Storage",
            region_type="facility",
            boundary=RegionBoundary(2, 15, 8, 20),
            material=MaterialType.DRYWALL,
            material_properties={},
            usage="storage",
            priority=1,
            user_density=0.01,
            device_density=0.05,
            polygon=storage_polygon
        )
        regions.append(storage)
        region_id += 1
        
        # Hexagonal server room
        center_x, center_y = 35, 18
        radius = 4
        server_polygon = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            server_polygon.append((x, y))
        
        server_room = BuildingRegion(
            id=f"region_{region_id}",
            name="Hexagonal Server Room",
            region_type="facility",
            boundary=RegionBoundary(center_x - radius, center_y - radius,
                                  center_x + radius, center_y + radius),
            material=MaterialType.CONCRETE,
            material_properties={},
            usage="server",
            priority=2,
            user_density=0.0,
            device_density=0.1,
            polygon=server_polygon
        )
        regions.append(server_room)
        region_id += 1
        
        # Corridor with bends
        corridor_polygon = [
            (15.0, 8.0), (18.0, 8.0), (18.0, 10.0), (25.0, 10.0), (25.0, 12.0), (30.0, 12.0), (30.0, 15.0), (25.0, 15.0)
        ]
        corridor = BuildingRegion(
            id=f"region_{region_id}",
            name="Bent Corridor",
            region_type="corridor",
            boundary=RegionBoundary(15, 8, 30, 15),
            material=MaterialType.TILE,
            material_properties={},
            usage="circulation",
            priority=1,
            user_density=0.02,
            device_density=0.05,
            polygon=corridor_polygon
        )
        regions.append(corridor)
        region_id += 1
        
        self.regions = regions
        logging.info(f"Created {len(regions)} polygonal regions")
        
        # Generate materials grid
        self._generate_materials_grid()
        
        return regions

def parse_floor_plan_json(json_path: str) -> List[BuildingRegion]:
    """
    Parse a floor plan JSON file and return a list of BuildingRegion objects.
    The JSON should contain a list of regions, each with:
      - name
      - type
      - boundary: list of (x, y) tuples or bounding box
      - material
      - usage (optional)
      - priority (optional)
      - user_density (optional)
      - device_density (optional)
      - polygon: list of (x, y) tuples for polygonal regions (optional)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    regions = []
    for i, region in enumerate(data.get('regions', [])):
        # Handle polygon definition
        polygon = None
        if 'polygon' in region and isinstance(region['polygon'], list):
            polygon = [(float(pt[0]), float(pt[1])) for pt in region['polygon'] if len(pt) >= 2]
        
        # Handle boundary definition
        if 'boundary' in region and isinstance(region['boundary'], dict):
            b = region['boundary']
            boundary = RegionBoundary(
                x_min=b['x_min'], y_min=b['y_min'],
                x_max=b['x_max'], y_max=b['y_max'],
                z_min=b.get('z_min', 0.0), z_max=b.get('z_max', 3.0)
            )
        elif polygon:
            # Compute bounding box from polygon
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            boundary = RegionBoundary(
                x_min=min(xs), y_min=min(ys),
                x_max=max(xs), y_max=max(ys),
                z_min=region.get('z_min', 0.0), z_max=region.get('z_max', 3.0)
            )
        else:
            continue
        
        mat = MaterialType(region.get('material', 'air'))
        regions.append(BuildingRegion(
            id=region.get('id', f'region_{i+1}'),
            name=region.get('name', f'Region {i+1}'),
            region_type=region.get('type', 'room'),
            boundary=boundary,
            material=mat,
            material_properties=region.get('material_properties', {}),
            usage=region.get('usage', 'general'),
            priority=region.get('priority', 1),
            user_density=region.get('user_density', 0.1),
            device_density=region.get('device_density', 0.15),
            interference_sensitivity=region.get('interference_sensitivity', 1.0),
            coverage_requirement=region.get('coverage_requirement', 0.9),
            polygon=polygon
        ))
    return regions

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    px, py = x, y
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
    return inside

# Optionally, add a method to FloorPlanAnalyzer to use this parser
setattr(FloorPlanAnalyzer, 'parse_floor_plan_json', staticmethod(parse_floor_plan_json))
setattr(FloorPlanAnalyzer, 'point_in_polygon', staticmethod(point_in_polygon)) 