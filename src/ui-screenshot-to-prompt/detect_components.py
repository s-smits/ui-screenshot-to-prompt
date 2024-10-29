import cv2
import easyocr
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from logging import getLogger
from config import (
    MIN_REGION_WIDTH_SIMPLE, 
    MIN_REGION_HEIGHT_SIMPLE, 
    MIN_COMPONENT_WIDTH_ADVANCED, 
    MIN_COMPONENT_HEIGHT_ADVANCED,
    MAX_UI_COMPONENTS,
    get_detection_term,
)


logger = getLogger(__name__)

@dataclass
class UIDetection:
    """Data class representing a detected UI element with its properties"""
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0
    text: str = ""
    
    def to_dict(self) -> Dict:
        """Convert detection to dictionary representation"""
        return {
            'type': self.type,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'text': self.text
        }

class DetectorBase:
    """Shared base functionality for all detectors"""
    
    @staticmethod
    def create_detection(
        bbox: Tuple[int, int, int, int], 
        detection_type: str = "unknown",
        location: str = "",
        confidence: float = 1.0,
        text: str = ""
    ) -> UIDetection:
        """Create a detection with consistent naming"""
        detection_term = get_detection_term()
        
        return UIDetection(
            bbox=bbox,
            type=f"{detection_term}_{detection_type}" if detection_type != "unknown" else detection_term,
            text=text or location,
            confidence=confidence
        )

    def visualize_detections(self, image: np.ndarray, detections: List[UIDetection], output_path: str):
        """Visualize detections on the image"""
        # Make a copy of the image to draw on
        viz_image = image.copy()
        
        # Draw each detection
        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            
            # Draw rectangle
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label = f"{i}: {detection.type}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Get text size to position background rectangle
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(viz_image, 
                        (x, y - text_height - 5),
                        (x + text_width, y),
                        color,
                        -1)  # Filled rectangle
            
            # Draw text
            cv2.putText(viz_image,
                       label,
                       (x, y - 5),
                       font,
                       font_scale,
                       (0, 0, 0),  # Black text
                       font_thickness)
        
        # Save the visualization
        cv2.imwrite(output_path, viz_image)

class BasicRegionDetector(DetectorBase):
    """Base class for grid-based region detection"""
    
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.height, self.width = self.image.shape[:2]
        
    def get_grid_pattern(self) -> Tuple[Tuple[int, int], List[str]]:
        """Return grid size and location names based on image dimensions"""
        aspect_ratio = self.width / self.height if self.height != 0 else 1

        if self.width < MIN_REGION_WIDTH_SIMPLE and self.height < MIN_REGION_HEIGHT_SIMPLE:
            return (1, 1), ['full']
        elif aspect_ratio >= 3:
            return (1, 3), ['left side', 'center portion', 'right side']
        elif aspect_ratio <= 0.33:
            return (3, 1), ['upper portion', 'middle portion', 'lower portion']
        elif aspect_ratio >= 2:
            return (2, 3), [
                'upper left section', 'upper center section', 'upper right section',
                'lower left section', 'lower center section', 'lower right section'
            ]
        elif aspect_ratio <= 0.5:
            return (3, 2), [
                'top left area', 'top right area',
                'middle left area', 'middle right area',
                'bottom left area', 'bottom right area'
            ]
        elif self.width >= 600 and self.height >= 600:
            return (3, 3), [
                'top left region', 'top center region', 'top right region',
                'middle left region', 'center region', 'middle right region',
                'bottom left region', 'bottom center region', 'bottom right region'
            ]
        elif self.width >= 400 and self.height >= 400:
            return (2, 2), [
                'top left quadrant', 'top right quadrant',
                'bottom left quadrant', 'bottom right quadrant'
            ]
        else:
            if self.width > self.height:
                return (1, 2), ['left half', 'right half']
            else:
                return (2, 1), ['upper half', 'lower half']
    
    def get_components(self) -> List[UIDetection]:
        """Get grid-based regions as components"""
        grid_size, locations = self.get_grid_pattern()
        rows, cols = grid_size
        
        cell_height = self.height // rows
        cell_width = self.width // cols
        
        regions = []
        idx = 0
        for row in range(rows):
            for col in range(cols):
                x = col * cell_width
                y = row * cell_height
                w = cell_width
                h = cell_height
                
                # Adjust last row/column to account for rounding
                if col == cols - 1:
                    w = self.width - x
                if row == rows - 1:
                    h = self.height - y
                    
                regions.append(self.create_detection(
                    bbox=(x, y, w, h),
                    detection_type='region',
                    text=locations[idx]
                ))
                idx += 1
                
        return regions

class ComponentDetectorBase(DetectorBase):
    """Base class for UI component detection"""
    
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.height, self.width = self.image.shape[:2]
        
    def get_components(self) -> List[UIDetection]:
        """Get detected UI components - must be implemented by derived classes"""
        raise NotImplementedError()

class AdvancedDetector(ComponentDetectorBase):
    """Advanced detector using OCR and traditional CV approaches"""
    def __init__(self, image_path: str, max_components: int, min_width: int, min_height: int):
        super().__init__(image_path)
        self.image_path = image_path
        self.min_width = min_width
        self.min_height = min_height
        self.MIN_DETECTION_AREA = self.min_width * self.min_height
        self.max_ui_components = max_components
        print("Initializing EasyOCR (this may download models on first run)...")
        self.reader = easyocr.Reader(['en'], download_enabled=True)
        print("EasyOCR initialization complete!")
    
    def detect_edges(self) -> np.ndarray:
        """Detect edges in the image using Canny edge detector"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)
    
    def classify_component(self, aspect_ratio: float, area: float) -> str:
        """Classify UI component based on its properties"""
        if aspect_ratio > 3:
            return "text_input"
        elif 0.9 <= aspect_ratio <= 1.1:
            return "button" if area < 1000 else "image"
        elif aspect_ratio < 0.5:
            return "dropdown"
        return "container"
    
    def get_components(self) -> List[UIDetection]:
        """Get detected UI components using advanced CV and OCR"""
        # Detect edges and find contours
        edges = self.detect_edges()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create list to store all potential components
        potential_components = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out components that are too small
            if w < self.min_width or h < self.min_height:
                continue
                
            area = cv2.contourArea(contour)
            potential_components.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': w / float(h)
            })
        
        # Sort components by area (largest first)
        potential_components.sort(key=lambda x: x['area'], reverse=True)
        
        # Filter overlapping components (keep only the largest)
        filtered_components = []
        for comp in potential_components:
            x1, y1, w1, h1 = comp['bbox']
            
            # Check if this component significantly overlaps with any larger component
            is_overlapping = False
            for existing in filtered_components:
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    smaller_area = min(w1 * h1, w2 * h2)
                    if intersection_area / smaller_area > 0.5:  # 50% overlap threshold
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered_components.append(comp)
                
            # Stop if we have enough components
            if len(filtered_components) >= self.max_ui_components:
                break
        
        # Convert filtered components to UIDetections
        components = []
        for comp in filtered_components[:self.max_ui_components]:
            x, y, w, h = comp['bbox']
            
            # Classify component
            component_type = self.classify_component(comp['aspect_ratio'], comp['area'])
            
            # Extract text using OCR if needed
            text = ""
            if component_type in ["text_input", "button"]:
                try:
                    roi = self.image[y:y+h, x:x+w]
                    results = self.reader.readtext(roi)
                    text = " ".join([result[1] for result in results])
                except Exception as e:
                    logger.warning(f"OCR failed for component: {e}")
            
            # Create detection with confidence based on area
            confidence = min(comp['area'] / (self.width * self.height), 1.0)
            
            components.append(self.create_detection(
                bbox=(x, y, w, h),
                detection_type=component_type,
                confidence=confidence,
                text=text
            ))
        
        return components[:self.max_ui_components]

def create_detector(
    method: str, 
    image_path: str,
    max_components: int = MAX_UI_COMPONENTS,
    min_width: int = MIN_COMPONENT_WIDTH_ADVANCED,
    min_height: int = MIN_COMPONENT_HEIGHT_ADVANCED
) -> DetectorBase:
    """Factory function to create appropriate detector based on method"""
    if method.lower() == "basic":
        return BasicRegionDetector(image_path)
    elif method.lower() == "advanced":
        return AdvancedDetector(
            image_path,
            max_components=max_components,
            min_width=min_width,
            min_height=min_height
        )
    else:
        raise ValueError(f"Unknown detection method: {method}")
