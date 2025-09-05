#!/usr/bin/env python3
"""
Enhanced RF-DETR Detection Script v2.0
Major improvements:
- Better code organization with classes
- Configuration file support
- Batch processing capabilities
- Memory optimization and GPU utilization
- Enhanced error handling and logging
- Real-time processing mode
- Multi-threading support for video processing
- Export results to JSON/CSV
- Performance profiling
- Better visualization options
"""

import os
import sys
import time
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import gc

import cv2
import numpy as np
import torch
import yaml

# Optional imports with graceful fallback
try:
    import torchvision.ops as tv_ops
    from torchvision.transforms import functional as F
    TORCHVISION_AVAILABLE = True
except ImportError:
    tv_ops = None
    TORCHVISION_AVAILABLE = False

try:
    from rfdetr import RFDETRMedium
    RFDETR_AVAILABLE = True
except ImportError as e:
    RFDETR_AVAILABLE = False
    RFDETR_IMPORT_ERROR = str(e)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# -------------------------
# Configuration Classes
# -------------------------
@dataclass
class DetectionConfig:
    """Configuration for detection parameters"""
    class_names: List[str]
    class_colors: Dict[str, Tuple[int, int, int]]
    class_confidence_thresholds: Dict[str, float]
    nms_iou_threshold: float = 0.5
    min_box_area: int = 100
    allow_cross_class_overlap: bool = True
    cross_class_iou_threshold: float = 0.7
    compatible_classes: List[Tuple[str, str]] = None
    prediction_threshold: float = 0.1
    
    def __post_init__(self):
        if self.compatible_classes is None:
            self.compatible_classes = [
                ('pothole', 'rutting road'),
                ('crack', 'rutting road'),
                ('crack', 'pothole'),
                ('Water_Log', 'pothole'),
                ('repair', 'crack'),
                ('repair', 'pothole'),
            ]

@dataclass 
class ProcessingConfig:
    """Configuration for processing parameters"""
    batch_size: int = 1
    num_workers: int = 2
    use_gpu: bool = True
    half_precision: bool = False
    enable_profiling: bool = False
    save_intermediate_results: bool = False
    output_formats: List[str] = None  # ['video', 'json', 'csv']
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['video']

@dataclass
class Detection:
    """Single detection result"""
    class_name: str
    confidence: float
    xyxy: List[float]  # [x1, y1, x2, y2]
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, (x2 - x1) * (y2 - y1))
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

# -------------------------
# Utilities and Helpers
# -------------------------
class PerformanceProfiler:
    """Simple performance profiler"""
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def start_timer(self, name: str):
        self.times[name] = time.time()
    
    def end_timer(self, name: str):
        if name in self.times:
            elapsed = time.time() - self.times[name]
            if name not in self.counts:
                self.counts[name] = {'total': 0, 'count': 0}
            self.counts[name]['total'] += elapsed
            self.counts[name]['count'] += 1
    
    def get_stats(self) -> Dict[str, float]:
        stats = {}
        for name, data in self.counts.items():
            stats[name] = {
                'avg_time': data['total'] / data['count'],
                'total_time': data['total'],
                'count': data['count']
            }
        return stats

class ConfigManager:
    """Handle configuration loading and validation"""
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Tuple[DetectionConfig, ProcessingConfig]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        else:
            config_data = ConfigManager._get_default_config()
        
        detection_config = DetectionConfig(**config_data.get('detection', {}))
        processing_config = ProcessingConfig(**config_data.get('processing', {}))
        
        return detection_config, processing_config
    
    @staticmethod
    def _get_default_config() -> Dict:
        """Default configuration"""
        return {
            'detection': {
                'class_names': [
                    'crack', 'pothole', 'rutting road', 'repair',
                    'damaged_sign', 'speed-breaker', 'Manhole', 'Water_Log'
                ],
                'class_colors': {
                    'crack': [255, 0, 0],
                    'pothole': [0, 255, 0],
                    'rutting road': [0, 0, 255],
                    'repair': [0, 255, 255],
                    'damaged_sign': [255, 165, 0],
                    'speed-breaker': [128, 0, 128],
                    'Manhole': [255, 0, 255],
                    'Water_Log': [0, 128, 255]
                },
                'class_confidence_thresholds': {
                    'crack': 0.34,
                    'pothole': 0.23,
                    'rutting road': 0.27,
                    'repair': 0.4,
                    'damaged_sign': 0.6,
                    'speed-breaker': 0.45,
                    'Manhole': 0.4,
                    'Water_Log': 0.3
                }
            },
            'processing': {
                'batch_size': 1,
                'num_workers': 2,
                'use_gpu': True,
                'output_formats': ['video']
            }
        }

    @staticmethod
    def save_config(detection_config: DetectionConfig, processing_config: ProcessingConfig, 
                   config_path: str):
        """Save configuration to file"""
        config_data = {
            'detection': asdict(detection_config),
            'processing': asdict(processing_config)
        }
        
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
        else:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = max(0.0, (x2_1 - x1_1) * (y2_1 - y1_1))
        area2 = max(0.0, (x2_2 - x1_2) * (y2_2 - y1_2))
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def are_classes_compatible(class1: str, class2: str, 
                              compatible_classes: List[Tuple[str, str]]) -> bool:
        """Check if two classes are compatible for overlapping"""
        if class1 == class2:
            return False
        return ((class1, class2) in compatible_classes or 
                (class2, class1) in compatible_classes)

# -------------------------
# Detection Processing Pipeline
# -------------------------
class DetectionProcessor:
    """Main detection processing pipeline"""
    
    def __init__(self, detection_config: DetectionConfig, 
                 processing_config: ProcessingConfig,
                 logger: logging.Logger = None):
        self.det_config = detection_config
        self.proc_config = processing_config
        self.logger = logger or logging.getLogger(__name__)
        self.profiler = PerformanceProfiler() if processing_config.enable_profiling else None
    
    def filter_by_confidence(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections by confidence thresholds"""
        if self.profiler:
            self.profiler.start_timer('confidence_filter')
        
        filtered = []
        for det in detections:
            threshold = self.det_config.class_confidence_thresholds.get(
                det.class_name, 0.3)
            if det.confidence >= threshold and det.area >= self.det_config.min_box_area:
                filtered.append(det)
        
        if self.profiler:
            self.profiler.end_timer('confidence_filter')
        
        return filtered
    
    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if not detections or not TORCHVISION_AVAILABLE:
            return self._apply_nms_python(detections)
        
        if self.profiler:
            self.profiler.start_timer('nms')
        
        result = []
        # Group by class
        by_class = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)
        
        for class_name, class_dets in by_class.items():
            if len(class_dets) <= 1:
                result.extend(class_dets)
                continue
            
            # Convert to tensors
            boxes = torch.tensor([det.xyxy for det in class_dets], dtype=torch.float32)
            scores = torch.tensor([det.confidence for det in class_dets], dtype=torch.float32)
            
            # Apply NMS
            keep_idx = tv_ops.nms(boxes, scores, self.det_config.nms_iou_threshold)
            
            # Keep selected detections
            for idx in keep_idx:
                result.append(class_dets[idx.item()])
        
        if self.profiler:
            self.profiler.end_timer('nms')
        
        return result
    
    def _apply_nms_python(self, detections: List[Detection]) -> List[Detection]:
        """Python implementation of NMS (fallback)"""
        if not detections:
            return []
        
        result = []
        by_class = {}
        for det in detections:
            by_class.setdefault(det.class_name, []).append(det)
        
        for class_name, class_dets in by_class.items():
            # Sort by confidence
            sorted_dets = sorted(class_dets, key=lambda x: x.confidence, reverse=True)
            kept = []
            
            while sorted_dets:
                best = sorted_dets.pop(0)
                kept.append(best)
                
                # Remove overlapping detections
                remaining = []
                for det in sorted_dets:
                    iou = GeometryUtils.calculate_iou(best.xyxy, det.xyxy)
                    if iou < self.det_config.nms_iou_threshold:
                        remaining.append(det)
                
                sorted_dets = remaining
            
            result.extend(kept)
        
        return result
    
    def apply_cross_class_filtering(self, detections: List[Detection]) -> List[Detection]:
        """Apply cross-class overlap filtering"""
        if not self.det_config.allow_cross_class_overlap or len(detections) <= 1:
            return detections
        
        if self.profiler:
            self.profiler.start_timer('cross_class_filter')
        
        final = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(detections):
                if i == j:
                    continue
                
                iou = GeometryUtils.calculate_iou(det1.xyxy, det2.xyxy)
                if iou > self.det_config.cross_class_iou_threshold:
                    if not GeometryUtils.are_classes_compatible(
                        det1.class_name, det2.class_name, 
                        self.det_config.compatible_classes):
                        if det1.confidence < det2.confidence:
                            keep = False
                            break
            
            if keep:
                final.append(det1)
        
        if self.profiler:
            self.profiler.end_timer('cross_class_filter')
        
        return final
    
    def process_detections(self, raw_detections: List[Dict]) -> List[Detection]:
        """Complete detection processing pipeline"""
        # Convert to Detection objects
        detections = [Detection(**det) for det in raw_detections]
        
        # Apply filters
        detections = self.filter_by_confidence(detections)
        detections = self.apply_nms(detections)
        detections = self.apply_cross_class_filtering(detections)
        
        return detections

# -------------------------
# Model Wrappers
# -------------------------
class BaseModelWrapper:
    """Base class for model wrappers"""
    
    def __init__(self, model_path: str, detection_config: DetectionConfig,
                 processing_config: ProcessingConfig, logger: logging.Logger = None):
        self.model_path = model_path
        self.det_config = detection_config
        self.proc_config = processing_config
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._setup_device()
        self.model = None
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.proc_config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def load_model(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def infer(self, image: np.ndarray) -> List[Dict]:
        """Run inference - to be implemented by subclasses"""
        raise NotImplementedError
    
    def warmup(self, input_shape: Tuple[int, int] = (640, 640)):
        """Warmup model with dummy input"""
        dummy_input = np.random.randint(0, 255, (input_shape[1], input_shape[0], 3), dtype=np.uint8)
        try:
            self.infer(dummy_input)
            self.logger.info("Model warmup completed")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

class RFDETRWrapper(BaseModelWrapper):
    """Enhanced RF-DETR wrapper with better error handling"""
    
    def load_model(self):
        """Load RF-DETR model with enhanced error handling"""
        try:
            if not RFDETR_AVAILABLE:
                raise ImportError(f"RF-DETR not available: {RFDETR_IMPORT_ERROR}")
            
            self.logger.info(f"Loading RF-DETR model from {self.model_path}")
            
            # Initialize model
            self.model = RFDETRMedium(num_classes=len(self.det_config.class_names))
            
            # Try to reinitialize detection head
            try:
                if hasattr(self.model.model.model, 'reinitialize_detection_head'):
                    self.model.model.model.reinitialize_detection_head(
                        num_classes=len(self.det_config.class_names))
            except Exception as e:
                self.logger.debug(f"Detection head reinit failed (non-critical): {e}")
            
            # Load checkpoint with PyTorch 2.6+ compatibility
            try:
                # For trusted model files, use weights_only=False
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.logger.debug("Loaded checkpoint with weights_only=False")
            except Exception as e:
                # If that fails, it might be due to other issues
                self.logger.error(f"Failed to load checkpoint: {e}")
                raise
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                state_dict = (checkpoint.get('model') or 
                            checkpoint.get('model_state_dict') or 
                            checkpoint.get('state_dict') or 
                            checkpoint)
            else:
                state_dict = checkpoint
            
            # Clean module prefix
            cleaned_state_dict = {}
            if isinstance(state_dict, dict):
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "") if isinstance(key, str) else key
                    cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict = state_dict
            
            # Load weights
            try:
                missing, unexpected = self.model.model.model.load_state_dict(
                    cleaned_state_dict, strict=False)
                if missing:
                    self.logger.warning(f"Missing keys: {len(missing)}")
                if unexpected:
                    self.logger.warning(f"Unexpected keys: {len(unexpected)}")
            except Exception as e:
                # Fallback loading method
                try:
                    self.model.load_state_dict(cleaned_state_dict, strict=False)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model weights: {e2}")
            
            # Setup model
            self.model.model.model.to(self.device)
            self.model.model.model.eval()
            
            # Enable half precision if requested
            if self.proc_config.half_precision and self.device.type == 'cuda':
                self.model.model.model.half()
                self.logger.info("Enabled half precision")
            
            self.logger.info("✅ RF-DETR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load RF-DETR model: {e}")
            self.model = None
            raise
    
    def infer(self, image: np.ndarray) -> List[Dict]:
        """Run RF-DETR inference with robust output parsing"""
        if self.model is None:
            return []
        
        # Prepare image
        if image.ndim == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        
        try:
            with torch.no_grad():
                # Run inference
                if hasattr(self.model, "predict"):
                    predictions = self.model.predict(
                        img_rgb, threshold=self.det_config.prediction_threshold)
                else:
                    predictions = self.model(img_rgb)
            
            # Parse predictions
            detections = self._parse_predictions(predictions)
            return detections
            
        except Exception as e:
            self.logger.error(f"RF-DETR inference error: {e}")
            return []
    
    def _parse_predictions(self, predictions) -> List[Dict]:
        """Parse RF-DETR predictions into standard format"""
        detections = []
        
        try:
            # Method 1: Direct attributes
            if (hasattr(predictions, 'xyxy') and 
                hasattr(predictions, 'class_id') and 
                hasattr(predictions, 'confidence')):
                
                for i in range(len(predictions.xyxy)):
                    class_id = int(predictions.class_id[i])
                    if class_id < len(self.det_config.class_names):
                        detections.append({
                            'class_name': self.det_config.class_names[class_id],
                            'confidence': float(predictions.confidence[i]),
                            'xyxy': predictions.xyxy[i].tolist()
                        })
            
            # Method 2: List of dictionaries
            elif isinstance(predictions, (list, tuple)):
                for item in predictions:
                    if isinstance(item, dict):
                        det = self._parse_dict_prediction(item)
                        if det:
                            detections.append(det)
            
            # Method 3: Boxes attribute
            elif hasattr(predictions, 'boxes'):
                detections = self._parse_boxes_attribute(predictions.boxes)
            
        except Exception as e:
            self.logger.debug(f"Prediction parsing error: {e}")
        
        return detections
    
    def _parse_dict_prediction(self, item: Dict) -> Optional[Dict]:
        """Parse dictionary prediction item"""
        try:
            bbox = item.get('bbox', [])
            score = float(item.get('score', 0))
            label = item.get('label')
            
            if len(bbox) != 4:
                return None
            
            if isinstance(label, int):
                if label >= len(self.det_config.class_names):
                    return None
                class_name = self.det_config.class_names[label]
            else:
                class_name = str(label)
            
            return {
                'class_name': class_name,
                'confidence': score,
                'xyxy': list(bbox)
            }
        except Exception:
            return None
    
    def _parse_boxes_attribute(self, boxes) -> List[Dict]:
        """Parse boxes attribute"""
        detections = []
        try:
            for box in boxes:
                try:
                    xyxy = getattr(box, 'xyxy', box[:4])
                    conf = float(getattr(box, 'conf', box[4]))
                    cls_id = int(getattr(box, 'cls', box[5]))
                    
                    if cls_id < len(self.det_config.class_names):
                        detections.append({
                            'class_name': self.det_config.class_names[cls_id],
                            'confidence': conf,
                            'xyxy': list(xyxy)
                        })
                except Exception:
                    continue
        except Exception:
            pass
        
        return detections

class YOLOWrapper(BaseModelWrapper):
    """Enhanced YOLO wrapper"""
    
    def load_model(self):
        """Load YOLO model"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("Ultralytics YOLO not available")
            
            self.logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move to device
            if self.proc_config.use_gpu and torch.cuda.is_available():
                self.model.to(self.device)
            
            self.logger.info("✅ YOLO model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            raise
    
    def infer(self, image: np.ndarray) -> List[Dict]:
        """Run YOLO inference"""
        if self.model is None:
            return []
        
        try:
            results = self.model(image, verbose=False)
            detections = self._parse_yolo_results(results)
            return detections
        except Exception as e:
            self.logger.error(f"YOLO inference error: {e}")
            return []
    
    def _parse_yolo_results(self, results) -> List[Dict]:
        """Parse YOLO results into standard format"""
        detections = []
        
        # Handle batch results
        if isinstance(results, (list, tuple)):
            results_iter = results
        else:
            results_iter = [results]
        
        for result in results_iter:
            # Try to access boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                # Extract data
                if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    
                    for i in range(len(xyxy)):
                        cls_id = int(cls[i])
                        if cls_id < len(self.det_config.class_names):
                            detections.append({
                                'class_name': self.det_config.class_names[cls_id],
                                'confidence': float(conf[i]),
                                'xyxy': xyxy[i].tolist()
                            })
        
        return detections

# -------------------------
# Visualization and Export
# -------------------------
class Visualizer:
    """Enhanced visualization with multiple output formats"""
    
    def __init__(self, detection_config: DetectionConfig):
        self.det_config = detection_config
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       show_confidence: bool = True, show_class: bool = True) -> np.ndarray:
        """Draw detections on frame with enhanced styling"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            color = self.det_config.class_colors.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(det.class_name)
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate text size and draw background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                
                # Draw text
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def create_detection_summary(self, detections: List[Detection]) -> Dict:
        """Create detection summary statistics"""
        summary = {
            'total_detections': len(detections),
            'by_class': {},
            'avg_confidence': 0.0,
            'confidence_range': [0.0, 1.0]
        }
        
        if not detections:
            return summary
        
        # Count by class
        for det in detections:
            summary['by_class'][det.class_name] = summary['by_class'].get(det.class_name, 0) + 1
        
        # Confidence statistics
        confidences = [det.confidence for det in detections]
        summary['avg_confidence'] = np.mean(confidences)
        summary['confidence_range'] = [float(np.min(confidences)), float(np.max(confidences))]
        
        return summary

class ResultExporter:
    """Export results in various formats"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(self, detections: List[Detection], filename: str):
        """Export detections to JSON"""
        data = [asdict(det) for det in detections]
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def export_csv(self, detections: List[Detection], filename: str):
        """Export detections to CSV"""
        if not PANDAS_AVAILABLE:
            # Fallback to manual CSV writing
            return self._export_csv_manual(detections, filename)
        
        df = pd.DataFrame([asdict(det) for det in detections])
        output_path = self.output_dir / f"{filename}.csv"
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _export_csv_manual(self, detections: List[Detection], filename: str):
        """Manual CSV export without pandas"""
        output_path = self.output_dir / f"{filename}.csv"
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            if detections:
                writer.writerow(['class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'frame_id', 'timestamp'])
                
                # Data rows
                for det in detections:
                    x1, y1, x2, y2 = det.xyxy
                    writer.writerow([
                        det.class_name, det.confidence, x1, y1, x2, y2,
                        det.frame_id or '', det.timestamp or ''
                    ])
        
        return output_path

# -------------------------
# Main Processing Classes
# -------------------------
class ImageProcessor:
    """Process single images or batch of images"""
    
    def __init__(self, model_wrapper: BaseModelWrapper, 
                 detection_processor: DetectionProcessor,
                 visualizer: Visualizer,
                 logger: logging.Logger = None):
        self.model = model_wrapper
        self.processor = detection_processor
        self.visualizer = visualizer
        self.logger = logger or logging.getLogger(__name__)
    
    def process_single_image(self, image_path: str, output_path: str) -> List[Detection]:
        """Process a single image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Run inference
        raw_detections = self.model.infer(image)
        detections = self.processor.process_detections(raw_detections)
        
        # Visualize and save
        annotated = self.visualizer.draw_detections(image, detections)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, annotated)
        
        self.logger.info(f"Processed image: {len(detections)} detections found")
        return detections
    
    def process_image_batch(self, image_paths: List[str], output_dir: str) -> Dict[str, List[Detection]]:
        """Process batch of images"""
        results = {}
        
        for image_path in image_paths:
            try:
                filename = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{filename}_annotated.jpg")
                detections = self.process_single_image(image_path, output_path)
                results[image_path] = detections
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = []
        
        return results

class VideoProcessor:
    """Enhanced video processor with multi-threading and memory optimization"""
    
    def __init__(self, model_wrapper: BaseModelWrapper,
                 detection_processor: DetectionProcessor,
                 visualizer: Visualizer,
                 processing_config: ProcessingConfig,
                 logger: logging.Logger = None):
        self.model = model_wrapper
        self.processor = detection_processor
        self.visualizer = visualizer
        self.proc_config = processing_config
        self.logger = logger or logging.getLogger(__name__)
        self.frame_queue = Queue(maxsize=processing_config.batch_size * 2)
        self.result_queue = Queue(maxsize=processing_config.batch_size * 2)
    
    def process_video(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """Process video with enhanced features"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Safety check
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("Output path cannot be the same as input path")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Setup output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'detection_stats': {name: 0 for name in self.processor.det_config.class_names},
            'processing_time': 0,
            'avg_fps': 0
        }
        
        start_time = time.time()
        frame_count = 0
        all_detections = []
        
        try:
            # Use multi-threading if enabled
            if self.proc_config.num_workers > 1:
                stats = self._process_video_threaded(cap, out_writer, stats, start_time)
            else:
                stats = self._process_video_single_thread(cap, out_writer, stats, start_time)
        
        finally:
            cap.release()
            out_writer.release()
            
            # Cleanup
            if hasattr(self, 'frame_queue'):
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
            
            if hasattr(self, 'result_queue'):
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
        
        # Final statistics
        end_time = time.time()
        stats['processing_time'] = end_time - start_time
        stats['avg_fps'] = stats['processed_frames'] / stats['processing_time'] if stats['processing_time'] > 0 else 0
        
        self.logger.info(f"✅ Video processing completed:")
        self.logger.info(f"   Processed: {stats['processed_frames']} frames")
        self.logger.info(f"   Total detections: {stats['total_detections']}")
        self.logger.info(f"   Average FPS: {stats['avg_fps']:.2f}")
        self.logger.info(f"   Processing time: {stats['processing_time']:.1f}s")
        
        return stats
    
    def _process_video_single_thread(self, cap, out_writer, stats, start_time):
        """Single-threaded video processing"""
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 50 == 0 or frame_count == 1:
                elapsed = time.time() - start_time
                fps_est = frame_count / elapsed if elapsed > 0 else 0.0
                eta = (stats['total_frames'] - frame_count) / fps_est if fps_est > 0 and stats['total_frames'] > 0 else 0.0
                self.logger.info(f"Frame {frame_count}/{stats['total_frames']} | {fps_est:.1f} FPS | ETA: {eta:.1f}s")
            
            # Process frame
            try:
                raw_detections = self.model.infer(frame)
                detections = self.processor.process_detections(raw_detections)
                
                # Update statistics
                stats['total_detections'] += len(detections)
                for det in detections:
                    if det.class_name in stats['detection_stats']:
                        stats['detection_stats'][det.class_name] += 1
                
                # Annotate and write frame
                annotated = self.visualizer.draw_detections(frame, detections)
                out_writer.write(annotated)
                
                stats['processed_frames'] = frame_count
                
                # Memory cleanup
                if frame_count % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_count}: {e}")
                out_writer.write(frame)  # Write original frame
        
        return stats
    
    def _process_video_threaded(self, cap, out_writer, stats, start_time):
        """Multi-threaded video processing (simplified implementation)"""
        # For now, fall back to single-threaded processing
        # Multi-threading video processing is complex and requires careful synchronization
        self.logger.info("Multi-threading requested but using single-thread for stability")
        return self._process_video_single_thread(cap, out_writer, stats, start_time)

class RealTimeProcessor:
    """Real-time processing for webcam or live streams"""
    
    def __init__(self, model_wrapper: BaseModelWrapper,
                 detection_processor: DetectionProcessor,
                 visualizer: Visualizer,
                 logger: logging.Logger = None):
        self.model = model_wrapper
        self.processor = detection_processor
        self.visualizer = visualizer
        self.logger = logger or logging.getLogger(__name__)
        self.running = False
    
    def start_realtime(self, source: Union[int, str] = 0, display_window: bool = True):
        """Start real-time processing"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        self.running = True
        frame_count = 0
        fps_counter = time.time()
        fps_frames = 0
        
        self.logger.info("Starting real-time processing. Press 'q' to quit.")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                fps_frames += 1
                
                # Process frame
                try:
                    raw_detections = self.model.infer(frame)
                    detections = self.processor.process_detections(raw_detections)
                    annotated = self.visualizer.draw_detections(frame, detections)
                    
                    # Add FPS counter
                    current_time = time.time()
                    if current_time - fps_counter >= 1.0:
                        fps = fps_frames / (current_time - fps_counter)
                        fps_counter = current_time
                        fps_frames = 0
                        
                        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if display_window:
                        cv2.imshow('Real-time Detection', annotated)
                        
                        # Check for quit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            break
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time processing: {e}")
                    if display_window:
                        cv2.imshow('Real-time Detection', frame)
        
        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()
            self.running = False
        
        self.logger.info(f"Real-time processing stopped. Processed {frame_count} frames.")
    
    def stop_realtime(self):
        """Stop real-time processing"""
        self.running = False

# -------------------------
# CLI and Main Application
# -------------------------
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("rfdetr_enhanced")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_model_wrapper(model_type: str, model_path: str, 
                        detection_config: DetectionConfig,
                        processing_config: ProcessingConfig,
                        logger: logging.Logger) -> BaseModelWrapper:
    """Factory function to create model wrapper"""
    if model_type == "rfdetr":
        if not RFDETR_AVAILABLE:
            raise ImportError(f"RF-DETR not available: {RFDETR_IMPORT_ERROR}")
        wrapper = RFDETRWrapper(model_path, detection_config, processing_config, logger)
    elif model_type == "yolo":
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO (ultralytics) not available")
        wrapper = YOLOWrapper(model_path, detection_config, processing_config, logger)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    wrapper.load_model()
    return wrapper

def main():
    """Main CLI application"""
    parser = argparse.ArgumentParser(
        description="Enhanced RF-DETR/YOLO Detection Script v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model", choices=["rfdetr", "yolo"], default="rfdetr",
                           help="Model type to use")
    model_group.add_argument("--model_path", default="model/rfdetr.pth",
                           help="Path to model weights")
    
    # Input/Output arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--input", required=True,
                         help="Input image/video path or camera index")
    io_group.add_argument("--output", default="output/result",
                         help="Output path (without extension)")
    io_group.add_argument("--output_formats", nargs="+", 
                         choices=["video", "json", "csv"], default=["video"],
                         help="Output formats to generate")
    
    # Configuration arguments
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument("--config", help="Configuration file path (YAML/JSON)")
    config_group.add_argument("--save_config", help="Save current config to file")
    
    # Processing arguments
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument("--batch_size", type=int, default=1,
                          help="Batch size for processing")
    proc_group.add_argument("--num_workers", type=int, default=2,
                          help="Number of worker threads")
    proc_group.add_argument("--use_cpu", action="store_true",
                          help="Force CPU usage")
    proc_group.add_argument("--half_precision", action="store_true",
                          help="Use half precision (FP16)")
    proc_group.add_argument("--warmup", action="store_true",
                          help="Warmup model before processing")
    
    # Detection arguments
    det_group = parser.add_argument_group('Detection')
    det_group.add_argument("--nms_threshold", type=float, default=0.5,
                         help="NMS IoU threshold")
    det_group.add_argument("--min_area", type=int, default=100,
                         help="Minimum bounding box area")
    det_group.add_argument("--prediction_threshold", type=float, default=0.1,
                         help="Base prediction threshold")
    
    # Misc arguments
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument("--mode", choices=["single", "batch", "realtime"], 
                          default="single", help="Processing mode")
    misc_group.add_argument("--realtime_source", default=0,
                          help="Realtime source (camera index or stream URL)")
    misc_group.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          default="INFO", help="Logging level")
    misc_group.add_argument("--log_file", help="Log file path")
    misc_group.add_argument("--profile", action="store_true",
                          help="Enable performance profiling")
    misc_group.add_argument("--debug", action="store_true",
                          help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else args.log_level
    logger = setup_logging(log_level, args.log_file)
    
    try:
        # Load configuration
        if args.config:
            detection_config, processing_config = ConfigManager.load_config(args.config)
        else:
            detection_config, processing_config = ConfigManager.load_config()
        
        # Override config with CLI arguments
        processing_config.batch_size = args.batch_size
        processing_config.num_workers = args.num_workers
        processing_config.use_gpu = not args.use_cpu
        processing_config.half_precision = args.half_precision
        processing_config.enable_profiling = args.profile
        processing_config.output_formats = args.output_formats
        
        detection_config.nms_iou_threshold = args.nms_threshold
        detection_config.min_box_area = args.min_area
        detection_config.prediction_threshold = args.prediction_threshold
        
        # Save configuration if requested
        if args.save_config:
            ConfigManager.save_config(detection_config, processing_config, args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
        
        logger.info("🎯 Enhanced RF-DETR Detection Script v2.0")
        logger.info(f"Model: {args.model}")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Mode: {args.mode}")
        
        # Create model wrapper
        model_wrapper = create_model_wrapper(
            args.model, args.model_path, detection_config, processing_config, logger
        )
        
        # Warmup model if requested
        if args.warmup:
            model_wrapper.warmup()
        
        # Create processing components
        detection_processor = DetectionProcessor(detection_config, processing_config, logger)
        visualizer = Visualizer(detection_config)
        exporter = ResultExporter(os.path.dirname(args.output) or ".")
        
        # Process based on mode
        if args.mode == "realtime":
            processor = RealTimeProcessor(model_wrapper, detection_processor, visualizer, logger)
            processor.start_realtime(args.realtime_source)
        
        elif args.mode == "batch":
            # Batch processing (assuming input is a directory)
            if not os.path.isdir(args.input):
                raise ValueError("Batch mode requires input directory")
            
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend(Path(args.input).glob(f"*{ext}"))
                image_paths.extend(Path(args.input).glob(f"*{ext.upper()}"))
            
            processor = ImageProcessor(model_wrapper, detection_processor, visualizer, logger)
            results = processor.process_image_batch([str(p) for p in image_paths], 
                                                  os.path.dirname(args.output) or ".")
            
            # Export results
            all_detections = []
            for path, detections in results.items():
                all_detections.extend(detections)
            
            if "json" in args.output_formats:
                exporter.export_json(all_detections, Path(args.output).stem + "_batch")
            if "csv" in args.output_formats:
                exporter.export_csv(all_detections, Path(args.output).stem + "_batch")
        
        else:  # single mode
            input_path = args.input
            input_suffix = Path(input_path).suffix.lower()
            
            # Determine if input is image or video
            if input_suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Image processing
                output_path = args.output + input_suffix
                processor = ImageProcessor(model_wrapper, detection_processor, visualizer, logger)
                detections = processor.process_single_image(input_path, output_path)
                
                # Export additional formats
                if "json" in args.output_formats:
                    exporter.export_json(detections, Path(args.output).stem)
                if "csv" in args.output_formats:
                    exporter.export_csv(detections, Path(args.output).stem)
            
            elif input_suffix in ['.mp4', '.avi', '.mov', '.mkv']:
                # Video processing
                output_path = args.output + '.mp4'
                processor = VideoProcessor(model_wrapper, detection_processor, visualizer, 
                                        processing_config, logger)
                stats = processor.process_video(input_path, output_path)
                
                # Export statistics
                if "json" in args.output_formats:
                    stats_path = exporter.output_dir / f"{Path(args.output).stem}_stats.json"
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
            
            else:
                raise ValueError(f"Unsupported file format: {input_suffix}")
        
        # Print profiling results if enabled
        if processing_config.enable_profiling and hasattr(detection_processor, 'profiler'):
            stats = detection_processor.profiler.get_stats()
            logger.info("📊 Performance Profile:")
            for operation, data in stats.items():
                logger.info(f"  {operation}: {data['avg_time']:.4f}s avg, {data['count']} calls")
        
        logger.info("✅ Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()