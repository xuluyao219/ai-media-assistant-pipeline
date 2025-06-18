#!/usr/bin/env python
"""
Multimodal AI Media Analysis System
Supports RTX 3080+, integrates CLIP, YOLOv8, Whisper V3
"""

import os
import json
import torch
import clip
import numpy as np
from pathlib import Path
import cv2
import logging
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import platform
import subprocess

def setup_ffmpeg():
    """自动设置FFmpeg路径"""
    if platform.system() == 'Windows':
        # Windows常见FFmpeg位置
        possible_paths = [
            r"C:\ffmpeg\bin",
            os.path.expanduser(r"~\scoop\shims"),
            os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin"),
            r"C:\ProgramData\chocolatey\bin",
            r"C:\Program Files\ffmpeg\bin"
        ]

        # 查找ffmpeg.exe
        for path in possible_paths:
            ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
            if os.path.exists(ffmpeg_exe):
                os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
                return True

        # 尝试使用where命令
        try:
            result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout:
                ffmpeg_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(ffmpeg_path):
                    ffmpeg_dir = os.path.dirname(ffmpeg_path)
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                    return True
        except:
            pass

    return False

# 启动时设置FFmpeg
if not setup_ffmpeg():
    print("Warning: FFmpeg not found in common locations")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MediaAnalysisResult:
    """Analysis result data class"""
    video_path: str
    scenes: List[Dict]
    objects: List[Dict]
    transcripts: Dict
    embeddings: np.ndarray
    processing_time: float

class MultiModalAnalyzer:
    """multimodal analyzer"""
    
    def __init__(self, device: str = "auto"):
        """Initialize analyzer"""
        self.device = self._setup_device(device)
        self.models = {}
        logger.info(f"Initializing multimodal analysis system")
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Load models
        self._load_models()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device"""
        if device == "auto":
            if torch.cuda.is_available():
                # Optimization for RTX 3080+
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                return torch.device('cuda')
            return torch.device('cpu')
        return torch.device(device)
    
    def _load_models(self):
        """Load all models"""
        try:
            # 1. Load CLIP model
            logger.info("Loading CLIP model...")
            import clip
            self.models['clip'], self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.models['clip'].eval()
            
            # 2. Load YOLOv8 model
            logger.info("Loading YOLOv8 model...")
            from ultralytics import YOLO
            self.models['yolo'] = YOLO('yolov8m.pt')  # Medium size model for balance
            
            # 3. Load Whisper model
            logger.info("Loading Whisper Large V3 model...")
            import whisper
            self.models['whisper'] = whisper.load_model("large-v3", device=self.device)
            
            logger.info("All models loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.info("Please run: pip install openai-clip ultralytics openai-whisper")
            raise
    
    def analyze_video(self, video_path: str, sample_rate: int = 30) -> MediaAnalysisResult:
        """Analyze video file"""
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting analysis: {video_path.name}")
        
        # 1. Scene understanding and segmentation
        scenes = self._analyze_scenes(video_path, sample_rate)
        
        # 2. Object detection and tracking
        objects = self._detect_objects(video_path, sample_rate)
        
        # 3. Speech recognition
        transcripts = self._transcribe_audio(video_path)
        
        # 4. Generate query embeddings
        embeddings = self._generate_embeddings(scenes, objects, transcripts)
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return MediaAnalysisResult(
            video_path=str(video_path),
            scenes=scenes,
            objects=objects,
            transcripts=transcripts,
            embeddings=embeddings,
            processing_time=processing_time
        )
    
    def _analyze_scenes(self, video_path: Path, sample_rate: int) -> List[Dict]:
        """Scene understanding using CLIP"""
        logger.info("Performing scene analysis...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        frame_features = []
        frame_indices = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Preprocess frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self.clip_preprocess(self._frame_to_pil(frame_rgb)).unsqueeze(0).to(self.device)
                
                # Extract CLIP features
                with torch.no_grad():
                    features = self.models['clip'].encode_image(image)
                    frame_features.append(features.cpu().numpy())
                    frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        # Detect scene boundaries
        scenes = self._detect_scene_boundaries(frame_features, frame_indices, fps)
        
        logger.info(f"  ✓ Detected {len(scenes)} scenes")
        return scenes
    
    def _detect_objects(self, video_path: Path, sample_rate: int) -> List[Dict]:
        """Object detection using YOLOv8"""
        logger.info("Performing object detection...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_detections = []
        object_tracks = {}  # Simple object tracking
        track_id_counter = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # YOLOv8 detection
                results = self.models['yolo'](frame, verbose=False)
                
                timestamp = frame_count / fps
                frame_detections = []
                
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            # Simple tracking logic
                            track_id = self._assign_track_id(
                                cls, xyxy, object_tracks, track_id_counter
                            )
                            if track_id == track_id_counter:
                                track_id_counter += 1
                            
                            detection = {
                                'frame': frame_count,
                                'timestamp': timestamp,
                                'track_id': track_id,
                                'class': self.models['yolo'].names[cls],
                                'confidence': conf,
                                'bbox': xyxy.tolist()
                            }
                            frame_detections.append(detection)
                
                if frame_detections:
                    all_detections.extend(frame_detections)
            
            frame_count += 1
        
        cap.release()
        
        # Statistics
        unique_objects = set(d['class'] for d in all_detections)
        logger.info(f"  ✓ Detected {len(unique_objects)} object types")
        
        return all_detections
    
    def _transcribe_audio(self, video_path: Path) -> Dict:
        """Speech recognition using Whisper"""
        logger.info("Performing speech recognition...")
        
        # Use Whisper to directly process video file
        result = self.models['whisper'].transcribe(
            str(video_path),
            language=None,  # None表示自动检测语言
            task='transcribe',
            verbose=False
        )
        
        # Process results
        transcripts = {
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': []
        }
        
        # Process segments
        for segment in result.get('segments', []):
            transcripts['segments'].append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('avg_logprob', 0) + 1  # Convert to 0-1 range
            })
        
        logger.info(f"  ✓ Detected language: {transcripts['language']}")
        logger.info(f"  ✓ Transcribed segments: {len(transcripts['segments'])}")
        
        return transcripts
    
    def _generate_embeddings(self, scenes: List[Dict], objects: List[Dict], 
                           transcripts: Dict) -> np.ndarray:
        """Generate embeddings for queries"""
        logger.info("Generating query embeddings...")
        
        embeddings = []
        
        # Scene description embeddings
        for scene in scenes:
            if 'description' in scene:
                text_input = clip.tokenize(scene['description']).to(self.device)
                with torch.no_grad():
                    embedding = self.models['clip'].encode_text(text_input)
                    embeddings.append(embedding.cpu().numpy())
        
        # Object class embeddings
        unique_objects = list(set(obj['class'] for obj in objects))
        for obj_class in unique_objects[:10]:  # Limit quantity
            text = f"a photo of {obj_class}"
            text_input = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                embedding = self.models['clip'].encode_text(text_input)
                embeddings.append(embedding.cpu().numpy())
        
        # Transcript embeddings
        for segment in transcripts.get('segments', [])[:20]:  # Limit quantity
            text_input = clip.tokenize(segment['text'][:77]).to(self.device)
            with torch.no_grad():
                embedding = self.models['clip'].encode_text(text_input)
                embeddings.append(embedding.cpu().numpy())
        
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.array([])
        
        logger.info(f"  ✓ Generated {len(embeddings)} query embeddings")
        return embeddings
    
    def search(self, query: str, results: MediaAnalysisResult, top_k: int = 5) -> List[Dict]:
        """
        Natural language query
        """
        # Encode query
        text_input = clip.tokenize(query).to(self.device)
        with torch.no_grad():
            query_embedding = self.models['clip'].encode_text(text_input).cpu().numpy()
        
        # Calculate similarity
        if len(results.embeddings) > 0:
            similarities = np.dot(results.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            matches = []
            for idx in top_indices:
                score = float(similarities[idx])
                
                # Determine match content
                if idx < len(results.scenes):
                    content = results.scenes[idx]
                    match_type = 'scene'
                elif idx < len(results.scenes) + len(set(obj['class'] for obj in results.objects)):
                    content = {'class': list(set(obj['class'] for obj in results.objects))[idx - len(results.scenes)]}
                    match_type = 'object'
                else:
                    segment_idx = idx - len(results.scenes) - len(set(obj['class'] for obj in results.objects))
                    if segment_idx < len(results.transcripts.get('segments', [])):
                        content = results.transcripts['segments'][segment_idx]
                        match_type = 'transcript'
                    else:
                        continue
                
                matches.append({
                    'type': match_type,
                    'content': content,
                    'score': score
                })
            
            return matches
        
        return []
    
    # Helper methods
    def _frame_to_pil(self, frame: np.ndarray):
        """Convert OpenCV frame to PIL image"""
        from PIL import Image
        return Image.fromarray(frame)
    
    def _detect_scene_boundaries(self, features: List[np.ndarray], 
                               indices: List[int], fps: float) -> List[Dict]:
        """Detect scene boundaries"""
        if len(features) < 2:
            return []
        
        scenes = []
        features_array = np.vstack(features)
        
        # Calculate similarity between adjacent frames
        similarities = []
        for i in range(len(features_array) - 1):
            sim = np.dot(features_array[i], features_array[i+1]) / (
                np.linalg.norm(features_array[i]) * np.linalg.norm(features_array[i+1])
            )
            similarities.append(sim)
        
        # Detect scene changes (sudden drop in similarity)
        threshold = np.mean(similarities) - np.std(similarities)
        scene_boundaries = [0]
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                scene_boundaries.append(i + 1)
        
        scene_boundaries.append(len(features))
        
        # Create scene list
        for i in range(len(scene_boundaries) - 1):
            start_idx = scene_boundaries[i]
            end_idx = scene_boundaries[i + 1]
            
            if end_idx > start_idx:
                scene = {
                    'scene_id': i + 1,
                    'start_frame': indices[start_idx],
                    'end_frame': indices[end_idx - 1] if end_idx < len(indices) else indices[-1],
                    'start_time': indices[start_idx] / fps,
                    'end_time': (indices[end_idx - 1] if end_idx < len(indices) else indices[-1]) / fps,
                    'duration': ((indices[end_idx - 1] if end_idx < len(indices) else indices[-1]) - indices[start_idx]) / fps,
                    'description': f"Scene {i + 1}"  # Can generate description using CLIP
                }
                scenes.append(scene)
        
        return scenes
    
    def _assign_track_id(self, cls: int, bbox: np.ndarray, 
                        tracks: Dict, counter: int) -> int:
        """Simple object tracking ID assignment"""
        # This is a very simplified tracking logic
        # Real applications should use more complex tracking algorithms
        
        min_distance = float('inf')
        best_track_id = counter
        
        for track_id, track_info in tracks.items():
            if track_info['class'] == cls:
                # Calculate bounding box center distance
                prev_center = track_info['last_center']
                curr_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                
                distance = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                                 (prev_center[1] - curr_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Distance threshold
                    min_distance = distance
                    best_track_id = track_id
        
        # Update tracking information
        tracks[best_track_id] = {
            'class': cls,
            'last_center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        }
        
        return best_track_id
