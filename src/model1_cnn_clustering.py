"""
Model 1: CNN-based Key Frame Detection using Feature Extraction and Clustering

This model uses a pre-trained CNN (ResNet50) to extract features from each frame,
then applies K-means clustering to identify representative key frames.

Dataset: Pre-trained on ImageNet (1.2M images, 1000 classes)
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from utils import preprocess_frame, get_device


class CNNKeyFrameDetector:
    """
    CNN-based key frame detector using feature extraction and clustering
    """
    
    def __init__(self, n_clusters=5, model_name='resnet50'):
        """
        Initialize the detector
        
        Args:
            n_clusters: Number of key frames to detect
            model_name: Pre-trained model name ('resnet50' or 'vgg16')
        """
        self.n_clusters = n_clusters
        self.device = get_device()
        self.model = self._load_pretrained_model(model_name)
        self.feature_extractor = self._create_feature_extractor()
        self.kmeans = None
        
    def _load_pretrained_model(self, model_name):
        """Load pre-trained CNN model"""
        if model_name == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif model_name == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _create_feature_extractor(self):
        """Create feature extractor from pre-trained model"""
        # Remove the final classification layer
        if isinstance(self.model, models.ResNet):
            # For ResNet, extract features before the final FC layer
            return nn.Sequential(*list(self.model.children())[:-1])
        elif isinstance(self.model, models.VGG):
            # For VGG, extract features before the classifier
            return self.model.features
        
    def extract_features(self, frames):
        """
        Extract features from frames using pre-trained CNN
        
        Args:
            frames: List of frames (numpy arrays)
        
        Returns:
            Feature matrix (n_frames, feature_dim)
        """
        features = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Extracting features from frames...")
        with torch.no_grad():
            for frame in tqdm(frames):
                # Preprocess frame
                frame_tensor = transform(frame).unsqueeze(0).to(self.device)
                
                # Extract features
                feature = self.feature_extractor(frame_tensor)
                feature = feature.view(feature.size(0), -1)  # Flatten
                features.append(feature.cpu().numpy().flatten())
        
        return np.array(features)
    
    def detect_keyframes(self, frames):
        """
        Detect key frames using clustering
        
        Args:
            frames: List of frames
        
        Returns:
            List of key frame indices
        """
        # Extract features
        features = self.extract_features(frames)
        
        # Apply K-means clustering
        print(f"Applying K-means clustering with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Find frames closest to cluster centers
        keyframe_indices = []
        for i in range(self.n_clusters):
            cluster_center = self.kmeans.cluster_centers_[i]
            cluster_points = features[cluster_labels == i]
            
            if len(cluster_points) > 0:
                # Find the point closest to the cluster center
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                
                # Get the original frame index
                cluster_indices = np.where(cluster_labels == i)[0]
                keyframe_idx = cluster_indices[closest_idx]
                keyframe_indices.append(int(keyframe_idx))
        
        # Sort indices
        keyframe_indices.sort()
        
        return keyframe_indices
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'name': 'CNN-based Feature Extraction with Clustering',
            'backbone': 'ResNet50',
            'dataset': 'ImageNet (1.2M images, 1000 classes)',
            'method': 'K-means clustering on CNN features',
            'n_clusters': self.n_clusters
        }

