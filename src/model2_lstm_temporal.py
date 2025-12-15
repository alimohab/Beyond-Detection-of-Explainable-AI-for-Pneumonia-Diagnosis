"""
Model 2: LSTM-based Temporal Key Frame Detection

This model uses CNN features extracted from frames and LSTM networks
to model temporal dependencies and detect key frames based on
temporal changes in the video sequence.

Dataset: Can be trained on video datasets like UCF-101, Kinetics-400
For this implementation, we use pre-trained CNN features and train
a simple LSTM for temporal modeling.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import preprocess_frame, get_device, calculate_frame_differences


class LSTMKeyFrameDetector:
    """
    LSTM-based key frame detector using temporal modeling
    """
    
    def __init__(self, feature_dim=2048, hidden_dim=256, n_layers=2, threshold_percentile=85):
        """
        Initialize the detector
        
        Args:
            feature_dim: Dimension of CNN features
            hidden_dim: LSTM hidden dimension
            n_layers: Number of LSTM layers
            threshold_percentile: Percentile for key frame detection threshold
        """
        self.device = get_device()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.threshold_percentile = threshold_percentile
        
        # Load pre-trained CNN for feature extraction
        self.cnn_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        self.feature_extractor = nn.Sequential(*list(self.cnn_model.children())[:-1])
        
        # LSTM model for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        ).to(self.device)
        
        # Change detection head
        self.change_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.scaler = StandardScaler()
        
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
        
        print("Extracting CNN features from frames...")
        with torch.no_grad():
            for frame in tqdm(frames):
                frame_tensor = transform(frame).unsqueeze(0).to(self.device)
                feature = self.feature_extractor(frame_tensor)
                feature = feature.view(feature.size(0), -1)
                features.append(feature.cpu().numpy().flatten())
        
        return np.array(features)
    
    def compute_temporal_changes(self, features):
        """
        Compute temporal changes using LSTM
        
        Args:
            features: Feature matrix (n_frames, feature_dim)
        
        Returns:
            Change scores for each frame
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Prepare sequence data
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Forward through LSTM
        with torch.no_grad():
            lstm_out, _ = self.lstm(features_tensor)
            change_scores = self.change_detector(lstm_out.squeeze(0))
            change_scores = change_scores.cpu().numpy().flatten()
        
        return change_scores
    
    def detect_keyframes(self, frames):
        """
        Detect key frames using temporal modeling
        
        Args:
            frames: List of frames
        
        Returns:
            List of key frame indices
        """
        # Extract CNN features
        features = self.extract_features(frames)
        
        # Compute temporal changes
        print("Computing temporal changes using LSTM...")
        change_scores = self.compute_temporal_changes(features)
        
        # Also compute frame differences as additional signal
        frame_diffs = calculate_frame_differences(frames)
        # Normalize frame differences
        if len(frame_diffs) > 0:
            frame_diffs = np.concatenate([[0], frame_diffs])  # Add first frame
            frame_diffs = (frame_diffs - frame_diffs.min()) / (frame_diffs.max() - frame_diffs.min() + 1e-8)
        else:
            frame_diffs = np.zeros(len(frames))
        
        # Combine LSTM scores and frame differences
        combined_scores = 0.7 * change_scores + 0.3 * frame_diffs
        
        # Find threshold
        threshold = np.percentile(combined_scores, self.threshold_percentile)
        
        # Detect key frames (peaks in change scores)
        keyframe_indices = []
        
        # Find local maxima above threshold
        for i in range(1, len(combined_scores) - 1):
            if (combined_scores[i] > threshold and 
                combined_scores[i] > combined_scores[i-1] and 
                combined_scores[i] > combined_scores[i+1]):
                keyframe_indices.append(i)
        
        # If no keyframes found, use top scores
        if len(keyframe_indices) == 0:
            n_keyframes = min(5, len(frames))
            top_indices = np.argsort(combined_scores)[-n_keyframes:]
            keyframe_indices = sorted(top_indices.tolist())
        
        # Remove duplicates and sort
        keyframe_indices = sorted(list(set(keyframe_indices)))
        
        return keyframe_indices
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'name': 'LSTM-based Temporal Key Frame Detection',
            'backbone': 'ResNet50 + LSTM',
            'dataset': 'ImageNet (CNN) + Temporal modeling',
            'method': 'LSTM temporal modeling with change detection',
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers
        }

