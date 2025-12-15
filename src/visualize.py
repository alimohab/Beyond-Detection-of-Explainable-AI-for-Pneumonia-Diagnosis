"""
Visualization utilities for key frame detection results
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA


def plot_keyframes_comparison(frames, keyframes_model1, keyframes_model2, 
                              model1_name="Model 1 (CNN)", 
                              model2_name="Model 2 (LSTM)",
                              save_path=None):
    """
    Create comparison visualization of key frames from both models
    
    Args:
        frames: List of all frames
        keyframes_model1: Key frame indices from model 1
        keyframes_model2: Key frame indices from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        save_path: Path to save the figure
    """
    n_keyframes = max(len(keyframes_model1), len(keyframes_model2))
    fig, axes = plt.subplots(2, n_keyframes, figsize=(4*n_keyframes, 8))
    
    if n_keyframes == 1:
        axes = axes.reshape(2, 1)
    
    # Plot model 1 keyframes
    for i, idx in enumerate(keyframes_model1):
        if i < n_keyframes:
            axes[0, i].imshow(frames[idx])
            axes[0, i].set_title(f'{model1_name}\nFrame {idx}')
            axes[0, i].axis('off')
    
    # Fill empty slots
    for i in range(len(keyframes_model1), n_keyframes):
        axes[0, i].axis('off')
    
    # Plot model 2 keyframes
    for i, idx in enumerate(keyframes_model2):
        if i < n_keyframes:
            axes[1, i].imshow(frames[idx])
            axes[1, i].set_title(f'{model2_name}\nFrame {idx}')
            axes[1, i].axis('off')
    
    # Fill empty slots
    for i in range(len(keyframes_model2), n_keyframes):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def plot_keyframe_timeline(frames, keyframes_model1, keyframes_model2,
                          model1_name="Model 1 (CNN)",
                          model2_name="Model 2 (LSTM)",
                          save_path=None):
    """
    Plot timeline showing when key frames are detected
    
    Args:
        frames: List of all frames
        keyframes_model1: Key frame indices from model 1
        keyframes_model2: Key frame indices from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    n_frames = len(frames)
    
    # Plot timeline
    ax.plot([0, n_frames], [0, 0], 'k-', linewidth=2, label='Timeline')
    
    # Plot model 1 keyframes
    for idx in keyframes_model1:
        ax.plot([idx, idx], [-0.1, 0.1], 'b-', linewidth=3)
        ax.plot(idx, 0.15, 'bo', markersize=10)
    
    # Plot model 2 keyframes
    for idx in keyframes_model2:
        ax.plot([idx, idx], [-0.1, 0.1], 'r-', linewidth=3)
        ax.plot(idx, -0.15, 'ro', markersize=10)
    
    ax.set_xlim(-5, n_frames + 5)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_title('Key Frame Detection Timeline', fontsize=14, fontweight='bold')
    ax.legend([model1_name, model2_name], loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Timeline plot saved to {save_path}")
    
    plt.close()


def create_model_diagram(model_name, save_path=None):
    """
    Create a simple block diagram for the model architecture
    
    Args:
        model_name: Name of the model ('cnn' or 'lstm')
        save_path: Path to save the diagram
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    if model_name == 'cnn':
        # CNN model diagram
        blocks = [
            ('Input Video', 0.1, 0.5),
            ('Frame Extraction', 0.25, 0.5),
            ('ResNet50\nFeature Extraction', 0.45, 0.5),
            ('Feature Vectors', 0.65, 0.5),
            ('K-means\nClustering', 0.85, 0.5),
            ('Key Frames', 0.98, 0.5)
        ]
        
        title = 'Model 1: CNN-based Key Frame Detection'
        
    elif model_name == 'lstm':
        # LSTM model diagram
        blocks = [
            ('Input Video', 0.1, 0.5),
            ('Frame Extraction', 0.25, 0.5),
            ('ResNet50\nFeature Extraction', 0.45, 0.5),
            ('LSTM\nTemporal Modeling', 0.65, 0.5),
            ('Change Detection', 0.85, 0.5),
            ('Key Frames', 0.98, 0.5)
        ]
        
        title = 'Model 2: LSTM-based Temporal Key Frame Detection'
    
    # Draw blocks
    for i, (label, x, y) in enumerate(blocks):
        # Draw rectangle
        rect = plt.Rectangle((x-0.08, y-0.15), 0.16, 0.3, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrow
        if i < len(blocks) - 1:
            ax.arrow(x+0.08, y, 0.09, 0, head_width=0.05, head_length=0.02, 
                    fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model diagram saved to {save_path}")
    
    plt.close()


def plot_feature_analysis(features, keyframes, save_path=None):
    """
    Visualize feature space and key frame selection
    
    Args:
        features: Feature matrix (n_frames, feature_dim)
        keyframes: Key frame indices
        save_path: Path to save the figure
    """
    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all frames
    ax.scatter(features_2d[:, 0], features_2d[:, 1], 
              c='lightgray', s=50, alpha=0.5, label='All Frames')
    
    # Plot keyframes
    keyframe_features = features_2d[keyframes]
    ax.scatter(keyframe_features[:, 0], keyframe_features[:, 1],
              c='red', s=200, marker='*', edgecolors='black', linewidths=2,
              label='Key Frames', zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('Feature Space Visualization (PCA)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature analysis plot saved to {save_path}")
    
    plt.close()

