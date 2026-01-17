"""
Gaussian Mixture Model classifier for signal point classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .base import BaseClassifier, ClassificationResult


class GMMClassifier(BaseClassifier):
    """
    Gaussian Mixture Model classifier for signal points.
    
    Implements the GMM approach from the original gene_calling pipeline,
    with support for layer-based classification and initialization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # GMM parameters
        self.covariance_type = config.get('covariance_type', 'diag')
        self.max_iter = config.get('max_iter', 100)
        self.tol = config.get('tol', 1e-3)
        self.n_init = config.get('n_init', 1)
        
        # Layer-based classification
        self.use_layers = config.get('use_layers', True)
        self.g_layer_column = config.get('g_layer_column', 'G_layer')
        self.num_per_layer = config.get('num_per_layer', 15)
        
        # Initialization
        self.centroid_init_dict = config.get('centroid_init_dict', {})
        self.use_initial_centroids = len(self.centroid_init_dict) > 0
        
        # Feature scaling
        self.scale_features = config.get('scale_features', True)
        self.scaler = StandardScaler() if self.scale_features else None
        
        # Models for each layer
        self.layer_models = {}
        self.layer_scalers = {}
        
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> 'GMMClassifier':
        """
        Fit GMM classifier to training data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Optional ground truth labels (not used in GMM)
            metadata: Optional metadata containing layer information
            
        Returns:
            Self for method chaining
        """
        if metadata is None or self.g_layer_column not in metadata:
            # Single layer GMM
            self._fit_single_layer(features)
        else:
            # Multi-layer GMM
            self._fit_multi_layer(features, metadata)
            
        self.is_fitted = True
        return self
    
    def _fit_single_layer(self, features: np.ndarray):
        """Fit single GMM model."""
        n_components = self.config.get('n_components', 15)
        
        # Scale features if requested
        if self.scale_features:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features
            
        # Initialize means if provided
        means_init = None
        if self.use_initial_centroids and 0 in self.centroid_init_dict:
            means_init = self.centroid_init_dict[0]
            
        # Fit GMM
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            means_init=means_init,
            random_state=42
        )
        
        self.model.fit(features_scaled)
        
    def _fit_multi_layer(self, features: np.ndarray, metadata: Dict[str, Any]):
        """Fit separate GMM models for each layer."""
        g_layers = metadata[self.g_layer_column]
        unique_layers = np.unique(g_layers)
        
        for layer in unique_layers:
            layer_mask = g_layers == layer
            layer_features = features[layer_mask]
            
            if len(layer_features) == 0:
                continue
                
            # Scale features for this layer
            if self.scale_features:
                layer_scaler = StandardScaler()
                layer_features_scaled = layer_scaler.fit_transform(layer_features)
                self.layer_scalers[layer] = layer_scaler
            else:
                layer_features_scaled = layer_features
                
            # Initialize means if provided
            means_init = None
            if self.use_initial_centroids and layer in self.centroid_init_dict:
                means_init = self.centroid_init_dict[layer]
                
            # Fit GMM for this layer
            layer_model = GaussianMixture(
                n_components=self.num_per_layer,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=self.n_init,
                means_init=means_init,
                random_state=42
            )
            
            layer_model.fit(layer_features_scaled)
            self.layer_models[layer] = layer_model
            
    def predict(self, features: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Predict labels for new data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            metadata: Optional metadata containing layer information
            
        Returns:
            ClassificationResult containing predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
            
        if self.use_layers and metadata is not None and self.g_layer_column in metadata:
            return self._predict_multi_layer(features, metadata)
        else:
            return self._predict_single_layer(features)
    
    def _predict_single_layer(self, features: np.ndarray) -> ClassificationResult:
        """Predict using single layer model."""
        # Scale features
        if self.scale_features:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
            
        # Predict labels and probabilities
        labels = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        centroids = self.model.means_
        
        return ClassificationResult(
            labels=labels,
            probabilities=probabilities,
            centroids=centroids,
            model_params={
                'means': self.model.means_,
                'covariances': self.model.covariances_,
                'weights': self.model.weights_
            }
        )
    
    def _predict_multi_layer(self, features: np.ndarray, metadata: Dict[str, Any]) -> ClassificationResult:
        """Predict using multi-layer models."""
        g_layers = metadata[self.g_layer_column]
        unique_layers = np.unique(g_layers)
        
        all_labels = np.full(len(features), -1, dtype=int)
        all_probabilities = None
        all_centroids = {}
        
        for layer in unique_layers:
            if layer not in self.layer_models:
                continue
                
            layer_mask = g_layers == layer
            layer_features = features[layer_mask]
            
            if len(layer_features) == 0:
                continue
                
            # Scale features
            if self.scale_features and layer in self.layer_scalers:
                layer_features_scaled = self.layer_scalers[layer].transform(layer_features)
            else:
                layer_features_scaled = layer_features
                
            # Predict for this layer
            layer_model = self.layer_models[layer]
            layer_labels = layer_model.predict(layer_features_scaled)
            layer_probabilities = layer_model.predict_proba(layer_features_scaled)
            
            # Adjust labels to be globally unique
            global_labels = layer_labels + int(layer * self.num_per_layer + 1)
            all_labels[layer_mask] = global_labels
            
            # Store probabilities and centroids
            if all_probabilities is None:
                n_components_total = sum(len(self.layer_models[l]) for l in self.layer_models)
                all_probabilities = np.zeros((len(features), n_components_total))
                
            # Map layer probabilities to global probability matrix
            start_idx = int(layer * self.num_per_layer)
            end_idx = start_idx + self.num_per_layer
            all_probabilities[layer_mask, start_idx:end_idx] = layer_probabilities
            
            all_centroids[layer] = layer_model.means_
            
        return ClassificationResult(
            labels=all_labels,
            probabilities=all_probabilities,
            centroids=all_centroids,
            model_params={
                'layer_models': {k: {
                    'means': v.means_,
                    'covariances': v.covariances_,
                    'weights': v.weights_
                } for k, v in self.layer_models.items()}
            }
        )
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Note: GMM doesn't provide direct feature importance.
        Returns None to indicate this is not available.
        """
        return None
    
    def get_bic_score(self, features: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate Bayesian Information Criterion score.
        
        Args:
            features: Feature matrix
            metadata: Optional metadata
            
        Returns:
            BIC score
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before calculating BIC")
            
        if self.use_layers and metadata is not None and self.g_layer_column in metadata:
            return self._calculate_multi_layer_bic(features, metadata)
        else:
            return self._calculate_single_layer_bic(features)
    
    def _calculate_single_layer_bic(self, features: np.ndarray) -> float:
        """Calculate BIC for single layer model."""
        if self.scale_features:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        return self.model.bic(features_scaled)
    
    def _calculate_multi_layer_bic(self, features: np.ndarray, metadata: Dict[str, Any]) -> float:
        """Calculate BIC for multi-layer models."""
        g_layers = metadata[self.g_layer_column]
        unique_layers = np.unique(g_layers)
        
        total_bic = 0.0
        for layer in unique_layers:
            if layer not in self.layer_models:
                continue
                
            layer_mask = g_layers == layer
            layer_features = features[layer_mask]
            
            if len(layer_features) == 0:
                continue
                
            if self.scale_features and layer in self.layer_scalers:
                layer_features_scaled = self.layer_scalers[layer].transform(layer_features)
            else:
                layer_features_scaled = layer_features
                
            total_bic += self.layer_models[layer].bic(layer_features_scaled)
            
        return total_bic
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        
        if self.use_layers:
            info.update({
                'num_layers': len(self.layer_models),
                'layers': list(self.layer_models.keys()),
                'components_per_layer': self.num_per_layer
            })
        else:
            info.update({
                'n_components': self.model.n_components if self.model else None,
                'converged': self.model.converged_ if self.model else None
            })
            
        return info
