import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import joblib

from src.config import (
    NUM_CLASSES,
    ATTACK_CLASSES,
    SAVED_MODELS_DIR,
    DEFAULT_PENULTIMATE_THRESHOLD,
)

class PenultimateVerifier:
    """
    Implements Algorithm 2 (Local Penultimate Decision Quality & FP Suppression):
    Maintains centroids in the 64-dimensional penultimate latent space for Benign (TN)
    and each Attack category (TP).
    Calculates sample-to-centroid distances and radius bounds to verify detection authenticity.
    """
    def __init__(self, threshold_factor: float = DEFAULT_PENULTIMATE_THRESHOLD):
        self.threshold_factor = threshold_factor
        self.class_centroids: Dict[int, np.ndarray] = {}
        self.class_radii: Dict[int, float] = {}
        self.is_calibrated = False

    def calibrate(self, penultimate_embeds: np.ndarray, labels: np.ndarray):
        """
        Calculates empirical centroids and 95th-percentile cluster radii for each class
        using training / validation penultimate representations.
        """
        for cls_idx in range(NUM_CLASSES):
            mask = (labels == cls_idx)
            if np.sum(mask) == 0:
                continue
            cls_vectors = penultimate_embeds[mask]
            centroid = np.mean(cls_vectors, axis=0)
            self.class_centroids[cls_idx] = centroid

            # Distances to centroid
            dists = np.linalg.norm(cls_vectors - centroid, axis=1)
            radius = float(np.percentile(dists, 95))
            self.class_radii[cls_idx] = radius

        self.is_calibrated = True

    def verify_sample(
        self,
        penultimate_vec: np.ndarray,
        predicted_class: int,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Verifies whether a predicted attack sample is a genuine True Positive (TP)
        or a potential False Positive (FP) by checking its distance to cluster centroids.
        """
        if not self.is_calibrated:
            return {
                "verified_label": predicted_class,
                "is_confirmed_attack": (predicted_class != 0),
                "quality_status": "UNVERIFIED (Not Calibrated)",
                "distance_to_centroid": 0.0,
                "radius_threshold": 1.0,
                "confidence": confidence,
            }

        vec = penultimate_vec.reshape(-1)
        
        # Distance to predicted class centroid
        target_centroid = self.class_centroids.get(predicted_class)
        target_radius = self.class_radii.get(predicted_class, 1.0)
        
        if target_centroid is None:
            dist_to_target = 0.0
        else:
            dist_to_target = float(np.linalg.norm(vec - target_centroid))

        # Distance to Benign centroid (class 0)
        benign_centroid = self.class_centroids.get(0)
        benign_radius = self.class_radii.get(0, 1.0)
        dist_to_benign = float(np.linalg.norm(vec - benign_centroid)) if benign_centroid is not None else 1e9

        # Decision Logic:
        # If model predicted attack (predicted_class != 0):
        if predicted_class != 0:
            # Check if vector is actually closer to Benign baseline than the attack centroid
            if dist_to_benign < dist_to_target * 0.9 and dist_to_benign <= benign_radius:
                # Flagged as False Positive! Suppress alert.
                return {
                    "verified_label": 0,
                    "verified_class_name": "Benign",
                    "original_predicted_class": predicted_class,
                    "is_confirmed_attack": False,
                    "quality_status": "FALSE_POSITIVE_SUPPRESSED",
                    "distance_to_attack_centroid": dist_to_target,
                    "distance_to_benign_centroid": dist_to_benign,
                    "radius_threshold": target_radius,
                    "confidence": confidence,
                    "mitigation_action": "SUPPRESS_ALERT_AND_REFIT_BASELINE"
                }
            
            # If distance exceeds 1.5x cluster radius, treat as uncertain/novel outlier
            if dist_to_target > (target_radius * 1.5) and confidence < 0.75:
                return {
                    "verified_label": predicted_class,
                    "verified_class_name": ATTACK_CLASSES[predicted_class],
                    "original_predicted_class": predicted_class,
                    "is_confirmed_attack": False,
                    "quality_status": "SUSPICIOUS_LOW_CONFIDENCE",
                    "distance_to_attack_centroid": dist_to_target,
                    "distance_to_benign_centroid": dist_to_benign,
                    "radius_threshold": target_radius,
                    "confidence": confidence,
                    "mitigation_action": "ENHANCED_LOGGING_NO_BLOCK"
                }

            # Confirmed True Positive!
            return {
                "verified_label": predicted_class,
                "verified_class_name": ATTACK_CLASSES[predicted_class],
                "original_predicted_class": predicted_class,
                "is_confirmed_attack": True,
                "quality_status": "TRUE_POSITIVE_CONFIRMED",
                "distance_to_attack_centroid": dist_to_target,
                "distance_to_benign_centroid": dist_to_benign,
                "radius_threshold": target_radius,
                "confidence": confidence,
                "mitigation_action": "TRIGGER_SDN_ISOLATION"
            }
        else:
            # Predicted Benign
            return {
                "verified_label": 0,
                "verified_class_name": "Benign",
                "original_predicted_class": 0,
                "is_confirmed_attack": False,
                "quality_status": "TRUE_NEGATIVE_NORMAL",
                "distance_to_attack_centroid": 0.0,
                "distance_to_benign_centroid": dist_to_benign,
                "radius_threshold": benign_radius,
                "confidence": confidence,
                "mitigation_action": "ALLOW_TRAFFIC"
            }

    def save(self, filepath: Optional[str] = None):
        path = filepath or str(SAVED_MODELS_DIR / "penultimate_verifier.joblib")
        joblib.dump({
            "centroids": self.class_centroids,
            "radii": self.class_radii,
            "threshold_factor": self.threshold_factor,
            "is_calibrated": self.is_calibrated
        }, path)

    def load(self, filepath: Optional[str] = None):
        path = filepath or str(SAVED_MODELS_DIR / "penultimate_verifier.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            self.class_centroids = data["centroids"]
            self.class_radii = data["radii"]
            self.threshold_factor = data.get("threshold_factor", DEFAULT_PENULTIMATE_THRESHOLD)
            self.is_calibrated = data.get("is_calibrated", True)
