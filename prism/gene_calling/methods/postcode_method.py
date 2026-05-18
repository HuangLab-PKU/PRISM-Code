"""
PoSTcode-based gene calling method implementation.

This module implements a probabilistic decoding method using PoSTcode for PRISM gene calling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import sys
import itertools

# Add external PoSTcode to path
# Path from code/src/gene_calling/methods/postcode_method.py to PRISM root
_prism_root = Path(__file__).parent.parent.parent.parent.parent
_postcode_path = _prism_root / "external" / "SPRINTseq-code" / "local" / "test" / "postcode" / "source-code"
if str(_postcode_path) not in sys.path:
    sys.path.insert(0, str(_postcode_path))

from postcode.decoding_functions import (
    decoding_function, e_step, torch_format, chol_sigma_from_vec, 
    kronecker_product, mat_sqrt, barcodes_01_from_channels
)
import torch
import pyro
from pyro.distributions import MultivariateNormal, Categorical
from pyro.distributions import constraints
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta
from pyro import poutine
from pyro.infer import config_enumerate

from ..base import ClassificationResult

logger = logging.getLogger(__name__)


# Modified model with channel-specific scaling
@config_enumerate
def model_constrained_tensor_channel_specific(data, N, D, C, R, K, codes, batch_size=None):
    """
    Modified PoSTcode model with channel-specific scaling.
    
    Changes from original:
    - codes_tr_v: (1, D) -> (C,) for channel-specific scaling
    - theta calculation: uses element-wise multiplication with broadcasting
    """
    w = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)
    
    # using tensor sigma
    sigma_ch_v = pyro.param('sigma_ch_v', torch.eye(C)[np.tril_indices(C, 0)])
    sigma_ch = chol_sigma_from_vec(sigma_ch_v, C)
    sigma_ro_v = pyro.param('sigma_ro_v', torch.eye(D)[np.tril_indices(R, 0)])
    sigma_ro = chol_sigma_from_vec(sigma_ro_v, R)
    sigma = kronecker_product(sigma_ro, sigma_ch)
    
    # Channel-specific scaling: (C,) instead of (1, D)
    codes_tr_v = pyro.param(
        'codes_tr_v', 
        3 * torch.ones(C),  # C = number of channels (5 for PRISM)
        constraint=constraints.greater_than(1.)
    )
    codes_tr_consts_v = pyro.param('codes_tr_consts_v', -1 * torch.ones(1, D))
    
    # Calculate theta with channel-specific scaling
    # codes: (K, D) where D = C * R
    # codes_tr_v: (C,)
    # We need to reshape codes to (K, C, R) to apply channel-specific scaling
    codes_reshaped = codes.reshape(K, C, R)  # (K, C, R)
    
    # Apply channel-specific scaling: codes_tr_v is (C,), expand to (1, C, 1) for broadcasting
    codes_tr_v_expanded = codes_tr_v.unsqueeze(0).unsqueeze(2)  # (1, C, 1)
    codes_scaled = codes_reshaped * codes_tr_v_expanded  # (K, C, R) * (1, C, 1) -> (K, C, R)
    
    # Reshape back to (K, D) and add constants
    codes_scaled_flat = codes_scaled.reshape(K, D)  # (K, D)
    codes_with_consts = codes_scaled_flat + codes_tr_consts_v  # (K, D)
    
    theta = torch.matmul(codes_with_consts, mat_sqrt(sigma, D))
    
    with pyro.plate('data', N, batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample('obs', MultivariateNormal(theta[z], sigma), obs=data[batch])


auto_guide_constrained_tensor_channel_specific = AutoDelta(
    poutine.block(
        model_constrained_tensor_channel_specific, 
        expose=['weights', 'codes_tr_v', 'codes_tr_consts_v', 'sigma_ch_v', 'sigma_ro_v']
    )
)


def train_channel_specific(svi, num_iterations, data, N, D, C, R, K, codes, print_training_progress, batch_size):
    """Training function for channel-specific model."""
    pyro.clear_param_store()
    losses = []
    if print_training_progress:
        from tqdm import tqdm
        for j in tqdm(range(num_iterations)):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    else:
        for j in range(num_iterations):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    return losses


def decoding_function_channel_specific(
    spots, barcodes_01,
    num_iter=60, batch_size=15000, up_prc_to_remove=99.95,
    modify_bkg_prior=True,
    estimate_bkg=True, estimate_additional_barcodes=None,
    add_remaining_barcodes_prior=0.05,
    print_training_progress=True, set_seed=1
):
    """
    Modified decoding function with channel-specific scaling.
    
    This is a wrapper around PoSTcode's decoding_function that uses
    channel-specific codes_tr_v instead of global scaling.
    """
    # Set device
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    
    N = spots.shape[0]
    if N == 0:
        logger.warning('There are no spots to decode.')
        return None
    
    C = spots.shape[1]
    R = spots.shape[2]
    K = barcodes_01.shape[0]
    D = C * R
    data = torch_format(spots)
    codes = torch_format(barcodes_01)
    
    # Include background
    if estimate_bkg:
        bkg_ind = codes.shape[0]
        codes = torch.cat((codes, torch.zeros(1, D)))
    else:
        bkg_ind = np.empty((0,), dtype=np.int32)
    
    if np.any(estimate_additional_barcodes is not None):
        inf_ind = codes.shape[0] + np.arange(estimate_additional_barcodes.shape[0])
        codes = torch.cat((codes, torch_format(estimate_additional_barcodes)))
    else:
        inf_ind = np.empty((0,), dtype=np.int32)
    
    # Normalize spot values (same as original)
    ind_keep = np.where(
        np.sum(data.cpu().numpy() < np.percentile(data.cpu().numpy(), up_prc_to_remove, axis=0), axis=1) == D
    )[0] if up_prc_to_remove < 100 else np.arange(0, N)
    
    s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 60, axis=0))
    max_s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 99.9, axis=0))
    min_s = torch.min(data[ind_keep, :], dim=0).values
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data[ind_keep, :], dim=0).values + 1e-10, other=log_add.float())
    data_log = torch.log10(data + log_add)
    data_log_mean = data_log[ind_keep, :].mean(dim=0, keepdim=True)
    data_log_std = data_log[ind_keep, :].std(dim=0, keepdim=True)
    # Guard against zero/NaN std to avoid generating inf/NaN values that break PyTorch MVN validation.
    data_log_std = torch.where(data_log_std < 1e-6, torch.ones_like(data_log_std), data_log_std)
    data_norm = (data_log - data_log_mean) / data_log_std
    data_norm = torch.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Model training with channel-specific model
    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(
        model_constrained_tensor_channel_specific, 
        auto_guide_constrained_tensor_channel_specific, 
        optim, 
        loss=TraceEnum_ELBO(max_plate_nesting=1)
    )
    pyro.set_rng_seed(set_seed)
    losses = train_channel_specific(
        svi,
        num_iter,
        data_norm[ind_keep, :],
        len(ind_keep),
        D,
        C,
        R,
        codes.shape[0],
        codes,
        print_training_progress,
        min(len(ind_keep), batch_size),
    )
    
    # Collect estimated parameters
    w_star = pyro.param('weights').detach()
    sigma_ch_v_star = pyro.param('sigma_ch_v').detach()
    sigma_ro_v_star = pyro.param('sigma_ro_v').detach()
    sigma_ro_star = chol_sigma_from_vec(sigma_ro_v_star, R)
    sigma_ch_star = chol_sigma_from_vec(sigma_ch_v_star, C)
    sigma_star = kronecker_product(sigma_ro_star, sigma_ch_star)
    codes_tr_v_star = pyro.param('codes_tr_v').detach()  # Now (C,) instead of (1, D)
    codes_tr_consts_v_star = pyro.param('codes_tr_consts_v').detach()
    
    # Recalculate theta with channel-specific scaling
    codes_reshaped = codes.reshape(codes.shape[0], C, R)
    codes_tr_v_expanded = codes_tr_v_star.unsqueeze(0).unsqueeze(2)  # (1, C, 1)
    codes_scaled = codes_reshaped * codes_tr_v_expanded
    codes_scaled_flat = codes_scaled.reshape(codes.shape[0], D)
    codes_with_consts = codes_scaled_flat + codes_tr_consts_v_star
    theta_star = torch.matmul(codes_with_consts, mat_sqrt(sigma_star, D))
    
    # Computing class probabilities (same logic as original)
    if modify_bkg_prior and w_star.shape[0] > K:
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star
    
    if add_remaining_barcodes_prior > 0:
        barcodes_1234 = np.array([p for p in itertools.product(np.arange(1, C + 1), repeat=R)])
        codes_inf = np.array(torch_format(barcodes_01_from_channels(barcodes_1234, C, R)).cpu())
        codes_inf = np.concatenate((np.zeros((1, D)), codes_inf))
        codes_cpu = codes.cpu()
        for b in range(codes_cpu.shape[0]):
            r = np.array(codes_cpu[b, :], dtype=np.int32)
            if np.where(np.all(codes_inf == r, axis=1))[0].shape[0] != 0:
                i = np.reshape(np.where(np.all(codes_inf == r, axis=1)), (1,))[0]
                codes_inf = np.delete(codes_inf, i, axis=0)

        # If there are no remaining infeasible codes, fall back to using only the
        # originally provided codes to avoid division by zero in prior construction.
        if codes_inf.shape[0] == 0:
            class_probs_star = e_step(
                data_norm,
                w_star_mod,
                theta_star,
                sigma_star,
                N,
                codes.shape[0],
                print_training_progress,
            )
        else:
            if not estimate_bkg:
                bkg_ind = codes_cpu.shape[0]
                inf_ind = np.append(inf_ind, codes_cpu.shape[0] + 1 + np.arange(codes_inf.shape[0]))
            else:
                inf_ind = np.append(inf_ind, codes_cpu.shape[0] + np.arange(codes_inf.shape[0]))
            codes_inf = torch.tensor(codes_inf).float()
            alpha = (1 - add_remaining_barcodes_prior)
            w_star_all = torch.cat(
                (
                    alpha * w_star_mod,
                    torch.tensor((1 - alpha) / codes_inf.shape[0], device=w_star_mod.device).repeat(
                        codes_inf.shape[0]
                    ),
                )
            )

            # Recalculate theta for infeasible codes with channel-specific scaling
            codes_inf_reshaped = codes_inf.reshape(codes_inf.shape[0], C, R)
            codes_tr_v_expanded_inf = codes_tr_v_star.unsqueeze(0).unsqueeze(2)  # (1, C, 1)
            codes_inf_scaled = codes_inf_reshaped * codes_tr_v_expanded_inf
            codes_inf_scaled_flat = codes_inf_scaled.reshape(codes_inf.shape[0], D)
            codes_inf_with_consts = codes_inf_scaled_flat + codes_tr_consts_v_star
            theta_inf = torch.matmul(codes_inf_with_consts, mat_sqrt(sigma_star, D))

            class_probs_star = e_step(
                data_norm,
                w_star_all,
                torch.cat((theta_star, theta_inf)),
                sigma_star,
                N,
                w_star_all.shape[0],
                print_training_progress,
            )
    else:
        class_probs_star = e_step(
            data_norm, w_star_mod, theta_star, sigma_star, N, codes.shape[0], print_training_progress
        )
    
    # Collapsing added barcodes (same as original)
    class_probs_star_s = torch.cat((
        torch.cat((class_probs_star[:, 0:K], class_probs_star[:, bkg_ind].reshape((N, 1))), dim=1),
        torch.sum(class_probs_star[:, inf_ind], dim=1).reshape((N, 1))
    ), dim=1)
    inf_ind_s = inf_ind[0] if len(inf_ind) > 0 else np.empty((0,), dtype=np.int32)
    
    # Handle NaN
    nan_spot_ind = torch.unique((torch.isnan(class_probs_star_s)).nonzero(as_tuple=False)[:, 0])
    if nan_spot_ind.shape[0] > 0:
        nan_class_ind = class_probs_star_s.shape[1]
        class_probs_star_s = torch.cat((class_probs_star_s, torch.zeros((class_probs_star_s.shape[0], 1))), dim=1)
        class_probs_star_s[nan_spot_ind, :] = 0
        class_probs_star_s[nan_spot_ind, nan_class_ind] = 1
    else:
        nan_class_ind = np.empty((0,), dtype=np.int32)
    
    class_probs = class_probs_star_s.cpu().numpy()
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")
    
    class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s, 'nan': nan_class_ind}
    torch_params = {
        'w_star': w_star_mod.cpu(), 
        'sigma_star': sigma_star.cpu(),
        'sigma_ro_star': sigma_ro_star.cpu(), 
        'sigma_ch_star': sigma_ch_star.cpu(),
        'theta_star': theta_star.cpu(), 
        'codes_tr_consts_v_star': codes_tr_consts_v_star.cpu(),
        'codes_tr_v_star': codes_tr_v_star.cpu(),  # Now (C,) shape
        'losses': losses
    }
    norm_const = {'log_add': log_add, 'data_log_mean': data_log_mean, 'data_log_std': data_log_std}
    
    return {'class_probs': class_probs, 'class_ind': class_ind, 'params': torch_params, 'norm_const': norm_const}


class PostcodeMethod:
    """
    PoSTcode-based gene calling method.
    
    This method uses probabilistic decoding with structured codebook constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PoSTcode method with configuration.

        Args:
            config: Configuration dictionary containing PoSTcode-specific parameters
        """
        self.config = config
        self.is_fitted = False
        self.postcode_result = None
        self.gene_names = None
        self.scale_factors = None
        
        # PoSTcode-specific parameters
        postcode_config = config.get("postcode", {})
        # Default codebook path: from code/src/gene_calling/methods/ to code/src/configs/codebooks/
        default_codebook = Path(__file__).parent.parent.parent / "configs" / "codebooks" / "prism30_codebook.csv"
        self.codebook_path = Path(postcode_config.get("codebook_path", str(default_codebook)))
        self.num_iter = postcode_config.get("num_iter", 60)
        self.batch_size = postcode_config.get("batch_size", 15000)
        self.estimate_bkg = postcode_config.get("estimate_bkg", True)
        self.add_remaining_barcodes_prior = postcode_config.get("add_remaining_barcodes_prior", 0.05)
        self.print_training_progress = postcode_config.get("print_training_progress", True)
        self.set_seed = postcode_config.get("set_seed", 1)
        
        # Preprocessing parameters
        self.p99_percentile = postcode_config.get("p99_percentile", 99)
        self.baseline_channel = postcode_config.get("baseline_channel", "ch1")
        
        # PRISM panel parameters
        self.prism_panel_type = config.get("prism_panel_type", "PRISM30")
        
        # Channel-specific scaling option
        self.use_channel_specific_scaling = postcode_config.get("use_channel_specific_scaling", False)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PoSTcode-specific preprocessing to signal data.
        
        This includes:
        1. P99 channel scaling (coarse correction)
        2. Construction of observation vector (ratios + intensity)
        
        Args:
            data: Intensity data with columns ['ch1', 'ch2', 'ch3', 'ch4']
        
        Returns:
            Preprocessed data with additional computed columns
        """
        if not self._validate_data(data):
            raise ValueError("Input data missing required channels")
        
        processed = data.copy()
        
        # Ensure standard ch1-ch4 channel names
        required_channels = ["ch1", "ch2", "ch3", "ch4"]
        missing_channels = [col for col in required_channels if col not in processed.columns]
        if missing_channels:
            raise ValueError(f"Missing required channels: {missing_channels}")
        
        # Step 1: P99 channel scaling
        scaled_data, scale_factors = self._p99_channel_scaling(processed)
        self.scale_factors = scale_factors
        
        # Step 2: Calculate ratios (for compatibility, stored in processed)
        A = scaled_data['ch1'] + scaled_data['ch2'] + scaled_data['ch4']
        A = A.replace(0, 1e-10)
        processed['ch1/A'] = scaled_data['ch1'] / A
        processed['ch2/A'] = scaled_data['ch2'] / A
        processed['ch3/A'] = scaled_data['ch3'] / A
        processed['ch4/A'] = scaled_data['ch4'] / A
        
        # Store scaled intensities for later use
        processed['scaled_ch1'] = scaled_data['ch1']
        processed['scaled_ch2'] = scaled_data['ch2']
        processed['scaled_ch3'] = scaled_data['ch3']
        processed['scaled_ch4'] = scaled_data['ch4']
        
        return processed

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from preprocessed signal data.
        
        Note: For PoSTcode, we don't use traditional features.
        This method is kept for interface compatibility.
        
        Args:
            data: Preprocessed data from PoSTcode preprocessing
        
        Returns:
            Feature matrix (n_samples, n_features) - empty for PoSTcode
        """
        # PoSTcode uses observation vectors directly, not traditional features
        return np.array([]).reshape(len(data), 0)

    def fit(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> "PostcodeMethod":
        """
        Fit the PoSTcode model to training data.

        Args:
            data: Training data with intensity values
            labels: Optional ground truth labels (not used in PoSTcode)
            metadata: Optional metadata (not used)

        Returns:
            Self for method chaining
        """
        logger.info("Fitting PoSTcode method...")
        
        # Preprocess data
        processed_data = self.preprocess(data)
        
        # Construct observation vector
        obs_vector = self._construct_observation_vector(processed_data)
        logger.info(f"Observation vector shape: {obs_vector.shape}")
        
        # Load codebook
        codebook, gene_names = self._load_codebook(self.codebook_path)
        logger.info(f"Loaded codebook with {len(gene_names)} genes")
        
        # Construct background code (if needed, PoSTcode will add it automatically)
        # We don't manually add it here since estimate_bkg=True will handle it
        
        # Call PoSTcode training
        logger.info("Starting PoSTcode training...")
        if self.use_channel_specific_scaling:
            logger.info("Using channel-specific scaling (codes_tr_v shape: (C,))")
            result = decoding_function_channel_specific(
                spots=obs_vector,  # (N, 5, 1)
                barcodes_01=codebook,  # (K, 5, 1) where K=31
                num_iter=self.num_iter,
                batch_size=min(self.batch_size, len(obs_vector)),
                up_prc_to_remove=99.95,
                modify_bkg_prior=True,
                estimate_bkg=self.estimate_bkg,
                estimate_additional_barcodes=None,
                add_remaining_barcodes_prior=self.add_remaining_barcodes_prior,
                print_training_progress=self.print_training_progress,
                set_seed=self.set_seed,
            )
        else:
            logger.info("Using global scaling (codes_tr_v shape: (1, D))")
            result = decoding_function(
                spots=obs_vector,  # (N, 5, 1)
                barcodes_01=codebook,  # (K, 5, 1) where K=31
                num_iter=self.num_iter,
                batch_size=min(self.batch_size, len(obs_vector)),
                up_prc_to_remove=99.95,
                modify_bkg_prior=True,
                estimate_bkg=self.estimate_bkg,
                estimate_additional_barcodes=None,
                add_remaining_barcodes_prior=self.add_remaining_barcodes_prior,
                print_training_progress=self.print_training_progress,
                set_seed=self.set_seed,
            )
        
        if result is None:
            raise ValueError("PoSTcode decoding returned None")
        
        # Save training results
        self.postcode_result = result
        self.gene_names = gene_names.copy()
        if self.estimate_bkg:
            self.gene_names.append('Background')
        # Infeasible will be handled in predict
        self.is_fitted = True
        
        logger.info("PoSTcode training completed successfully")
        
        return self

    def predict(self, data: pd.DataFrame) -> ClassificationResult:
        """
        Predict labels for new data.

        Args:
            data: Data to predict (can be same as training data)

        Returns:
            ClassificationResult containing predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before making predictions")
        
        logger.info("Making predictions with PoSTcode method...")
        
        # Preprocess data
        processed_data = self.preprocess(data)
        
        # Construct observation vector
        obs_vector = self._construct_observation_vector(processed_data)
        
        # Extract trained parameters and normalization constants
        class_ind = self.postcode_result['class_ind']
        params = self.postcode_result['params']
        norm_const = self.postcode_result['norm_const']
        
        # Re-run E-step for new data using trained parameters
        class_probs = self._run_e_step_for_new_data(
            obs_vector, params, norm_const, class_ind
        )
        
        # Calculate entropy
        entropy = self._calculate_entropy(class_probs)
        
        # Get Top1 gene
        top1_indices = np.argmax(class_probs, axis=1)
        top1_probs = class_probs[np.arange(len(class_probs)), top1_indices]
        
        # Map indices to gene names
        top1_genes = []
        for idx in top1_indices:
            if idx < len(self.gene_names):
                top1_genes.append(self.gene_names[idx])
            elif 'inf' in class_ind and idx == class_ind['inf']:
                top1_genes.append('Infeasible')
            elif 'nan' in class_ind and idx == class_ind['nan']:
                top1_genes.append('NaN')
            else:
                top1_genes.append('Unknown')
        
        # Create labels (1-based)
        labels = top1_indices + 1
        
        # Create result
        result = ClassificationResult(
            labels=labels,
            probabilities=class_probs,
            model_params=self.get_model_info(),
            metadata={
                'method': 'Postcode',
                'gene_names': self.gene_names,
                'entropy': entropy,
                'top1_genes': top1_genes,
                'top1_probs': top1_probs,
                'class_ind': class_ind,
            }
        )
        
        logger.info(f"Predicted {len(np.unique(labels))} unique labels")
        
        return result

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required channels."""
        standard_channels = ["ch1", "ch2", "ch3", "ch4"]
        has_standard_channels = all(col in data.columns for col in standard_channels)
        
        # Also support legacy PRISM channel names
        prism_channels = ["R", "Ye", "B", "G"]
        has_prism_channels = all(col in data.columns for col in prism_channels)
        
        return has_standard_channels or has_prism_channels

    def _p99_channel_scaling(self, intensity_df: pd.DataFrame) -> tuple:
        """
        Calculate P99 percentile for each channel and scale relative to baseline channel.
        
        Returns:
            - scaled_df: Scaled intensity DataFrame
            - scale_factors: Dictionary of scale factors
        """
        # Calculate P99 percentiles
        p99_values = {}
        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            if ch in intensity_df.columns:
                p99_values[ch] = np.percentile(intensity_df[ch], self.p99_percentile)
            else:
                p99_values[ch] = 1.0
        
        # Use baseline channel as reference
        baseline = p99_values.get(self.baseline_channel, 1.0)
        if baseline == 0:
            baseline = 1.0
        
        # Calculate scale factors
        scale_factors = {}
        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            if ch == self.baseline_channel:
                scale_factors[ch] = 1.0
            else:
                p99_val = p99_values.get(ch, 1.0)
                if p99_val > 0:
                    scale_factors[ch] = baseline / p99_val
                else:
                    scale_factors[ch] = 1.0
        
        # Apply scaling
        scaled_df = intensity_df.copy()
        for ch, factor in scale_factors.items():
            if ch in scaled_df.columns:
                scaled_df[ch] = scaled_df[ch] * factor
        
        logger.info(f"P99 scale factors: {scale_factors}")
        
        return scaled_df, scale_factors

    def _construct_observation_vector(self, processed_data: pd.DataFrame) -> np.ndarray:
        """
        Construct 5D observation vector: [ch1/A, ch2/A, ch3/A, ch4/A, sum_graded_channels].
        
        Note: sum uses only graded channels (ch1, ch2, ch4), not ch3.

        Important:
        - The original PoSTcode `decoding_function` applies its own log10 transform internally:
          `data_log = log10(data + log_add)`.
          Therefore, we must NOT pre-log the magnitude dimension here.
          If we pass log10(sum) here, PoSTcode would compute log10(log10(sum)+log_add),
          which can be invalid and produce NaNs.
        
        Returns:
            numpy array (N, 5, 1) for PoSTcode
        """
        # Calculate ratios (A = ch1 + ch2 + ch4, excluding ch3)
        A = processed_data['scaled_ch1'] + processed_data['scaled_ch2'] + processed_data['scaled_ch4']
        A = A.replace(0, 1e-10)  # Avoid division by zero
        
        # Calculate ratios
        ratios = pd.DataFrame({
            'ch1/A': processed_data['scaled_ch1'] / A,
            'ch2/A': processed_data['scaled_ch2'] / A,
            'ch3/A': processed_data['scaled_ch3'] / A,
            'ch4/A': processed_data['scaled_ch4'] / A,
        })
        
        # Magnitude (only graded channels: ch1, ch2, ch4). Keep it in linear scale.
        sum_graded = processed_data["scaled_ch1"] + processed_data["scaled_ch2"] + processed_data["scaled_ch4"]
        sum_graded = np.maximum(sum_graded.values.astype(float), 0.0)
        
        # Construct observation vector (N, 5, 1)
        N = len(processed_data)
        obs_vector = np.zeros((N, 5, 1))
        obs_vector[:, 0, 0] = ratios['ch1/A'].values
        obs_vector[:, 1, 0] = ratios['ch2/A'].values
        obs_vector[:, 2, 0] = ratios['ch3/A'].values
        obs_vector[:, 3, 0] = ratios['ch4/A'].values
        obs_vector[:, 4, 0] = sum_graded
        
        return obs_vector

    def _load_codebook(self, codebook_path: Path) -> tuple:
        """
        Load codebook and convert to PoSTcode format.
        
        Codebook format: CSV with columns ['Gene', 'ch1/A', 'ch2/A', 'ch3/A', 'ch4/A', 'log10_sum']
        Converts to PoSTcode format: (K, 5, 1) where K=31
        
        Returns:
            - codebook_array: numpy array (K, 5, 1)
            - gene_names: list of gene names
        """
        if not codebook_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {codebook_path}")
        
        codebook_df = pd.read_csv(codebook_path)
        
        # Extract ratios.
        #
        # For the 5th (magnitude) dimension, we use a constant positive value (1.0) for all genes.
        # This dimension is meant to represent a positive linear-scale magnitude, and PoSTcode will
        # apply log10 internally + learn transformation parameters.
        codebook_array = np.zeros((len(codebook_df), 5, 1), dtype=float)
        codebook_array[:, 0, 0] = codebook_df["ch1/A"].values.astype(float)
        codebook_array[:, 1, 0] = codebook_df["ch2/A"].values.astype(float)
        codebook_array[:, 2, 0] = codebook_df["ch3/A"].values.astype(float)
        codebook_array[:, 3, 0] = codebook_df["ch4/A"].values.astype(float)
        codebook_array[:, 4, 0] = 1.0
        
        gene_names = codebook_df['Gene'].tolist()
        
        logger.info(f"Loaded codebook: {len(gene_names)} genes")
        
        return codebook_array, gene_names

    def _run_e_step_for_new_data(
        self,
        obs_vector: np.ndarray,
        params: Dict[str, Any],
        norm_const: Dict[str, Any],
        class_ind: Dict[str, Any],
    ) -> np.ndarray:
        """
        Run E-step for new data using trained parameters.
        
        Args:
            obs_vector: Observation vector (N, 5, 1)
            params: Trained parameters from PoSTcode
            norm_const: Normalization constants from training
            class_ind: Class indices dictionary
        
        Returns:
            Class probabilities (N, K) where K includes genes + background + infeasible
        """
        # Convert to torch format (keep everything on a single device)
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        # Format data: (N, D, 1) -> (N, D) where D can be 4 or 5
        data = torch_format(obs_vector)  # (N, D) on default device
        device = data.device
        N, D = data.shape
        C = D  # channels (4 for 4D mode, 5 for 5D mode)
        R = 1  # rounds
        
        # Apply normalization using training constants
        log_add = norm_const["log_add"].to(device)
        data_log_mean = norm_const["data_log_mean"].to(device)
        data_log_std = norm_const["data_log_std"].to(device)
        
        data_log = torch.log10(data + log_add)
        data_log_std = torch.where(data_log_std < 1e-6, torch.ones_like(data_log_std), data_log_std)
        data_norm = (data_log - data_log_mean) / data_log_std
        data_norm = torch.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get trained parameters
        w_star = params["w_star"].to(device)
        sigma_star = params["sigma_star"].to(device)
        theta_star = params["theta_star"].to(device)
        codes_tr_v_star = params["codes_tr_v_star"].to(device)
        codes_tr_consts_v_star = params["codes_tr_consts_v_star"].to(device)
        
        # Reconstruct codes (we need to reload codebook to get codes)
        codebook, _ = self._load_codebook(self.codebook_path)
        codes = torch_format(codebook).to(device)  # (K, D) where D matches obs_vector dimension
        K = codes.shape[0]
        
        # Verify codes dimension matches observation dimension
        if codes.shape[1] != D:
            raise ValueError(
                f"Codebook dimension ({codes.shape[1]}) does not match observation dimension ({D})"
            )
        
        # Add background code if estimate_bkg was True
        if self.estimate_bkg:
            # Background code in linear scale; PoSTcode will log10 internally.
            # For 5D mode: set intensity dimension (index 4) to 0
            # For 4D mode: all dimensions are 0 (no special intensity dimension)
            bg_code = torch.zeros(1, D, device=device)
            if D == 5:
                # 5D mode: intensity is at index 4
                bg_code[0, 4] = 0.0
            # For 4D mode, bg_code is already all zeros
            codes = torch.cat([codes, bg_code], dim=0)
            K_total = K + 1
        else:
            K_total = K
        
        # Reconstruct theta for all codes using trained parameters
        # Check if using channel-specific scaling (codes_tr_v shape: (C,)) or global scaling (shape: (1, D))
        if self.use_channel_specific_scaling and codes_tr_v_star.shape[0] == C:
            # Channel-specific scaling: codes_tr_v is (C,)
            codes_reshaped = codes.reshape(K_total, C, R)  # (K_total, C, R)
            codes_tr_v_expanded = codes_tr_v_star.unsqueeze(0).unsqueeze(2)  # (1, C, 1)
            codes_scaled = codes_reshaped * codes_tr_v_expanded  # (K_total, C, R)
            codes_scaled_flat = codes_scaled.reshape(K_total, D)  # (K_total, D)
            
            # codes_tr_consts_v_star shape: (1, D)
            if codes_tr_consts_v_star.dim() == 2:
                codes_tr_consts_expanded = codes_tr_consts_v_star.expand(K_total, -1)
            elif codes_tr_consts_v_star.dim() == 1:
                codes_tr_consts_expanded = codes_tr_consts_v_star.unsqueeze(0).expand(K_total, -1)
            else:
                codes_tr_consts_expanded = codes_tr_consts_v_star
            
            codes_with_consts = codes_scaled_flat + codes_tr_consts_expanded
            theta_all = torch.matmul(codes_with_consts, mat_sqrt(sigma_star, D))
        else:
            # Global scaling: codes_tr_v is (1, D) or (D,)
            if codes_tr_v_star.dim() == 2:
                # Shape: (1, D)
                codes_tr_v_expanded = codes_tr_v_star.expand(K_total, -1)
            elif codes_tr_v_star.dim() == 1:
                # Shape: (D,) - expand to (K_total, D)
                codes_tr_v_expanded = codes_tr_v_star.unsqueeze(0).expand(K_total, -1)
            else:
                codes_tr_v_expanded = codes_tr_v_star
            
            # codes_tr_consts_v_star shape: (1, D)
            if codes_tr_consts_v_star.dim() == 2:
                codes_tr_consts_expanded = codes_tr_consts_v_star.expand(K_total, -1)
            elif codes_tr_consts_v_star.dim() == 1:
                codes_tr_consts_expanded = codes_tr_consts_v_star.unsqueeze(0).expand(K_total, -1)
            else:
                codes_tr_consts_expanded = codes_tr_consts_v_star
            
            # Calculate theta: theta = (codes * codes_tr_v + codes_tr_consts_v) @ mat_sqrt(sigma, D)
            theta_all = torch.matmul(
                codes * codes_tr_v_expanded + codes_tr_consts_expanded,
                mat_sqrt(sigma_star, D)
            )
        
        # Run E-step
        class_probs_torch = e_step(
            data_norm, w_star, theta_all, sigma_star, N, K_total,
            print_training_progress=False
        )
        
        # Convert to numpy
        class_probs = class_probs_torch.cpu().numpy()
        
        # Reset torch default type
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.FloatTensor")
        
        return class_probs

    def _calculate_entropy(self, class_probs: np.ndarray) -> np.ndarray:
        """
        Calculate Shannon entropy: H = -Σ p_i * log(p_i)
        
        Args:
            class_probs: Probability matrix (N, K)
        
        Returns:
            Entropy array (N,)
        """
        probs_safe = np.clip(class_probs, 1e-10, 1.0)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
        return entropy

    def to_dataframe(
        self,
        result: ClassificationResult,
        intensity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert ClassificationResult to DataFrame format with all class probabilities and entropy.
        
        Output columns:
        - index: spot index
        - Gene: Top1 gene name
        - Probability: Top1 probability
        - Entropy: Shannon entropy
        - Prob_Gene_1, Prob_Gene_2, ..., Prob_Background, Prob_Infeasible: All class probabilities
        
        Args:
            result: ClassificationResult from predict()
            intensity_df: Original intensity DataFrame
        
        Returns:
            DataFrame with mapping results
        """
        # Base columns
        mapping_df = pd.DataFrame({
            'index': intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values,
            'Gene': result.metadata['top1_genes'],
            'Probability': result.metadata['top1_probs'],
            'Entropy': result.metadata['entropy'],
        })
        
        # Add all class probabilities
        gene_names = result.metadata['gene_names']
        class_ind = result.metadata.get('class_ind', {})
        
        # Map probabilities to gene names
        n_classes = result.probabilities.shape[1]
        
        # Genes (0 to K-1)
        K = len([g for g in gene_names if g != 'Background'])
        for i in range(min(K, n_classes)):
            gene_name = gene_names[i] if i < len(gene_names) else f'Gene_{i+1}'
            mapping_df[f'Prob_{gene_name}'] = result.probabilities[:, i]
        
        # Background (if exists)
        if 'bkg' in class_ind and isinstance(class_ind['bkg'], (int, np.integer)):
            bkg_idx = class_ind['bkg']
            if bkg_idx < n_classes:
                mapping_df['Prob_Background'] = result.probabilities[:, bkg_idx]
        
        # Infeasible (if exists)
        if 'inf' in class_ind:
            inf_idx = class_ind['inf']
            if isinstance(inf_idx, (int, np.integer)) and inf_idx < n_classes:
                mapping_df['Prob_Infeasible'] = result.probabilities[:, inf_idx]
            elif isinstance(inf_idx, np.ndarray) and len(inf_idx) > 0:
                # Sum over all infeasible classes
                inf_mask = np.isin(np.arange(n_classes), inf_idx)
                if np.any(inf_mask):
                    mapping_df['Prob_Infeasible'] = result.probabilities[:, inf_mask].sum(axis=1)
        
        # NaN (if exists)
        if 'nan' in class_ind:
            nan_idx = class_ind['nan']
            if isinstance(nan_idx, (int, np.integer)) and nan_idx < n_classes:
                mapping_df['Prob_NaN'] = result.probabilities[:, nan_idx]
        
        return mapping_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            "method": "Postcode",
            "is_fitted": self.is_fitted,
            "prism_panel_type": self.prism_panel_type,
            "codebook_path": str(self.codebook_path),
            "num_iter": self.num_iter,
            "batch_size": self.batch_size,
            "estimate_bkg": self.estimate_bkg,
        }
        
        if self.is_fitted and self.postcode_result:
            info["gene_names"] = self.gene_names
            info["n_genes"] = len(self.gene_names) if self.gene_names else 0
        
        return info
