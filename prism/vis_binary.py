"""Explicit5-channel,1-round binary PoSTcode; no SBS one-hot expansion or phasing.

The numerical decoder is promoted from the established VIS5c arbitrary-binary
implementation. R=1 eliminates cleavage/carryover correction; learned background
is retained rather than importing another experiment's fixed background fraction.
"""
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
logger=logging.getLogger(__name__)

def brightness_covariances(covariance, codes, log_std, weights, variance):
    """Distribute shared log10 brightness variation across the on bits of each code.

    Keep the prior-weighted covariance equal to the training covariance.
    The all-dark background has no common on-bit signal.
    """
    import torch

    if not np.isfinite(variance) or variance < 0:
        raise ValueError("log_brightness_variance must be finite and nonnegative")
    if variance == 0:
        return covariance
    directions = codes / log_std.reshape(1, -1)
    outer = directions.unsqueeze(-1) * directions.unsqueeze(-2)
    mean_outer = (outer * (weights / weights.sum())[:, None, None]).sum(dim=0)
    result = covariance + variance * (outer - mean_outer)
    if torch.any(torch.linalg.cholesky_ex(result).info != 0):
        raise ValueError("log_brightness_variance exceeds the fitted covariance budget")
    return result


def decode(spots,barcodes_01,gene_list,code_list,num_iter=100,batch_size=100000,inference_chunk_size=100000,verbose=True,background_prior=None,log_brightness_variance=0.):
    spots=np.asarray(spots,dtype=np.float32)
    barcodes_01=np.asarray(barcodes_01,dtype=np.float32)
    if spots.ndim!=3 or spots.shape[1:]!=(5,1) or barcodes_01.shape!=(31,5,1):
        raise ValueError('Expected N x5 x1 intensities and31 x5 x1 binary codes')
    if len(spots)<32 or not np.isfinite(spots).all() or (spots<0).any():
        raise ValueError('Insufficient or invalid spot intensities')
    if not np.isin(barcodes_01,[0,1]).all() or len({tuple(v.ravel()) for v in barcodes_01})!=31 or (barcodes_01.sum(axis=(1,2))==0).any():
        raise ValueError('Expected31 distinct nonzero binary codes; all-dark is background')
    if len(gene_list)!=31 or len(set(gene_list))!=31 or len(code_list)!=31:
        raise ValueError('Expected31 unique target assignments')
    if background_prior is not None and not 0<background_prior<1:
        raise ValueError('Invalid explicit background prior')
    if not np.isfinite(log_brightness_variance) or log_brightness_variance < 0:
        raise ValueError('log_brightness_variance must be finite and nonnegative')
    import torch
    import pyro
    from pyro.infer import SVI, TraceEnum_ELBO
    from pyro.optim import Adam
    from postcode.decoding_functions import (
        model_constrained_tensor, auto_guide_constrained_tensor, train, e_step,
        chol_sigma_from_vec, kronecker_product, mat_sqrt, torch_format,
    )

    n_spots = len(spots)
    # ----- begin region replicated from postcode_mapping (codebook already built) -----
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    N, C, R = spots.shape
    K = barcodes_01.shape[0]
    D = C * R

    pyro.clear_param_store()
    data = torch_format(spots)
    codes = torch_format(barcodes_01)

    up_prc_to_remove = 99.95
    # DIVERGENCE: modify_bkg_prior=False. The postcode default (True) pins the background-class
    # prior to the MIN gene weight, so Background never wins argmax and every spot is forced to a
    # gene — with this high-background data (480 autofluorescence, 3.28M over-detections) that
    # floods the top all-dark-ish code (Gfap 889k at Q20). postcode's own docs say to use False
    # "when there is a lot of background signal". False keeps the LEARNED (high) background weight
    # so background spots route to the Background class and drop out of the gene calls.
    modify_bkg_prior = False
    estimate_bkg = True
    add_remaining_barcodes_prior = 0.0  # <-- DIVERGENCE: skip SBS one-hot decoys
    set_seed = 1

    if estimate_bkg:
        bkg_ind = codes.shape[0]
        codes = torch.cat((codes, torch.zeros(1, D)))
    else:
        bkg_ind = np.empty((0,), dtype=np.int32)
    inf_ind = np.empty((0,), dtype=np.int32)  # no infeasible decoys

    if verbose:
        logger.info("Normalizing data...")
    data_cpu = data.cpu().numpy()
    if up_prc_to_remove < 100:
        sums = np.sum(data_cpu < np.percentile(data_cpu, up_prc_to_remove, axis=0), axis=1)
        ind_keep = np.where(sums == D)[0]
    else:
        ind_keep = np.arange(0, N)
    if len(ind_keep) == 0:
        logger.warning("No spots kept after percentile filter; using all spots.")
        ind_keep = np.arange(0, N)

    s = torch.tensor(np.percentile(data_cpu[ind_keep, :], 60, axis=0))
    max_s = torch.tensor(np.percentile(data_cpu[ind_keep, :], 99.9, axis=0))
    min_s = torch.min(data[ind_keep, :], dim=0).values
    if torch.cuda.is_available():
        s = s.cuda()
        max_s = max_s.cuda()
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    global_min = torch.min(data, dim=0).values
    #1e-10 vanishes at camera-intensity float32 scales; bound against every observed value.
    eps = torch.maximum(global_min.abs(), torch.ones_like(global_min)) * torch.finfo(data.dtype).eps * 8
    log_add = torch.max(-global_min + eps, other=log_add.float())

    data_log_subset = torch.log10(data[ind_keep, :] + log_add)
    data_log_mean = data_log_subset.mean(dim=0, keepdim=True)
    data_log_std = data_log_subset.std(dim=0, keepdim=True)
    if not torch.isfinite(data_log_mean).all() or not torch.isfinite(data_log_std).all() or (data_log_std <= 0).any():
        raise ValueError('Unobservable/degenerate intensity normalization')
    del data_log_subset

    if verbose:
        logger.info(f"Training (SVI) on {len(ind_keep)} spots, {num_iter} iters, batch {batch_size}...")
    data_norm_subset = (torch.log10(data[ind_keep, :] + log_add) - data_log_mean) / data_log_std
    optim = Adam({"lr": 0.085, "betas": [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor, optim,
              loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi, num_iter, data_norm_subset, len(ind_keep), D, C, R,
                   codes.shape[0], codes, verbose, min(len(ind_keep), batch_size))
    del data_norm_subset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w_star = pyro.param("weights").detach()
    sigma_ch_v_star = pyro.param("sigma_ch_v").detach()
    sigma_ro_v_star = pyro.param("sigma_ro_v").detach()
    sigma_ro_star = chol_sigma_from_vec(sigma_ro_v_star, R)
    sigma_ch_star = chol_sigma_from_vec(sigma_ch_v_star, C)
    sigma_star = kronecker_product(sigma_ro_star, sigma_ch_star)
    codes_tr_v_star = pyro.param("codes_tr_v").detach()
    codes_tr_consts_v_star = pyro.param("codes_tr_consts_v").detach()
    theta_star = torch.matmul(codes * codes_tr_v_star + codes_tr_consts_v_star, mat_sqrt(sigma_star, D))

    if background_prior is not None and estimate_bkg and w_star.shape[0] > K:
        # fixed high background prior: genes share (1 - background_prior), background gets background_prior
        wg = w_star[0:K] / w_star[0:K].sum() * (1.0 - background_prior)
        w_star_mod = torch.cat((wg, torch.tensor([background_prior], device=w_star.device, dtype=w_star.dtype)))
        if verbose:
            logger.info(f"Using FIXED background prior = {background_prior}")
    elif modify_bkg_prior and w_star.shape[0] > K:
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star

    # add_remaining_barcodes_prior == 0 -> no infeasible codes appended
    w_star_final = w_star_mod
    theta_final = theta_star
    K_inference = codes.shape[0]

    if verbose:
        logger.info(f"Inference (E-step) in chunks of {inference_chunk_size}...")
    res_indices = np.zeros(N, dtype=int)
    res_probs_1 = np.zeros(N, dtype=float)
    res_probs_2 = np.zeros(N, dtype=float)
    res_entropy = np.zeros(N, dtype=float)
    n_chunks = (N + inference_chunk_size - 1) // inference_chunk_size

    inference_cov = brightness_covariances(
        sigma_star, codes, data_log_std, w_star_final, log_brightness_variance)
    dist = torch.distributions.MultivariateNormal(theta_final, inference_cov)
    for i in tqdm(range(n_chunks), desc="Inference", disable=not verbose):
        a = i * inference_chunk_size
        b = min((i + 1) * inference_chunk_size, N)
        chunk_data = data[a:b]
        if torch.cuda.is_available() and chunk_data.device.type == "cpu":
            chunk_data = chunk_data.cuda()
        chunk_norm = (torch.log10(chunk_data + log_add) - data_log_mean) / data_log_std
        # Evaluate the same mixture posterior in log space to avoid all-class underflow.
        scores = dist.log_prob(chunk_norm[:, None, :]) + torch.log(w_star_final.clamp_min(torch.finfo(data.dtype).tiny))
        class_probs_chunk = torch.softmax(scores, dim=1)
        if not torch.isfinite(class_probs_chunk).all():
            raise ValueError('Nonfinite binary posteriors')

        probs_genes = class_probs_chunk[:, 0:K]
        probs_bkg = class_probs_chunk[:, [bkg_ind]].reshape(chunk_norm.shape[0], -1) if estimate_bkg \
            else torch.zeros((chunk_norm.shape[0], 0), device=class_probs_chunk.device)
        collapsed_probs = torch.cat((probs_genes, probs_bkg), dim=1)

        probs_safe = collapsed_probs.clamp(min=1e-10)
        entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1)
        k_top = min(2, collapsed_probs.shape[1])
        top_probs, top_indices = torch.topk(collapsed_probs, k=k_top, dim=1)
        res_indices[a:b] = top_indices[:, 0].cpu().numpy()
        res_probs_1[a:b] = top_probs[:, 0].cpu().numpy()
        res_entropy[a:b] = entropy.cpu().numpy()
        res_probs_2[a:b] = top_probs[:, 1].cpu().numpy() if k_top >= 2 else 0.0
        del chunk_norm, class_probs_chunk, collapsed_probs, probs_safe, top_probs, top_indices, entropy

    mapped_genes = np.empty(N, dtype=object)
    mapped_seqs = np.empty(N, dtype=object)
    gene_lookup = np.array(gene_list)
    seq_lookup = np.array(code_list)
    gene_mask = res_indices < K
    mapped_genes[gene_mask] = gene_lookup[res_indices[gene_mask]]
    mapped_seqs[gene_mask] = seq_lookup[res_indices[gene_mask]]
    bkg_col_idx = K if estimate_bkg else -1
    if bkg_col_idx != -1:
        bm = res_indices == bkg_col_idx
        mapped_genes[bm] = "Background"
        mapped_seqs[bm] = "Background"
    other = ~(gene_mask | (res_indices == bkg_col_idx))
    mapped_genes[other] = np.nan
    mapped_seqs[other] = np.nan
    # ----- end replicated region -----

    result_df = pd.DataFrame({
        "index": np.arange(n_spots),
        "Sequence": mapped_seqs,
        "Gene": mapped_genes,
        "Probability": res_probs_1,
        "Probability_2": res_probs_2,
        "Entropy": res_entropy,
    })
    diagnostics = {"losses": losses, "n_training_spots": int(len(ind_keep)), "n_total_spots": int(N)}
    diagnostics['normalization'] = {'offset': log_add.detach().cpu().tolist(), 'mean': data_log_mean.detach().cpu().tolist(), 'std': data_log_std.detach().cpu().tolist()}
    diagnostics['model'] = {'weights': w_star_final.cpu().tolist(), 'theta': theta_final.cpu().tolist(), 'covariance': sigma_star.cpu().tolist()}
    if log_brightness_variance:
        diagnostics['model'].update(
            log_brightness_variance=float(log_brightness_variance),
            inference_covariances=inference_cov.detach().cpu().tolist(),
            inference_codes=codes.detach().cpu().tolist(),
        )
    diagnostics['acquired_rounds'] = 1
    diagnostics['phasing_correction'] = False
    diagnostics['error_correction_redundancy'] = False
    return result_df, diagnostics




