import numpy as np
import pytest
import torch

from prism.vis_binary import brightness_covariances


def test_weighted_mean_and_log_brightness_direction():
    cov = torch.eye(2)
    codes = torch.tensor([[1., 1.], [0., 1.], [0., 0.]])
    sd = torch.tensor([[1., 2.]])
    weights = torch.tensor([.3, .5, .2])
    assert brightness_covariances(cov, codes, sd, weights, 0) is cov
    result = brightness_covariances(cov, codes, sd, weights, .1)
    assert torch.allclose((result * weights[:, None, None]).sum(0), cov, atol=1e-6)
    assert torch.allclose(result[0] - result[2], torch.tensor([[.1, .05], [.05, .025]]), atol=1e-6)
    with pytest.raises(ValueError, match="covariance budget"):
        brightness_covariances(cov, codes, sd, weights, 100.)
    for invalid in [np.nan, np.inf, -1]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            brightness_covariances(cov, codes, sd, weights, invalid)


def test_brightness_posteriors_replay_from_saved_model():
    from scipy.special import softmax
    from scipy.stats import multivariate_normal
    from prism.vis_binary import decode

    rng = np.random.default_rng(11)
    words = [f"{i:05b}" for i in range(1, 32)]
    bits = np.array([list(word) for word in words], dtype=np.float32)
    labels = np.concatenate([np.repeat(np.arange(31), 16), np.full(320, 31)])
    panel = np.vstack([bits, np.zeros((1, 5))])
    values = rng.poisson(15 + 900 * panel[labels]).astype(np.float32)
    result, diag = decode(values[:, :, None], bits[:, :, None], words, words,
                          num_iter=20, batch_size=256, inference_chunk_size=128,
                          verbose=False, log_brightness_variance=.001)
    model = diag["model"]
    scale = diag["normalization"]
    normalized = (np.log10(values + scale["offset"]) - scale["mean"]) / scale["std"]
    scores = np.column_stack([
        np.log(w) + multivariate_normal.logpdf(normalized, mean=mean, cov=cov)
        for w, mean, cov in zip(model["weights"], model["theta"], model["inference_covariances"])
    ])
    posterior = softmax(scores, axis=1)
    assert np.array_equal(model["inference_codes"], panel)
    assert np.array_equal(result.Gene.to_numpy(), np.array(words + ["Background"])[posterior.argmax(1)])
    assert np.allclose(result.Probability, posterior.max(1), atol=2e-5)
