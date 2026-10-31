"""Round 7: correlation pooling in FiLM, N-candidate scoring, same-trial
negatives and the trial-swap control permutation."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aad_xai.data.base import Trial
from aad_xai.data.negatives import (
    SameTrialNegativesDataset,
    cross_trial_permutation,
    sample_negative_starts,
)
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset
from aad_xai.models.sgat_film import TimeResolvedFiLM
from tests.test_stimulus_gat import _make_stimulus_gat


def _random_film(pooling):
    torch.manual_seed(0)
    film = TimeResolvedFiLM(speech_dim=3, n_features=4, pooling=pooling)
    with torch.no_grad():
        film.proj.weight.normal_()
    return film


# --- FiLM pooling -------------------------------------------------------------

def test_cov_pooling_matches_original_formula():
    film = _random_film("cov")
    h, s = torch.randn(2, 5, 4, 50), torch.randn(2, 3, 50)
    gamma, beta = film.proj(s).chunk(2, dim=1)
    expected = (h * (1 + gamma[:, None]) + beta[:, None]).mean(-1)
    torch.testing.assert_close(film(h, s), expected, atol=1e-5, rtol=1e-5)


def test_corr_pooling_is_pearson_correlation():
    film = _random_film("corr")
    h, s = torch.randn(2, 5, 4, 200), torch.randn(2, 3, 200)
    gamma, beta = film.proj(s).chunk(2, dim=1)
    term = (film(h, s) - h.mean(-1) - beta.mean(-1)[:, None]).detach().numpy()   # (B, C, F)
    hn, gn = h.numpy(), gamma.detach().numpy()
    for b in range(2):
        for c in range(5):
            for f in range(4):
                assert term[b, c, f] == pytest.approx(np.corrcoef(hn[b, c, f], gn[b, f])[0, 1], abs=1e-3)


def test_corr_candidate_difference_is_scale_invariant():
    """Scaling the EEG features changes mean_t(h) for both candidates equally
    and leaves the correlation unchanged, so the candidate difference is fixed."""
    film = _random_film("corr")
    h, sa, sb = torch.randn(2, 5, 4, 100), torch.randn(2, 3, 100), torch.randn(2, 3, 100)
    diff = film(h, sa) - film(h, sb)
    diff_scaled = film(3.0 * h + 1.0, sa) - film(3.0 * h + 1.0, sb)
    torch.testing.assert_close(diff, diff_scaled, atol=1e-4, rtol=1e-4)


def test_corr_pooling_finite_gradient_at_init():
    film = TimeResolvedFiLM(speech_dim=3, n_features=4, pooling="corr")   # zero weights: gamma constant
    h, s = torch.randn(2, 5, 4, 50), torch.randn(2, 3, 50)
    out = film(h, s)
    torch.testing.assert_close(out, h.mean(-1) + film.proj.bias[4:].detach()[None, None, :])
    out.pow(2).sum().backward()
    assert torch.isfinite(film.proj.weight.grad).all() and film.proj.weight.grad.abs().sum() > 0


def test_unknown_pooling_raises():
    with pytest.raises(ValueError):
        TimeResolvedFiLM(speech_dim=3, n_features=4, pooling="max")


# --- N-candidate scoring --------------------------------------------------------

@pytest.mark.parametrize("pooling", ["cov", "corr"])
def test_score_candidates_matches_forward_and_is_per_candidate(pooling):
    torch.manual_seed(1)
    model = _make_stimulus_gat(film_pooling=pooling).eval()
    eeg, env = torch.randn(3, 6, 128), torch.randn(3, 5, 128)
    with torch.no_grad():
        scores = model.score_candidates(eeg, env)                          # (3, 5)
        assert scores.shape == (3, 5)
        torch.testing.assert_close(model(eeg, env[:, :2]), scores[:, :2])
        # In eval mode a candidate's score does not depend on the others.
        torch.testing.assert_close(model(eeg, env[:, [4, 2]]), scores[:, [4, 2]])
        perm = [3, 0, 4, 1, 2]
        torch.testing.assert_close(model.score_candidates(eeg, env[:, perm]), scores[:, perm])


def test_forward_rejects_more_than_two_candidates():
    model = _make_stimulus_gat()
    with pytest.raises(ValueError):
        model(torch.randn(2, 6, 128), torch.randn(2, 3, 128))


def test_forward_attention_still_returned_per_candidate():
    model = _make_stimulus_gat().eval()
    logits, attn = model(torch.randn(2, 6, 128), torch.randn(2, 2, 128), return_attention_weights=True)
    assert logits.shape == (2, 2)
    assert attn["attn_a"].shape == attn["attn_b"].shape == (2, 2, 6, 6)


# --- Same-trial negatives -------------------------------------------------------

def test_sample_negative_starts_respects_gap_and_bounds():
    rng = np.random.default_rng(0)
    s = sample_negative_starts(rng, start=100, n_times=1000, win_len=320, min_gap=192, n=500)
    assert s.min() >= 0 and s.max() <= 1000 - 320
    assert np.all(np.abs(s - 100) >= 192)
    with pytest.raises(ValueError):
        sample_negative_starts(rng, start=0, n_times=330, win_len=320, min_gap=192, n=1)


def _ramp_trial(tid, n_times=1000, label=0, bands=None):
    # Envelope value at sample t encodes (stream, t): A = t, B = 10000 + t.
    t = np.arange(n_times, dtype=np.float32)
    a, b = t, 10000 + t
    if bands:
        a, b = np.stack([a] * bands), np.stack([b] * bands)
    return Trial(eeg=np.random.default_rng(0).standard_normal((4, n_times)).astype(np.float32),
                 sfreq=64.0, label=label, subject_id="S1", trial_id=tid, group_id="g",
                 audio_a=a, audio_b=b)


@pytest.mark.parametrize("bands", [None, 3])
def test_negatives_dataset_layout(bands):
    base = WindowedEEGAudioDataset([_ramp_trial("T1", bands=bands), _ramp_trial("T2", label=1, bands=bands)],
                                   window_s=5.0, overlap_s=2.5)
    ds = SameTrialNegativesDataset(base, n_per_stream=3, min_gap_s=3.0, seed=0)
    for i in range(len(ds)):
        eeg, env, label = ds[i]
        eeg0, env0, label0 = base[i]
        start = base.window_indices[i].start
        assert label == label0 and torch.equal(eeg, eeg0)
        assert env.shape == ((8, 320) if bands is None else (8, 3, 320))
        assert torch.equal(env[:2], env0)
        first = env[..., 0] if bands is None else env[:, 0, 0]               # (8,) encodes stream and start
        neg_a, neg_b = first[2:5].numpy(), first[5:8].numpy() - 10000
        assert np.all((neg_a >= 0) & (neg_a < 10000)) and np.all(neg_b >= 0)
        assert np.all(np.abs(neg_a - start) >= 192) and np.all(np.abs(neg_b - start) >= 192)


def test_negatives_are_redrawn_but_reproducible():
    base = WindowedEEGAudioDataset([_ramp_trial("T1")], window_s=5.0)
    d1, d2 = (SameTrialNegativesDataset(base, 3, 3.0, seed=7) for _ in range(2))
    first = d1[0][1]
    assert torch.equal(first, d2[0][1])
    assert any(not torch.equal(first, d1[0][1]) for _ in range(5))


# --- Trial-swap control -----------------------------------------------------------

def test_cross_trial_permutation():
    trials = np.array(["b", "a", "a", "c", "b", "a", "c", "d", "d"])
    perm = cross_trial_permutation(trials)
    assert sorted(perm) == list(range(len(trials)))
    assert np.all(trials[perm] != trials)
    with pytest.raises(ValueError):
        cross_trial_permutation(["a", "a", "a", "b"])
