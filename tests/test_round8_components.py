"""Round 8: configurable speech-encoder kernels, and training on 10 s windows
with evaluation on 5 s windows."""
from __future__ import annotations

import pytest
import torch

from aad_xai.models.sgat_speech_encoder import SharedSpeechEncoder
from tests.test_stimulus_gat import _make_stimulus_gat


def test_default_speech_kernels_unchanged():
    model = _make_stimulus_gat()
    convs = [m for m in model.speech_encoder.net if isinstance(m, torch.nn.Conv1d)]
    assert [c.kernel_size[0] for c in convs] == [9, 9]


def test_long_speech_kernels_keep_time_length():
    enc = SharedSpeechEncoder(embed_dim=6, hidden=4, kernel_sizes=(33, 33))
    out = enc(torch.randn(3, 320))                       # (B, T) -> (B, 6, T)
    assert out.shape == (3, 6, 320)


def test_even_speech_kernel_rejected():
    with pytest.raises(ValueError):
        SharedSpeechEncoder(kernel_sizes=(32, 33))


def test_long_speech_kernels_widen_receptive_field():
    """An impulse in the envelope reaches outputs up to (k1 + k2 - 2) / 2 samples away."""
    torch.manual_seed(0)
    enc = SharedSpeechEncoder(embed_dim=4, hidden=4, kernel_sizes=(33, 33), dropout=0.0).eval()
    env = torch.zeros(1, 200, requires_grad=True)
    enc(env)[0, :, 100].sum().backward()                 # output at t = 100
    reach = env.grad[0].nonzero().flatten()
    assert reach.min() >= 100 - 32 and reach.max() <= 100 + 32
    assert reach.max() - reach.min() > 17                 # wider than the (9, 9) default


def test_model_with_long_kernels_matches_forward_and_score_candidates():
    model = _make_stimulus_gat(film_pooling="corr", speech_kernel_sizes=(33, 33)).eval()
    eeg, env = torch.randn(2, 6, 320), torch.randn(2, 2, 320)
    torch.testing.assert_close(model(eeg, env), model.score_candidates(eeg, env))


def test_train_on_10s_then_score_5s_windows():
    """The model has no length-dependent layer: a step on 640-sample windows,
    then scoring 320-sample windows, gives finite (B, 2) logits."""
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = torch.nn.functional.cross_entropy(
        model.score_candidates(torch.randn(4, 6, 640), torch.randn(4, 2, 640)), torch.tensor([0, 1, 0, 1]))
    loss.backward(); opt.step()
    model.eval()
    logits = model(torch.randn(3, 6, 320), torch.randn(3, 2, 320))
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
