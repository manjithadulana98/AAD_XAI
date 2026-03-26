"""Tests for deep model forward passes and basic properties."""
import pytest
import torch
import numpy as np

from aad_xai.models.aadnet import AADNet
from aad_xai.models.stgcn import STGCN


class TestAADNet:
    def test_forward_shape(self):
        model = AADNet(n_channels=64, n_classes=2)
        x = torch.randn(4, 64, 128)
        out = model(x)
        assert out.shape == (4, 2)

    def test_feature_extractor_accessible(self):
        model = AADNet(n_channels=32)
        assert hasattr(model, "feature_extractor")

    def test_gradient_flows(self):
        model = AADNet(n_channels=16)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestSTGCN:
    def test_forward_shape(self):
        model = STGCN(n_channels=32, n_classes=2)
        x = torch.randn(4, 32, 128)
        out = model(x)
        assert out.shape == (4, 2)

    def test_custom_adjacency(self):
        adj = torch.eye(16) + torch.randn(16, 16).abs() * 0.1
        model = STGCN(n_channels=16, adj=adj)
        x = torch.randn(2, 16, 64)
        out = model(x)
        assert out.shape == (2, 2)

    def test_gradient_flows(self):
        model = STGCN(n_channels=16)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
