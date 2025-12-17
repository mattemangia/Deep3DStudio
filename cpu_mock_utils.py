#!/usr/bin/env python3
"""
CPU Mock Utilities for ONNX Export Scripts
==========================================
Provides comprehensive mocks for CUDA-only dependencies to enable
CPU-only ONNX export on systems without CUDA/GPU support.

This module must be imported BEFORE any other imports that might
trigger loading of spconv, flash_attn, xformers, etc.

Usage:
    # At the very top of your export script:
    from cpu_mock_utils import setup_cpu_only_environment
    setup_cpu_only_environment()

    # Then proceed with normal imports
    import torch
    import torch.nn as nn
"""

import os
import sys
from unittest.mock import MagicMock


def setup_cpu_only_environment(verbose=True):
    """
    Install comprehensive mocks for CUDA-only dependencies BEFORE they're imported.
    This is critical for CPU-only export on systems without CUDA.

    Args:
        verbose: If True, print status messages during setup
    """
    if verbose:
        print("=" * 60)
        print("Setting up CPU-only execution environment...")
        print("=" * 60)

    # Import torch early for nn.Module subclasses
    import torch
    import torch.nn as nn

    # =========================================================================
    # Mock spconv - Sparse Convolution library (requires CUDA)
    # =========================================================================
    class MockSparseConvTensor:
        """Mock for spconv.SparseConvTensor"""
        def __init__(self, features=None, indices=None, spatial_shape=None,
                     batch_size=None, grid=None, voxel_num=None, indice_dict=None):
            self.features = features
            self.indices = indices
            self.spatial_shape = spatial_shape
            self.batch_size = batch_size
            self.grid = grid
            self.voxel_num = voxel_num
            self.indice_dict = indice_dict or {}

        @property
        def sparity(self):
            return 0.5

        def dense(self, channels_first=True):
            """Convert sparse tensor to dense"""
            if self.features is not None:
                return self.features
            return torch.zeros(1)

        def replace_feature(self, new_features):
            self.features = new_features
            return self

    class MockSparseModule(nn.Module):
        """Base mock for sparse modules"""
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._args = args
            self._kwargs = kwargs

        def forward(self, x):
            return x

    class MockSparseConv3d(nn.Module):
        """Mock for spconv.SparseConv3d"""
        def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                     stride=1, padding=0, dilation=1, groups=1, bias=True,
                     indice_key=None, algo=None, *args, **kwargs):
            super().__init__()
            self.in_channels = in_channels or 1
            self.out_channels = out_channels or in_channels or 1
            self.kernel_size = kernel_size
            self.indice_key = indice_key
            # Create a simple weight tensor as placeholder
            if in_channels and out_channels:
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, 1) * 0.01)

        def forward(self, x):
            if isinstance(x, MockSparseConvTensor):
                return x
            return x

    class MockSubMConv3d(MockSparseConv3d):
        """Mock for spconv.SubMConv3d (submanifold sparse conv)"""
        pass

    class MockSparseInverseConv3d(MockSparseConv3d):
        """Mock for spconv.SparseInverseConv3d"""
        pass

    class MockSparseConvTranspose3d(MockSparseConv3d):
        """Mock for spconv.SparseConvTranspose3d"""
        pass

    class MockSparseSequential(nn.Sequential):
        """Mock for spconv.SparseSequential"""
        pass

    class MockSparseMaxPool3d(nn.Module):
        """Mock for spconv.SparseMaxPool3d"""
        def __init__(self, kernel_size=None, stride=None, padding=0, dilation=1, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class MockSparseAvgPool3d(MockSparseMaxPool3d):
        """Mock for spconv.SparseAvgPool3d"""
        pass

    # Create mock spconv modules
    spconv_mock = MagicMock()
    spconv_pytorch_mock = MagicMock()

    for mock_module in [spconv_mock, spconv_pytorch_mock]:
        mock_module.SparseConvTensor = MockSparseConvTensor
        mock_module.SparseConv3d = MockSparseConv3d
        mock_module.SubMConv3d = MockSubMConv3d
        mock_module.SparseInverseConv3d = MockSparseInverseConv3d
        mock_module.SparseConvTranspose3d = MockSparseConvTranspose3d
        mock_module.SparseModule = MockSparseModule
        mock_module.SparseSequential = MockSparseSequential
        mock_module.SparseMaxPool3d = MockSparseMaxPool3d
        mock_module.SparseAvgPool3d = MockSparseAvgPool3d
        # Algorithm enum mock
        mock_module.ConvAlgo = MagicMock()
        mock_module.ConvAlgo.Native = 0
        mock_module.ConvAlgo.MaskImplicitGemm = 1
        mock_module.ConvAlgo.MaskSplitImplicitGemm = 2

    # Register in sys.modules BEFORE any imports can happen
    sys.modules['spconv'] = spconv_mock
    sys.modules['spconv.pytorch'] = spconv_pytorch_mock
    sys.modules['spconv.core'] = MagicMock()
    sys.modules['spconv.utils'] = MagicMock()
    sys.modules['spconv.pytorch.core'] = MagicMock()
    sys.modules['spconv.pytorch.utils'] = MagicMock()
    sys.modules['spconv.pytorch.ops'] = MagicMock()
    sys.modules['spconv.pytorch.conv'] = spconv_pytorch_mock
    sys.modules['spconv.pytorch.modules'] = spconv_pytorch_mock
    sys.modules['spconv.pytorch.pool'] = spconv_pytorch_mock

    if verbose:
        print("  [OK] spconv mocked (SparseConv3d, SubMConv3d, SparseConvTensor, etc.)")

    # =========================================================================
    # Mock flash_attn - Flash Attention library (requires CUDA)
    # =========================================================================
    flash_attn_mock = MagicMock()

    def mock_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
        """Mock flash attention function using standard attention"""
        batch_size, seq_len, num_heads, head_dim = q.shape
        scale = softmax_scale or (head_dim ** -0.5)

        # Standard attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def mock_flash_attn_varlen_func(*args, **kwargs):
        """Mock variable length flash attention"""
        return mock_flash_attn_func(*args, **kwargs)

    flash_attn_mock.flash_attn_func = mock_flash_attn_func
    flash_attn_mock.flash_attn_varlen_func = mock_flash_attn_varlen_func

    sys.modules['flash_attn'] = flash_attn_mock
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_mock
    sys.modules['flash_attn.bert_padding'] = MagicMock()
    sys.modules['flash_attn.layers'] = MagicMock()
    sys.modules['flash_attn.layers.rotary'] = MagicMock()
    sys.modules['flash_attn.modules'] = MagicMock()
    sys.modules['flash_attn.modules.mha'] = MagicMock()

    if verbose:
        print("  [OK] flash_attn mocked")

    # =========================================================================
    # Mock xformers - Memory efficient attention (requires CUDA)
    # =========================================================================
    xformers_mock = MagicMock()
    xformers_ops_mock = MagicMock()

    def mock_memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
        """Mock xformers memory efficient attention"""
        scale = scale or (query.shape[-1] ** -0.5)
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    xformers_ops_mock.memory_efficient_attention = mock_memory_efficient_attention
    xformers_mock.ops = xformers_ops_mock

    sys.modules['xformers'] = xformers_mock
    sys.modules['xformers.ops'] = xformers_ops_mock
    sys.modules['xformers.components'] = MagicMock()
    sys.modules['xformers.components.attention'] = MagicMock()

    if verbose:
        print("  [OK] xformers mocked")

    # =========================================================================
    # Mock other CUDA-specific modules
    # =========================================================================
    # triton (used by some models)
    sys.modules['triton'] = MagicMock()
    sys.modules['triton.language'] = MagicMock()

    # apex (NVIDIA mixed precision)
    sys.modules['apex'] = MagicMock()
    sys.modules['apex.amp'] = MagicMock()
    sys.modules['apex.normalization'] = MagicMock()

    # fused_dense (TripoSG)
    sys.modules['fused_dense'] = MagicMock()
    sys.modules['fused_dense_lib'] = MagicMock()

    if verbose:
        print("  [OK] triton, apex, fused_dense mocked")
        print("=" * 60)
        print("CPU-only mocks installed successfully!")
        print("=" * 60 + "\n")


def force_cpu_execution():
    """
    Patch PyTorch CUDA functions to force CPU execution.
    Call this after importing torch.
    """
    import torch

    print("Forcing CPU execution by patching torch.cuda functions...")
    try:
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        torch.cuda.current_device = lambda: None
    except Exception as e:
        print(f"Warning: Could not patch torch.cuda functions: {e}")


def get_safe_device(requested_device='cpu'):
    """
    Get a safe device for execution, defaulting to CPU if CUDA unavailable.

    Args:
        requested_device: The requested device ('cpu', 'cuda', or 'cuda:N')

    Returns:
        str: The safe device to use
    """
    import torch

    if requested_device == 'cpu':
        return 'cpu'

    if 'cuda' in requested_device:
        if torch.cuda.is_available():
            return requested_device
        else:
            print(f"Warning: CUDA requested but not available. Using CPU.")
            return 'cpu'

    return 'cpu'


# Auto-setup when imported as main module
if __name__ == '__main__':
    setup_cpu_only_environment()
    print("\nCPU mock utilities loaded successfully!")
    print("You can now import torch and other libraries.")
