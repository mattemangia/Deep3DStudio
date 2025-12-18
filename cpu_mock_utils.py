#!/usr/bin/env python3
"""
CPU Mock Utilities for ONNX Export Scripts
==========================================
Provides comprehensive mocks for ALL CUDA-only dependencies to enable
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

Mocked Dependencies:
    - spconv (Sparse Convolution) - All components
    - flash_attn (Flash Attention) - All modules
    - xformers (Memory Efficient Attention) - All modules
    - triton (GPU Programming Framework)
    - apex (NVIDIA Mixed Precision)
    - fused_dense (Fused Operations)
    - bitsandbytes (Quantization)
    - pytorch3d (3D Vision)
    - MinkowskiEngine (Sparse Networks)
    - cupy (CUDA Arrays)
    - nvdiffrast (NVIDIA Differentiable Rendering)
    - tiny-cuda-nn (NVIDIA Tiny CUDA Neural Networks)
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
                     batch_size=None, grid=None, voxel_num=None, indice_dict=None,
                     benchmark=False, *args, **kwargs):
            self.features = features if features is not None else torch.zeros(1, 1)
            self.indices = indices if indices is not None else torch.zeros(1, 4, dtype=torch.int32)
            self.spatial_shape = spatial_shape or [1, 1, 1]
            self.batch_size = batch_size or 1
            self.grid = grid
            self.voxel_num = voxel_num
            self.indice_dict = indice_dict or {}
            self.benchmark = benchmark

        @property
        def sparity(self):
            return 0.5

        @property
        def shape(self):
            return self.features.shape

        def dense(self, channels_first=True):
            """Convert sparse tensor to dense"""
            if self.features is not None:
                return self.features
            return torch.zeros(1)

        def replace_feature(self, new_features):
            self.features = new_features
            return self

        def to(self, device):
            self.features = self.features.to(device)
            self.indices = self.indices.to(device)
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
                     indice_key=None, algo=None, fp32_accum=None, *args, **kwargs):
            super().__init__()
            self.in_channels = in_channels or 1
            self.out_channels = out_channels or in_channels or 1
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.indice_key = indice_key
            # Create weight tensor as placeholder
            if in_channels and out_channels:
                k = kernel_size if isinstance(kernel_size, int) else (kernel_size[0] if kernel_size else 3)
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, k, k, k) * 0.01)
                if bias:
                    self.bias = nn.Parameter(torch.zeros(out_channels))

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

    class MockSparseGlobalMaxPool(nn.Module):
        """Mock for spconv.SparseGlobalMaxPool"""
        def forward(self, x):
            if isinstance(x, MockSparseConvTensor):
                return x.features.max(dim=0)[0]
            return x

    class MockSparseGlobalAvgPool(nn.Module):
        """Mock for spconv.SparseGlobalAvgPool"""
        def forward(self, x):
            if isinstance(x, MockSparseConvTensor):
                return x.features.mean(dim=0)
            return x

    class MockSparseBatchNorm(nn.Module):
        """Mock for spconv.SparseBatchNorm"""
        def __init__(self, num_features, *args, **kwargs):
            super().__init__()
            self.bn = nn.BatchNorm1d(num_features)

        def forward(self, x):
            if isinstance(x, MockSparseConvTensor):
                x.features = self.bn(x.features)
                return x
            return x

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
        mock_module.SparseGlobalMaxPool = MockSparseGlobalMaxPool
        mock_module.SparseGlobalAvgPool = MockSparseGlobalAvgPool
        mock_module.SparseBatchNorm = MockSparseBatchNorm
        # Algorithm enum mock
        mock_module.ConvAlgo = MagicMock()
        mock_module.ConvAlgo.Native = 0
        mock_module.ConvAlgo.MaskImplicitGemm = 1
        mock_module.ConvAlgo.MaskSplitImplicitGemm = 2

    # Register ALL spconv modules
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
    sys.modules['spconv.pytorch.functional'] = MagicMock()
    sys.modules['spconv.pytorch.tables'] = MagicMock()
    sys.modules['spconv.pytorch.hash'] = MagicMock()
    sys.modules['spconv.pytorch.cppcore'] = MagicMock()
    sys.modules['spconv.cppconstants'] = MagicMock()

    if verbose:
        print("  [OK] spconv mocked (SparseConv3d, SubMConv3d, SparseConvTensor, etc.)")

    # =========================================================================
    # Mock flash_attn - Flash Attention library (requires CUDA)
    # =========================================================================
    flash_attn_mock = MagicMock()
    flash_attn_interface_mock = MagicMock()

    def mock_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                             return_attn_probs=False, *args, **kwargs):
        """Mock flash attention function using standard attention"""
        # Handle different input shapes
        if q.dim() == 4:  # (batch, seqlen, heads, headdim)
            batch_size, seq_len, num_heads, head_dim = q.shape
        elif q.dim() == 3:  # (batch * heads, seqlen, headdim)
            batch_heads, seq_len, head_dim = q.shape
            batch_size = 1
            num_heads = batch_heads

        scale = softmax_scale or (head_dim ** -0.5)

        # Reshape if needed
        if q.dim() == 4:
            q_reshaped = q.transpose(1, 2)  # (batch, heads, seqlen, headdim)
            k_reshaped = k.transpose(1, 2)
            v_reshaped = v.transpose(1, 2)
        else:
            q_reshaped = q
            k_reshaped = k
            v_reshaped = v

        # Standard attention computation
        attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * scale
        if causal:
            seq_len_q = q_reshaped.shape[-2]
            seq_len_k = k_reshaped.shape[-2]
            mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v_reshaped)

        # Reshape back if needed
        if q.dim() == 4:
            output = output.transpose(1, 2)  # (batch, seqlen, heads, headdim)

        if return_attn_probs:
            return output, attn_weights
        return output

    def mock_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                     dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
        """Mock variable length flash attention"""
        return mock_flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                    dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal).squeeze(0)

    def mock_flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
        """Mock QKV-packed flash attention"""
        q, k, v = qkv.unbind(dim=2)
        return mock_flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

    def mock_flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
        """Mock KV-packed flash attention"""
        k, v = kv.unbind(dim=2)
        return mock_flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

    flash_attn_mock.flash_attn_func = mock_flash_attn_func
    flash_attn_mock.flash_attn_varlen_func = mock_flash_attn_varlen_func
    flash_attn_mock.flash_attn_qkvpacked_func = mock_flash_attn_qkvpacked_func
    flash_attn_mock.flash_attn_kvpacked_func = mock_flash_attn_kvpacked_func
    flash_attn_interface_mock.flash_attn_func = mock_flash_attn_func
    flash_attn_interface_mock.flash_attn_varlen_func = mock_flash_attn_varlen_func
    flash_attn_interface_mock.flash_attn_qkvpacked_func = mock_flash_attn_qkvpacked_func
    flash_attn_interface_mock.flash_attn_kvpacked_func = mock_flash_attn_kvpacked_func

    # Register ALL flash_attn modules
    sys.modules['flash_attn'] = flash_attn_mock
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface_mock
    sys.modules['flash_attn.flash_attn_triton'] = flash_attn_mock
    sys.modules['flash_attn.bert_padding'] = MagicMock()
    sys.modules['flash_attn.layers'] = MagicMock()
    sys.modules['flash_attn.layers.rotary'] = MagicMock()
    sys.modules['flash_attn.modules'] = MagicMock()
    sys.modules['flash_attn.modules.mha'] = MagicMock()
    sys.modules['flash_attn.modules.mlp'] = MagicMock()
    sys.modules['flash_attn.modules.block'] = MagicMock()
    sys.modules['flash_attn.modules.embedding'] = MagicMock()
    sys.modules['flash_attn.ops'] = MagicMock()
    sys.modules['flash_attn.ops.fused_dense'] = MagicMock()
    sys.modules['flash_attn.ops.layer_norm'] = MagicMock()
    sys.modules['flash_attn.ops.rms_norm'] = MagicMock()
    sys.modules['flash_attn.utils'] = MagicMock()
    sys.modules['flash_attn.losses'] = MagicMock()
    sys.modules['flash_attn.losses.cross_entropy'] = MagicMock()

    if verbose:
        print("  [OK] flash_attn mocked (all modules)")

    # =========================================================================
    # Mock xformers - Memory efficient attention (requires CUDA)
    # =========================================================================
    xformers_mock = MagicMock()
    xformers_ops_mock = MagicMock()

    def mock_memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, *args, **kwargs):
        """Mock xformers memory efficient attention"""
        scale = scale or (query.shape[-1] ** -0.5)
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    def mock_memory_efficient_attention_forward(query, key, value, attn_bias=None, p=0.0, scale=None, *args, **kwargs):
        return mock_memory_efficient_attention(query, key, value, attn_bias, p, scale)

    class MockLowerTriangularMask:
        pass

    class MockAttentionBias:
        @staticmethod
        def from_seqlens(q_seqlen, kv_seqlen):
            return None

    xformers_ops_mock.memory_efficient_attention = mock_memory_efficient_attention
    xformers_ops_mock.memory_efficient_attention_forward = mock_memory_efficient_attention_forward
    xformers_ops_mock.LowerTriangularMask = MockLowerTriangularMask
    xformers_ops_mock.AttentionBias = MockAttentionBias
    xformers_ops_mock.fmha = MagicMock()
    xformers_ops_mock.fmha.attn_bias = MagicMock()
    xformers_ops_mock.fmha.attn_bias.LowerTriangularMask = MockLowerTriangularMask
    xformers_mock.ops = xformers_ops_mock

    # Register ALL xformers modules
    sys.modules['xformers'] = xformers_mock
    sys.modules['xformers.ops'] = xformers_ops_mock
    sys.modules['xformers.ops.fmha'] = xformers_ops_mock.fmha
    sys.modules['xformers.ops.fmha.attn_bias'] = xformers_ops_mock.fmha.attn_bias
    sys.modules['xformers.components'] = MagicMock()
    sys.modules['xformers.components.attention'] = MagicMock()
    sys.modules['xformers.components.attention.attention_patterns'] = MagicMock()
    sys.modules['xformers.components.feedforward'] = MagicMock()
    sys.modules['xformers.components.positional_embedding'] = MagicMock()
    sys.modules['xformers.components.multi_head_dispatch'] = MagicMock()
    sys.modules['xformers.factory'] = MagicMock()
    sys.modules['xformers.triton'] = MagicMock()
    sys.modules['xformers.profiler'] = MagicMock()

    if verbose:
        print("  [OK] xformers mocked (all modules)")

    # =========================================================================
    # Mock triton - GPU programming framework
    # =========================================================================
    triton_mock = MagicMock()
    triton_language_mock = MagicMock()

    # Common triton.language functions
    triton_language_mock.load = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.store = MagicMock()
    triton_language_mock.program_id = MagicMock(return_value=0)
    triton_language_mock.arange = MagicMock(return_value=torch.arange(1024))
    triton_language_mock.zeros = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.full = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.where = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.dot = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.sum = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.max = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.exp = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.log = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.sqrt = MagicMock(return_value=torch.zeros(1))
    triton_language_mock.sigmoid = MagicMock(return_value=torch.zeros(1))

    triton_mock.language = triton_language_mock
    triton_mock.jit = MagicMock(side_effect=lambda fn: fn)
    triton_mock.autotune = MagicMock(side_effect=lambda **kwargs: lambda fn: fn)
    triton_mock.heuristics = MagicMock(side_effect=lambda kwargs: lambda fn: fn)
    triton_mock.Config = MagicMock()

    # Create triton.backends mock with compiler submodule (needed by torch._inductor)
    triton_backends_mock = MagicMock()
    triton_backends_compiler_mock = MagicMock()
    triton_backends_mock.compiler = triton_backends_compiler_mock

    triton_mock.backends = triton_backends_mock

    # Create comprehensive triton.compiler mock with all submodules
    triton_compiler_mock = MagicMock()
    triton_compiler_compiler_mock = MagicMock()  # triton.compiler.compiler submodule
    triton_compiler_mock.compiler = triton_compiler_compiler_mock

    sys.modules['triton'] = triton_mock
    sys.modules['triton.language'] = triton_language_mock
    sys.modules['triton.runtime'] = MagicMock()
    sys.modules['triton.runtime.jit'] = MagicMock()
    sys.modules['triton.runtime.autotuner'] = MagicMock()
    sys.modules['triton.runtime.driver'] = MagicMock()
    sys.modules['triton.runtime.cache'] = MagicMock()
    sys.modules['triton.compiler'] = triton_compiler_mock
    sys.modules['triton.compiler.compiler'] = triton_compiler_compiler_mock
    sys.modules['triton.compiler.code_generator'] = MagicMock()
    sys.modules['triton.compiler.make_launcher'] = MagicMock()
    sys.modules['triton.compiler.errors'] = MagicMock()
    sys.modules['triton.ops'] = MagicMock()
    sys.modules['triton.testing'] = MagicMock()
    sys.modules['triton.backends'] = triton_backends_mock
    sys.modules['triton.backends.compiler'] = triton_backends_compiler_mock
    sys.modules['triton.backends.nvidia'] = MagicMock()
    sys.modules['triton.backends.nvidia.driver'] = MagicMock()
    sys.modules['triton.backends.amd'] = MagicMock()

    if verbose:
        print("  [OK] triton mocked")

    # =========================================================================
    # Mock apex - NVIDIA mixed precision training
    # =========================================================================
    apex_mock = MagicMock()
    apex_amp_mock = MagicMock()

    # Mock apex.amp.initialize
    def mock_amp_initialize(model, optimizer=None, opt_level="O1", *args, **kwargs):
        if optimizer is not None:
            return model, optimizer
        return model

    apex_amp_mock.initialize = mock_amp_initialize
    apex_amp_mock.scale_loss = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    apex_mock.amp = apex_amp_mock

    sys.modules['apex'] = apex_mock
    sys.modules['apex.amp'] = apex_amp_mock
    sys.modules['apex.normalization'] = MagicMock()
    sys.modules['apex.normalization.fused_layer_norm'] = MagicMock()
    sys.modules['apex.optimizers'] = MagicMock()
    sys.modules['apex.parallel'] = MagicMock()
    sys.modules['apex.fp16_utils'] = MagicMock()
    sys.modules['apex.multi_tensor_apply'] = MagicMock()
    sys.modules['apex.contrib'] = MagicMock()
    sys.modules['apex.transformer'] = MagicMock()

    if verbose:
        print("  [OK] apex mocked")

    # =========================================================================
    # Mock fused operations
    # =========================================================================
    fused_dense_mock = MagicMock()
    fused_dense_mock.FusedDense = nn.Linear
    fused_dense_mock.FusedMLP = MagicMock(return_value=nn.Sequential())

    sys.modules['fused_dense'] = fused_dense_mock
    sys.modules['fused_dense_lib'] = MagicMock()

    if verbose:
        print("  [OK] fused_dense mocked")

    # =========================================================================
    # Mock bitsandbytes - Quantization library (requires CUDA)
    # =========================================================================
    bnb_mock = MagicMock()

    class MockLinear8bitLt(nn.Linear):
        def __init__(self, input_features, output_features, bias=True, *args, **kwargs):
            super().__init__(input_features, output_features, bias)

    class MockLinear4bit(nn.Linear):
        def __init__(self, input_features, output_features, bias=True, *args, **kwargs):
            super().__init__(input_features, output_features, bias)

    bnb_mock.nn = MagicMock()
    bnb_mock.nn.Linear8bitLt = MockLinear8bitLt
    bnb_mock.nn.Linear4bit = MockLinear4bit
    bnb_mock.nn.Int8Params = MagicMock()
    bnb_mock.optim = MagicMock()
    bnb_mock.functional = MagicMock()

    sys.modules['bitsandbytes'] = bnb_mock
    sys.modules['bitsandbytes.nn'] = bnb_mock.nn
    sys.modules['bitsandbytes.optim'] = bnb_mock.optim
    sys.modules['bitsandbytes.functional'] = bnb_mock.functional
    sys.modules['bitsandbytes.cuda_setup'] = MagicMock()
    sys.modules['bitsandbytes.autograd'] = MagicMock()

    if verbose:
        print("  [OK] bitsandbytes mocked")

    # =========================================================================
    # Mock pytorch3d - 3D Computer Vision (requires CUDA)
    # =========================================================================
    pytorch3d_mock = MagicMock()

    sys.modules['pytorch3d'] = pytorch3d_mock
    sys.modules['pytorch3d.structures'] = MagicMock()
    sys.modules['pytorch3d.renderer'] = MagicMock()
    sys.modules['pytorch3d.ops'] = MagicMock()
    sys.modules['pytorch3d.loss'] = MagicMock()
    sys.modules['pytorch3d.transforms'] = MagicMock()
    sys.modules['pytorch3d.io'] = MagicMock()
    sys.modules['pytorch3d.utils'] = MagicMock()
    sys.modules['pytorch3d._C'] = MagicMock()

    if verbose:
        print("  [OK] pytorch3d mocked")

    # =========================================================================
    # Mock MinkowskiEngine - Sparse Neural Networks (requires CUDA)
    # =========================================================================
    ME_mock = MagicMock()

    class MockMinkowskiConvolution(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                     bias=False, dimension=3, *args, **kwargs):
            super().__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=kernel_size//2, dilation=dilation, bias=bias)

        def forward(self, x):
            return x

    ME_mock.MinkowskiConvolution = MockMinkowskiConvolution
    ME_mock.MinkowskiConvolutionTranspose = MockMinkowskiConvolution
    ME_mock.MinkowskiDepthwiseConvolution = MockMinkowskiConvolution
    ME_mock.MinkowskiBatchNorm = MagicMock(return_value=nn.Identity())
    ME_mock.MinkowskiReLU = MagicMock(return_value=nn.ReLU())
    ME_mock.MinkowskiGELU = MagicMock(return_value=nn.GELU())
    ME_mock.MinkowskiSiLU = MagicMock(return_value=nn.SiLU())
    ME_mock.MinkowskiPooling = MagicMock()
    ME_mock.MinkowskiGlobalPooling = MagicMock()
    ME_mock.SparseTensor = MagicMock()
    ME_mock.TensorField = MagicMock()

    sys.modules['MinkowskiEngine'] = ME_mock
    sys.modules['MinkowskiEngine.MinkowskiConvolution'] = MagicMock()
    sys.modules['MinkowskiEngine.MinkowskiPooling'] = MagicMock()
    sys.modules['MinkowskiEngine.MinkowskiNonlinearity'] = MagicMock()
    sys.modules['MinkowskiEngine.MinkowskiNormalization'] = MagicMock()
    sys.modules['MinkowskiEngine.MinkowskiSparseTensor'] = MagicMock()

    if verbose:
        print("  [OK] MinkowskiEngine mocked")

    # =========================================================================
    # Mock cupy - CUDA arrays (requires CUDA)
    # =========================================================================
    cupy_mock = MagicMock()
    cupy_mock.array = MagicMock(side_effect=lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x)
    cupy_mock.asnumpy = MagicMock(side_effect=lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
    cupy_mock.asarray = MagicMock(side_effect=lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x)

    sys.modules['cupy'] = cupy_mock
    sys.modules['cupy.cuda'] = MagicMock()
    sys.modules['cupy.cuda.runtime'] = MagicMock()
    sys.modules['cupyx'] = MagicMock()
    sys.modules['cupyx.scipy'] = MagicMock()

    if verbose:
        print("  [OK] cupy mocked")

    # =========================================================================
    # Mock nvdiffrast - NVIDIA Differentiable Rendering (requires CUDA)
    # =========================================================================
    nvdiffrast_mock = MagicMock()
    nvdiffrast_torch_mock = MagicMock()

    nvdiffrast_torch_mock.RasterizeGLContext = MagicMock()
    nvdiffrast_torch_mock.RasterizeCudaContext = MagicMock()
    nvdiffrast_torch_mock.rasterize = MagicMock(return_value=(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4)))
    nvdiffrast_torch_mock.interpolate = MagicMock(return_value=(torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3)))
    nvdiffrast_torch_mock.antialias = MagicMock(return_value=torch.zeros(1, 1, 1, 4))
    nvdiffrast_torch_mock.texture = MagicMock(return_value=torch.zeros(1, 1, 1, 3))

    nvdiffrast_mock.torch = nvdiffrast_torch_mock

    sys.modules['nvdiffrast'] = nvdiffrast_mock
    sys.modules['nvdiffrast.torch'] = nvdiffrast_torch_mock

    if verbose:
        print("  [OK] nvdiffrast mocked")

    # =========================================================================
    # Mock tiny-cuda-nn - NVIDIA Tiny CUDA Neural Networks (requires CUDA)
    # =========================================================================
    tcnn_mock = MagicMock()

    class MockTCNNNetwork(nn.Module):
        def __init__(self, n_input_dims, n_output_dims, network_config=None, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(n_input_dims, n_output_dims)

        def forward(self, x):
            return self.linear(x)

    class MockTCNNEncoding(nn.Module):
        def __init__(self, n_input_dims, encoding_config=None, *args, **kwargs):
            super().__init__()
            self.n_input_dims = n_input_dims
            # Default output dimensions based on common encodings
            self.n_output_dims = n_input_dims * 4

        def forward(self, x):
            return x.repeat(1, 4) if x.dim() == 2 else x

    class MockTCNNNetworkWithInputEncoding(nn.Module):
        def __init__(self, n_input_dims, n_output_dims, encoding_config=None, network_config=None, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(n_input_dims * 4, n_output_dims)

        def forward(self, x):
            x = x.repeat(1, 4) if x.dim() == 2 else x
            return self.linear(x)

    tcnn_mock.Network = MockTCNNNetwork
    tcnn_mock.Encoding = MockTCNNEncoding
    tcnn_mock.NetworkWithInputEncoding = MockTCNNNetworkWithInputEncoding

    sys.modules['tinycudann'] = tcnn_mock
    sys.modules['tiny-cuda-nn'] = tcnn_mock

    if verbose:
        print("  [OK] tiny-cuda-nn mocked")

    # =========================================================================
    # Mock kaolin - NVIDIA 3D Deep Learning (requires CUDA)
    # =========================================================================
    kaolin_mock = MagicMock()

    sys.modules['kaolin'] = kaolin_mock
    sys.modules['kaolin.ops'] = MagicMock()
    sys.modules['kaolin.ops.mesh'] = MagicMock()
    sys.modules['kaolin.ops.spc'] = MagicMock()
    sys.modules['kaolin.render'] = MagicMock()
    sys.modules['kaolin.render.mesh'] = MagicMock()
    sys.modules['kaolin.render.camera'] = MagicMock()
    sys.modules['kaolin.metrics'] = MagicMock()
    sys.modules['kaolin.io'] = MagicMock()

    if verbose:
        print("  [OK] kaolin mocked")

    # =========================================================================
    # Mock diff-gaussian-rasterization - 3D Gaussian Splatting (requires CUDA)
    # =========================================================================
    diff_gaussian_mock = MagicMock()

    sys.modules['diff_gaussian_rasterization'] = diff_gaussian_mock
    sys.modules['diff_gaussian_rasterization._C'] = MagicMock()

    if verbose:
        print("  [OK] diff-gaussian-rasterization mocked")

    # =========================================================================
    # Mock simple-knn - KNN for Gaussian Splatting (requires CUDA)
    # =========================================================================
    simple_knn_mock = MagicMock()
    simple_knn_mock.distCUDA2 = MagicMock(return_value=torch.zeros(1))

    sys.modules['simple_knn'] = simple_knn_mock
    sys.modules['simple_knn._C'] = MagicMock()

    if verbose:
        print("  [OK] simple-knn mocked")

    # =========================================================================
    # Mock pointops - Point Cloud Operations (requires CUDA)
    # =========================================================================
    pointops_mock = MagicMock()

    sys.modules['pointops'] = pointops_mock
    sys.modules['pointops._C'] = MagicMock()

    if verbose:
        print("  [OK] pointops mocked")

    # =========================================================================
    # Final message
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("CPU-only mocks installed successfully!")
        print("All CUDA dependencies have been mocked.")
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
        torch.cuda.get_device_name = lambda x=None: "CPU (mocked)"
        torch.cuda.get_device_capability = lambda x=None: (0, 0)
        torch.cuda.get_device_properties = lambda x=None: MagicMock(
            name="CPU (mocked)", total_memory=0, major=0, minor=0
        )
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


def filter_cuda_requirements(requirements_file):
    """
    Filter out CUDA-only packages from a requirements file.

    Args:
        requirements_file: Path to the requirements.txt file

    Returns:
        list: Filtered requirements that work on CPU
    """
    cuda_packages = [
        'flash-attn', 'flash_attn', 'xformers', 'spconv',
        'triton', 'apex', 'bitsandbytes', 'pytorch3d',
        'MinkowskiEngine', 'cupy', 'nvdiffrast', 'tinycudann',
        'tiny-cuda-nn', 'kaolin', 'diff-gaussian-rasterization',
        'simple-knn', 'pointops', 'fused-dense', 'fused_dense'
    ]

    filtered = []
    try:
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Check if this package should be filtered
                package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip().lower()
                if not any(cuda_pkg.lower() in package_name for cuda_pkg in cuda_packages):
                    filtered.append(line)
    except Exception as e:
        print(f"Warning: Could not filter requirements: {e}")
        return []

    return filtered


# Auto-setup when imported as main module
if __name__ == '__main__':
    setup_cpu_only_environment()
    print("\nCPU mock utilities loaded successfully!")
    print("You can now import torch and other libraries.")
