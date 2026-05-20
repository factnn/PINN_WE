"""
Unit tests for PINN model implementations.
"""
import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PINNsrc.PINNs import PINNs_WE_Euler_1D, PINNs_Euler_1D, PINNs_WE_Euler_2D


class TestPINNs1D:
    """Test 1D PINN models."""

    @pytest.fixture
    def model_we_1d(self):
        """Create a 1D weakly-enforced PINN model."""
        return PINNs_WE_Euler_1D(
            input_dim=2,
            output_dim=3,
            hidden_layers=4,
            hidden_units=20
        )

    @pytest.fixture
    def model_euler_1d(self):
        """Create a 1D Euler PINN model."""
        return PINNs_Euler_1D(
            input_dim=2,
            output_dim=3,
            hidden_layers=4,
            hidden_units=20
        )

    def test_model_initialization(self, model_we_1d):
        """Test that model initializes correctly."""
        assert model_we_1d is not None
        assert hasattr(model_we_1d, 'dnn')

    def test_forward_pass_shape(self, model_we_1d):
        """Test forward pass output shape."""
        x = torch.randn(100, 1)
        t = torch.randn(100, 1)
        xt = torch.cat([x, t], dim=1)
        
        output = model_we_1d(xt)
        assert output.shape == (100, 3)  # rho, u, p

    def test_forward_pass_no_nan(self, model_we_1d):
        """Test that forward pass doesn't produce NaN values."""
        x = torch.randn(50, 1)
        t = torch.randn(50, 1)
        xt = torch.cat([x, t], dim=1)
        
        output = model_we_1d(xt)
        assert not torch.isnan(output).any()

    def test_cuda_compatibility(self, model_we_1d):
        """Test CUDA device compatibility."""
        if torch.cuda.is_available():
            model_we_1d.cuda()
            x = torch.randn(50, 1).cuda()
            t = torch.randn(50, 1).cuda()
            xt = torch.cat([x, t], dim=1)
            
            output = model_we_1d(xt)
            assert output.is_cuda

    def test_loss_pde_computation(self, model_we_1d):
        """Test PDE loss computation."""
        if torch.cuda.is_available():
            model_we_1d.cuda()
            x = torch.randn(50, 1, requires_grad=True).cuda()
            
            loss = model_we_1d.loss_pde(x, k=0.2)
            assert not torch.isnan(loss)
            assert loss.requires_grad


class TestPINNs2D:
    """Test 2D PINN models."""

    @pytest.fixture
    def model_we_2d(self):
        """Create a 2D weakly-enforced PINN model."""
        return PINNs_WE_Euler_2D(
            input_dim=3,
            output_dim=4,
            hidden_layers=4,
            hidden_units=20
        )

    def test_model_initialization(self, model_we_2d):
        """Test that 2D model initializes correctly."""
        assert model_we_2d is not None
        assert hasattr(model_we_2d, 'dnn')

    def test_forward_pass_shape(self, model_we_2d):
        """Test forward pass output shape for 2D."""
        x = torch.randn(100, 1)
        y = torch.randn(100, 1)
        t = torch.randn(100, 1)
        xyt = torch.cat([x, y, t], dim=1)
        
        output = model_we_2d(xyt)
        assert output.shape == (100, 4)  # rho, u, v, p

    def test_forward_pass_no_nan(self, model_we_2d):
        """Test that forward pass doesn't produce NaN values."""
        x = torch.randn(50, 1)
        y = torch.randn(50, 1)
        t = torch.randn(50, 1)
        xyt = torch.cat([x, y, t], dim=1)
        
        output = model_we_2d(xyt)
        assert not torch.isnan(output).any()


class TestGradients:
    """Test gradient computation utilities."""

    def test_gradients_basic(self):
        """Test basic gradient computation."""
        from PINNsrc.PINNs import gradients
        
        x = torch.randn(10, 1, requires_grad=True)
        y = x ** 2
        
        dy_dx = gradients(y, x)[0]
        
        assert dy_dx.shape == x.shape
        assert not torch.isnan(dy_dx).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
