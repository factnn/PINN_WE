"""
Unit tests for utility functions.
"""
import pytest
import torch
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PINNsrc.utility import select_gpu, save_results


class TestGPUSelection:
    """Test GPU selection utilities."""

    def test_select_gpu_returns_valid(self):
        """Test that select_gpu returns valid GPU ID or None."""
        gpu_id = select_gpu(default_auto=True)
        
        if torch.cuda.is_available():
            assert gpu_id is None or (isinstance(gpu_id, int) and gpu_id >= 0)
        else:
            assert gpu_id is None

    def test_select_gpu_cuda_availability(self):
        """Test GPU selection based on CUDA availability."""
        if not torch.cuda.is_available():
            result = select_gpu(default_auto=True)
            assert result is None


class TestSaveResults:
    """Test result saving functionality."""

    def test_save_results_basic(self):
        """Test basic result saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple model
            model = torch.nn.Linear(2, 3)
            
            # Dummy predictions
            pred_dict = {
                'x': torch.randn(10, 1).numpy(),
                'rho': torch.randn(10, 1).numpy(),
                'u': torch.randn(10, 1).numpy(),
                'p': torch.randn(10, 1).numpy(),
            }
            
            config = {'lr': 0.001, 'epochs': 100}
            
            # Save results
            output_dir, metrics = save_results(
                case_name='test_case',
                model=model,
                pred_dict=pred_dict,
                config=config,
                save_model=True,
                save_plot=False,  # Skip plotting to avoid display issues
                base_dir=tmpdir
            )
            
            assert os.path.exists(output_dir)
            assert os.path.isfile(os.path.join(output_dir, 'config.json'))


class TestVisualization:
    """Test visualization utilities."""

    def test_plot_generation(self):
        """Test that plots can be generated without errors."""
        # This is a minimal test - full visual testing would require image comparison
        import matplotlib.pyplot as plt
        
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.close(fig)
        
        assert True  # If we get here without exception, test passes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
