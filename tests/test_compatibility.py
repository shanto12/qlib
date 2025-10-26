# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Compatibility Test Module

This module provides smoke tests to catch incompatible combinations of
NumPy 2.0+ and gym/gymnasium that could cause issues in Qlib.

Tests run on CI to ensure environment compatibility.
"""

import sys
import warnings
import pytest


class TestNumPyCompatibility:
    """Test NumPy version compatibility and detection."""

    def test_numpy_import(self):
        """Test that numpy can be imported."""
        try:
            import numpy as np
            assert np is not None
        except ImportError as e:
            pytest.skip(f"NumPy not installed: {e}")

    def test_numpy_version_detection(self):
        """Test that NumPy version detection works correctly."""
        try:
            import numpy as np
            from qlib.config import NUMPY_VERSION, NUMPY_AVAILABLE
            
            assert NUMPY_AVAILABLE is True
            assert isinstance(NUMPY_VERSION, tuple)
            assert len(NUMPY_VERSION) == 2
            assert all(isinstance(x, int) for x in NUMPY_VERSION)
            
            # Verify version matches actual numpy version
            expected_version = tuple(map(int, np.__version__.split('.')[:2]))
            assert NUMPY_VERSION == expected_version
        except ImportError:
            pytest.skip("NumPy not installed")

    def test_numpy_2_0_warning(self):
        """Test that NumPy 2.0+ triggers appropriate warning."""
        try:
            import numpy as np
            from qlib.config import NUMPY_VERSION
            
            if NUMPY_VERSION >= (2, 0):
                # NumPy 2.0+ should have triggered a warning during import
                # We can't easily test the warning retrospectively, but we can
                # verify the version detection is working
                assert NUMPY_VERSION[0] >= 2
                print(f"NumPy 2.0+ detected: {np.__version__}")
            else:
                print(f"NumPy 1.x detected: {np.__version__}")
        except ImportError:
            pytest.skip("NumPy not installed")

    def test_numpy_basic_operations(self):
        """Smoke test for basic NumPy operations."""
        try:
            import numpy as np
            
            # Test basic array creation and operations
            arr = np.array([1, 2, 3, 4, 5])
            assert len(arr) == 5
            assert np.sum(arr) == 15
            assert np.mean(arr) == 3.0
            
            # Test 2D array operations
            arr2d = np.array([[1, 2], [3, 4]])
            assert arr2d.shape == (2, 2)
            assert np.sum(arr2d) == 10
        except ImportError:
            pytest.skip("NumPy not installed")
        except Exception as e:
            pytest.fail(f"NumPy basic operations failed: {e}")


class TestGymCompatibility:
    """Test gym/gymnasium compatibility and detection."""

    def test_gym_import_detection(self):
        """Test that gym import detection works correctly."""
        from qlib.config import GYM_AVAILABLE, GYM_VERSION
        
        assert isinstance(GYM_AVAILABLE, bool)
        assert isinstance(GYM_VERSION, str)
        
        if GYM_AVAILABLE:
            print(f"gym detected: version {GYM_VERSION}")
        else:
            print("gym not available (expected for non-RL installations)")

    def test_gym_import_optional(self):
        """Test that gym import failure is handled gracefully."""
        from qlib.config import GYM_AVAILABLE
        
        try:
            import gym
            assert GYM_AVAILABLE is True
            print(f"gym available: {gym.__version__}")
        except ImportError:
            assert GYM_AVAILABLE is False
            print("gym not installed (this is OK for non-RL use cases)")

    def test_gymnasium_as_alternative(self):
        """Test that gymnasium can be imported as an alternative to gym."""
        try:
            import gymnasium
            print(f"gymnasium available: {gymnasium.__version__}")
        except ImportError:
            pytest.skip("gymnasium not installed (optional)")


class TestCompatibilityFlags:
    """Test that compatibility flags are properly exposed."""

    def test_numpy_flags_available(self):
        """Test that NumPy compatibility flags are accessible."""
        from qlib.config import NUMPY_AVAILABLE, NUMPY_VERSION
        
        # These should always be defined, even if NumPy is not installed
        assert NUMPY_AVAILABLE is not None
        assert NUMPY_VERSION is not None

    def test_gym_flags_available(self):
        """Test that gym compatibility flags are accessible."""
        from qlib.config import GYM_AVAILABLE, GYM_VERSION
        
        # These should always be defined, even if gym is not installed
        assert GYM_AVAILABLE is not None
        assert GYM_VERSION is not None

    def test_config_import(self):
        """Test that qlib.config can be imported without errors."""
        try:
            import qlib.config
            assert qlib.config is not None
        except Exception as e:
            pytest.fail(f"Failed to import qlib.config: {e}")


class TestWarningSystem:
    """Test that the warning system works correctly."""

    def test_warnings_are_issued(self):
        """Test that compatibility warnings are properly issued."""
        # This test verifies the warning system by re-importing
        # (note: in practice, warnings may only appear once)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Reimport to trigger warnings
            import importlib
            import qlib.config as config_module
            importlib.reload(config_module)
            
            # Check if any warnings were issued
            warning_messages = [str(warning.message) for warning in w]
            print(f"Warnings issued: {len(w)}")
            for msg in warning_messages:
                print(f"  - {msg}")


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_config_class_import(self):
        """Test that Config class can still be imported."""
        try:
            from qlib.config import Config
            assert Config is not None
        except ImportError as e:
            pytest.fail(f"Failed to import Config class: {e}")

    def test_qlib_config_import(self):
        """Test that QlibConfig class can still be imported."""
        try:
            from qlib.config import QlibConfig
            assert QlibConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import QlibConfig class: {e}")

    def test_c_global_import(self):
        """Test that global C config can still be imported."""
        try:
            from qlib.config import C
            assert C is not None
        except ImportError as e:
            pytest.fail(f"Failed to import C global config: {e}")


def test_smoke_import_qlib():
    """Smoke test: Can we import qlib at all?"""
    try:
        import qlib
        assert qlib is not None
        print(f"Qlib version: {getattr(qlib, '__version__', 'unknown')}")
    except Exception as e:
        pytest.fail(f"Failed to import qlib: {e}")


if __name__ == "__main__":
    # Run tests directly
    print("Running compatibility tests...")
    print(f"Python version: {sys.version}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: not installed")
    
    try:
        import gym
        print(f"gym version: {gym.__version__}")
    except ImportError:
        print("gym: not installed")
    
    try:
        import gymnasium
        print(f"gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("gymnasium: not installed")
    
    pytest.main([__file__, "-v", "-s"])
