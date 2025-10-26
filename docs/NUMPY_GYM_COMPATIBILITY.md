# NumPy 2.0+ and Gym Compatibility Guide

## Overview

This document describes the compatibility considerations for NumPy 2.0+ and gym/gymnasium in Qlib.

## NumPy 2.0+ Compatibility

### Changes in NumPy 2.0+

NumPy 2.0 introduced several breaking changes that may affect Qlib functionality:

- Type promotion changes
- Deprecated APIs removed
- Changes to C API
- New copy semantics

### Qlib Compatibility Guards

Qlib now includes automatic detection and warnings for NumPy 2.0+:

```python
# In qlib/config.py
import numpy as np
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
if NUMPY_VERSION >= (2, 0):
    warnings.warn(
        f"NumPy {np.__version__} (2.0+) detected. Some APIs may have changed.",
        UserWarning
    )
```

### Recommended NumPy Versions

**For production use:**
- NumPy 1.x series: `numpy>=1.20.0,<2.0.0` (recommended)
- NumPy 2.x series: Use with caution, test thoroughly

**To install compatible NumPy:**
```bash
pip install 'numpy>=1.20.0,<2.0.0'
```

## Gym/Gymnasium Compatibility

### Gym Transition to Gymnasium

The original OpenAI Gym library has been deprecated in favor of Gymnasium:

- `gym` < 0.26: Legacy API
- `gym` >= 0.26: Major breaking changes
- `gymnasium`: Modern maintained version

### Qlib Compatibility Guards

Qlib detects gym availability and provides warnings:

```python
# In qlib/config.py
try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn(
        "gym not found. RL-related functionality will be unavailable.",
        UserWarning
    )
```

### Recommended Gym Versions

**For RL functionality (if needed):**

Option 1 - Legacy gym:
```bash
pip install 'gym<0.26'
```

Option 2 - Modern gymnasium:
```bash
pip install gymnasium
```

Note: If you don't use RL features in Qlib, gym/gymnasium is optional.

## Compatibility Testing

Qlib includes automated compatibility checks in `tests/test_compatibility.py`:

```bash
pytest tests/test_compatibility.py
```

This test verifies:
- NumPy version detection
- Gym availability detection
- Warning system functionality

## Troubleshooting

### NumPy 2.0+ Issues

If you encounter issues with NumPy 2.0+:

1. **Downgrade to NumPy 1.x:**
   ```bash
   pip install 'numpy<2.0.0' --force-reinstall
   ```

2. **Check for deprecated APIs:**
   - Review NumPy 2.0 migration guide
   - Update code to use new APIs

### Gym/Gymnasium Issues

If you encounter gym-related issues:

1. **For legacy code:**
   ```bash
   pip install 'gym<0.26'
   ```

2. **For modern code:**
   ```bash
   pip uninstall gym
   pip install gymnasium
   ```

3. **Update imports:**
   ```python
   # Old
   import gym
   
   # New
   import gymnasium as gym
   ```

## CI/CD Considerations

For continuous integration, pin dependencies to avoid unexpected breaks:

**requirements.txt:**
```
numpy>=1.20.0,<2.0.0
gym<0.26  # Optional, only if RL features are used
```

**setup.py:**
```python
install_requires=[
    'numpy>=1.20.0,<2.0.0',
    # ... other dependencies
],
extras_require={
    'rl': ['gym<0.26'],  # Optional RL features
}
```

## Migration Timeline

- **Current (v0.x):** NumPy 1.x, gym < 0.26 recommended
- **Future (v1.x):** Full NumPy 2.0+ and gymnasium support planned

## Resources

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI Gym to Gymnasium Migration](https://gymnasium.farama.org/content/migration-guide/)

## Reporting Issues

If you encounter compatibility issues:

1. Check this guide first
2. Review existing GitHub issues
3. Create a new issue with:
   - Your NumPy version (`pip show numpy`)
   - Your gym/gymnasium version (if applicable)
   - Error messages and stack traces
   - Minimal reproducible example
