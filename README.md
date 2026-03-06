# Sbanks

<p align="center">
     <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
     <a href="https://github.com/atsyplenkov/sbanks/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/atsyplenkov/sbanks/ci.yml?style=flat&labelColor=1C2C2E&color=88AB26&logo=GitHub%20Actions&logoColor=white&label=CI"></a>
     <a href="https://pypi.org/project/sbanks/"><img src="https://img.shields.io/pypi/v/sbanks?style=flat&labelColor=1C2C2E&color=88AB26&logo=Python&logoColor=white"></a>
</p>

A pure Python library providing GIS-agnostic implementations of geometry smoothing algorithms. It serves as backend for the [sbanks](https://github.com/atsyplenkov/sbanks-plugins) QGIS and ArcGIS plugins, offering Whittaker-Eilers and Savitzky-Golay filtering with optional spline resampling.

> Placeholder for performance plot or demo

## Installation

### From PyPI
```bash
pip install sbanks
```

### Build from source
1. Clone the repository:
```bash
git clone https://github.com/atsyplenkov/sbanks.git
```

2. Install using pip:
```bash
cd sbanks
pip install .
```

## Usage

```python
import numpy as np
from sbanks import WhittakerSmoother, resample_and_smooth

# Sample coordinates
x = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
y = np.array([0.0, 5.2, 1.1, 6.3, 0.8, 4.9, 0.0])

# Resample and smooth using spline interpolation
x_smooth, y_smooth = resample_and_smooth(x, y, delta_s=5.0)

# Or use the Whittaker smoother directly
smoother = WhittakerSmoother(lmbda=10.0, order=2, data_length=len(y))
y_smoothed = smoother.smooth(y)
```

- `smoothing_factor` is passed directly to SciPy's spline `s` parameter (absolute value, not scaled by point count).
- If spline fitting fails, `resample_and_smooth` emits `RuntimeWarning` and returns the original input coordinates unchanged.

## Performance

This library provides a Python implementation of smoothing algorithms with minimal dependencies, using only `numpy` and `scipy`. This makes it easy to install and use in any Python environment without compilation or binary dependencies (like QGIS and ArcPro). It's even possible to install the lib or plugin inside extremely manageble environment (Apptainer/Docker). This is especially the case for the Whittaker-Eilers, which has already have a nice Python implementation with Rust core ([`whittaker-eilers`](https://pypi.org/project/whittaker-eilers/)).

Therefore, **for performance-critical applications**, consider using the [`whittaker-eilers`](https://pypi.org/project/whittaker-eilers/) package, which offers significantly faster smoothing operations (see benches below).

| Data Length | Metric         | sbanks (ms) | whittaker-eilers (ms) |
|-------------|----------------|------------:|----------------------:|
| n=100       | Initialization |       0.981 |                 0.186 |
|             | Smoothing      |       0.048 |                 0.014 |
|             | **Total**      |   **1.029** |             **0.200** |
| n=1,000     | Initialization |       2.085 |                 0.372 |
|             | Smoothing      |       0.097 |                 0.099 |
|             | **Total**      |   **2.182** |             **0.472** |
| n=10,000    | Initialization |       8.739 |                 4.978 |
|             | Smoothing      |       0.864 |                 0.712 |
|             | **Total**      |   **9.604** |             **5.690** |

*Benchmarks run via `pytest tests/test_whittaker_comparison.py -m benchmark -v -s` on Intel(R) Core(TM) i7-10710U CPU, 15 GiB RAM, Python 3.12.3. Lambda=10000, order=2.*

## License
This code is open-source and licensed under the GPL-3.0 or later license.

## Acknowledgements
Portions of the smoothing logic were inspired by [**Zoltán Sylvester's**](https://github.com/zsylvester) work on river meandering and the [channelmapper](https://github.com/zsylvester/channelmapper) repository.
