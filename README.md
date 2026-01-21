# Sbanks Core

<p align="center">
     <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
     <a href="https://github.com/atsyplenkov/sbanks-lib/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/atsyplenkov/sbanks-lib/ci.yml?style=flat&labelColor=1C2C2E&color=88AB26&logo=GitHub%20Actions&logoColor=white&label=CI"></a>
     <a href="https://pypi.org/project/sbanks-core/"><img src="https://img.shields.io/pypi/v/sbanks-core?style=flat&labelColor=1C2C2E&color=88AB26&logo=Python&logoColor=white"></a>
</p>

A pure Python library providing GIS-agnostic implementations of geometry smoothing algorithms. It serves as backend for the [sbanks](https://github.com/atsyplenkov/sbanks-tools) QGIS and ArcGIS plugins, offering Whittaker-Eilers and Savitzky-Golay filtering with optional spline resampling.

> Placeholder for performance plot or demo

## Installation

### From PyPI (⚠️ NOT YET AVAILABLE)
```bash
pip install sbanks-core
```

### Build from source
1. Clone the repository:
```bash
git clone https://github.com/atsyplenkov/sbanks-lib.git
```

2. Install using pip:
```bash
cd sbanks-lib
pip install .
```

## Usage

```python
import numpy as np
from sbanks_core import WhittakerSmoother, resample_and_smooth

# Sample coordinates
x = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
y = np.array([0.0, 5.2, 1.1, 6.3, 0.8, 4.9, 0.0])

# Resample and smooth using spline interpolation
x_smooth, y_smooth = resample_and_smooth(x, y, delta_s=5.0)

# Or use the Whittaker smoother directly
smoother = WhittakerSmoother(lam=10.0, d=2)
y_smoothed = smoother.smooth(y)
```

## Performance

This library provides a Python implementation of smoothing algorithms with minimal dependencies, using only `numpy` and `scipy`. This makes it easy to install and use in any Python environment without compilation or binary dependencies (like QGIS and ArcPro). It's even possible to install the lib or plugin inside extremely manageble environment (Apptainer/Docker). This is especially the case for the Whittaker-Eilers, which has already have a nice Python implementation with Rust core ([`whittaker-eilers`](https://pypi.org/project/whittaker-eilers/)).

Therefore, **for performance-critical applications**, consider using the [`whittaker-eilers`](https://pypi.org/project/whittaker-eilers/) package, which offers significantly faster smoothing operations (see benches below).

| Data Length | Metric         | sbanks (ms) | whittaker-eilers (ms) |
|-------------|----------------|------------:|----------------------:|
| n=100       | Initialization |       1.289 |                 0.194 |
|             | Smoothing      |       0.187 |                 0.011 |
|             | **Total**      |   **1.477** |             **0.205** |
| n=1,000     | Initialization |       1.329 |                 0.562 |
|             | Smoothing      |       1.012 |                 0.095 |
|             | **Total**      |   **2.341** |             **0.657** |
| n=10,000    | Initialization |       2.877 |                 6.473 |
|             | Smoothing      |       9.089 |                 0.979 |
|             | **Total**      |  **11.966** |             **7.453** |

*Benchmarks run on Intel Core i7-10710U @ 3.50GHz, 16GB RAM, Python 3.12.3. Lambda=10000, order=2.*

## License
This code is open-source and licensed under the GPL-3.0 or later license.

## Acknowledgements
Portions of the smoothing logic were inspired by [**Zoltán Sylvester's**](https://github.com/zsylvester) work on river meandering and the [channelmapper](https://github.com/zsylvester/channelmapper) repository.
