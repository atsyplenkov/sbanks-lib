# Sbanks Core

<p align="center">
     <a href="https://github.com/atsyplenkov/sbanks-lib/.github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/atsyplenkov/sbanks-lib/ci.yml?style=flat&labelColor=1C2C2E&color=88AB26&logo=GitHub%20Actions&logoColor=white&label=CI"></a>
     <a href="https://pypi.org/project/sbanks-core/"><img src="https://img.shields.io/pypi/v/sbanks-core?style=flat&labelColor=1C2C2E&color=88AB26&logo=Python&logoColor=white"></a>
</p>

A pure Python library providing GIS-agnostic implementations of geometry smoothing algorithms. It serves as the core computational backend for the [sbanks](https://github.com/atsyplenkov/sbanks-tools) QGIS and ArcGIS plugins, offering Whittaker-Eilers and Savitzky-Golay filtering with optional spline resampling.

> Placeholder for performance plot or demo

## Features

* **Whittaker-Eilers Smoother**: Robust smoothing with support for unevenly spaced data (distance-aware).
* **Savitzky-Golay Filter**: Traditional polynomial smoothing for evenly spaced data.
* **Geometry Utilities**: Tools for coordinate processing, including Haversine distance calculations and ring padding.
* **GIS Agnostic**: Built on `numpy` and `scipy`, usable in any Python environment independent of QGIS or ArcPy.

## Installation

### From PyPI ()
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

# Example usage for smoothing coordinate arrays
coords = np.array([...])  # Your Nx2 array of coordinates
smoothed_coords = resample_and_smooth(
    coords,
    method='whittaker',
    lam=500,
    sampling_interval=10.0
)

```

## License
This code is open-source and licensed under the GPL-2.0+ license.

## Acknowledgements
Portions of the smoothing logic were inspired by [**Zoltán Sylvester's**](https://github.com/zsylvester) work on river meandering and the [channelmapper](https://github.com/zsylvester/channelmapper) repository.
