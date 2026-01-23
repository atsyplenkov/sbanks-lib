# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-23

### Added
- Initial release of sbanks-core library
- Whittaker-Eilers smoothing algorithm implementation
- Savitzky-Golay filter with custom padding logic
- Geometry utilities:
  - Haversine distance calculation for geographic coordinates
  - Cumulative distance computation
  - Anti-hook padding for open geometries
  - Ring padding for closed geometries
  - Spline-based resampling and smoothing
  - Endpoint snapping
  - Geometry densification
- Comprehensive test suite with unit and property-based tests
- PyPI package configuration
