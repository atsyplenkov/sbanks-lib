"""Shared fixtures and Hypothesis strategies for property-based testing."""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sbanks_core.whittaker import WhittakerSmoother


@st.composite
def signal_data(draw, min_size=10, max_size=200):
    """Generate various types of signal data for testing.

    Creates signals with different characteristics:
    - Random noise
    - Sinusoids (clean and noisy)
    - Linear trends
    - Monotonic sequences
    - Constant signals
    - Step functions

    Parameters
    ----------
    min_size : int
        Minimum signal length (default: 10)
    max_size : int
        Maximum signal length (default: 200)

    Returns
    -------
    np.ndarray
        Signal data with values in range [-100, 100]
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    signal_type = draw(
        st.sampled_from(
            [
                "random_noise",
                "sinusoid",
                "noisy_sinusoid",
                "linear_trend",
                "monotonic",
                "constant",
                "step_function",
            ]
        )
    )

    if signal_type == "random_noise":
        y = draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(
                    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
                ),
            )
        )

    elif signal_type == "sinusoid":
        freq = draw(st.floats(min_value=0.1, max_value=5.0))
        amplitude = draw(st.floats(min_value=1, max_value=50))
        phase = draw(st.floats(min_value=0, max_value=2 * np.pi))
        x = np.linspace(0, 10, n)
        y = amplitude * np.sin(2 * np.pi * freq * x + phase)

    elif signal_type == "noisy_sinusoid":
        freq = draw(st.floats(min_value=0.1, max_value=5.0))
        amplitude = draw(st.floats(min_value=1, max_value=50))
        phase = draw(st.floats(min_value=0, max_value=2 * np.pi))
        noise_level = draw(st.floats(min_value=0.1, max_value=0.5))
        x = np.linspace(0, 10, n)
        y = amplitude * np.sin(2 * np.pi * freq * x + phase)
        noise = draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(
                    min_value=-1, max_value=1, allow_nan=False, allow_infinity=False
                ),
            )
        )
        y = y + noise_level * amplitude * noise

    elif signal_type == "linear_trend":
        slope = draw(
            st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            )
        )
        intercept = draw(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            )
        )
        x = np.arange(n, dtype=np.float64)
        y = slope * x + intercept

    elif signal_type == "monotonic":
        # Generate strictly increasing sequence
        increments = draw(
            arrays(
                dtype=np.float64,
                shape=n - 1,
                elements=st.floats(
                    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        start = draw(
            st.floats(
                min_value=-50, max_value=50, allow_nan=False, allow_infinity=False
            )
        )
        y = np.concatenate([[start], start + np.cumsum(increments)])

    elif signal_type == "constant":
        value = draw(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            )
        )
        y = np.full(n, value, dtype=np.float64)

    else:  # step_function
        num_steps = draw(st.integers(min_value=2, max_value=min(10, n // 2)))
        step_positions = sorted(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=n - 1),
                    min_size=num_steps - 1,
                    max_size=num_steps - 1,
                    unique=True,
                )
            )
        )
        step_values = draw(
            arrays(
                dtype=np.float64,
                shape=num_steps,
                elements=st.floats(
                    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
                ),
            )
        )
        y = np.zeros(n, dtype=np.float64)
        step_positions = [0] + step_positions + [n]
        for i in range(num_steps):
            y[step_positions[i] : step_positions[i + 1]] = step_values[i]

    return y


@st.composite
def smoother_params(draw, min_length=5, max_length=200):
    """Generate valid WhittakerSmoother parameter combinations.

    Parameters
    ----------
    min_length : int
        Minimum data length (default: 5)
    max_length : int
        Maximum data length (default: 200)

    Returns
    -------
    dict
        Dictionary with keys: data_length, lmbda, order, x (optional)
    """
    data_length = draw(st.integers(min_value=min_length, max_value=max_length))

    # Lambda on logarithmic scale from 0 to 1e8
    lmbda = draw(
        st.one_of(
            st.just(0.0),
            st.floats(
                min_value=1e-2, max_value=1e8, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Order constrained by data_length (need at least order+1 points)
    max_order = min(4, data_length - 1)
    order = draw(st.integers(min_value=1, max_value=max_order))

    # Optional non-uniform x spacing
    use_nonuniform = draw(st.booleans())
    if use_nonuniform:
        # Generate strictly increasing x values
        increments = draw(
            arrays(
                dtype=np.float64,
                shape=data_length - 1,
                elements=st.floats(
                    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        x_input = np.concatenate([[0], np.cumsum(increments)])
    else:
        x_input = None

    return {
        "data_length": data_length,
        "lmbda": lmbda,
        "order": order,
        "x_input": x_input,
    }


@st.composite
def smoother_with_data(draw, min_length=5, max_length=200):
    """Generate WhittakerSmoother instance with compatible data.

    Ensures parameter compatibility between smoother and data.

    Parameters
    ----------
    min_length : int
        Minimum data length (default: 5)
    max_length : int
        Maximum data length (default: 200)

    Returns
    -------
    tuple
        (smoother, y_data, params) where:
        - smoother: WhittakerSmoother instance
        - y_data: np.ndarray signal data
        - params: dict with smoother parameters
    """
    params = draw(smoother_params(min_length=min_length, max_length=max_length))

    # Generate signal data with matching length
    y = draw(
        signal_data(min_size=params["data_length"], max_size=params["data_length"])
    )

    # Create smoother instance
    smoother = WhittakerSmoother(
        lmbda=params["lmbda"],
        order=params["order"],
        data_length=params["data_length"],
        x_input=params["x_input"],
    )

    return smoother, y, params
