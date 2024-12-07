import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nptricubic import tricubic_interp

# grid and interpolated values size
n, m = 100, 100


def test_tricubic_regular_mesh():
    xs = np.linspace(0.1, 1.0, n, endpoint=True)
    ys = np.linspace(1.0, 2.0, n, endpoint=True)
    zs = np.linspace(2.0, 10.0, n, endpoint=True)

    data = np.arange(n**3).reshape(n, n, n)

    x = np.random.uniform(0.2, 0.9, m)
    y = np.random.uniform(1.1, 1.9, m)
    z = np.random.uniform(2.1, 9.9, m)

    interp = RegularGridInterpolator((xs, ys, zs), data, method="cubic")
    scipy_vals = interp((x, y, z))
    vals = tricubic_interp(x, y, z, data, xs, ys, zs)

    assert np.allclose(vals, scipy_vals)
