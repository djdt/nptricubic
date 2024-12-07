import numpy as np

from nptricubic.matrix import M


def calculate_cubic_coef(
    data: np.ndarray, xi: np.ndarray, yi: np.ndarray, zi: np.ndarray
) -> np.ndarray:
    """Create the tricubic coefficients :math:`a_{ijk}`.

    Calculates the 64 coefficients at each (``xi``, ``yi``, ``zi``).

    Args:
        data: 3d array of values
        xi: x indicies for interpolation
        yi: y indicies for interpolation
        zi: z indicies for interpolation

    Returns:
        tricubic interpolation coefficients, shape (n, 64)
    """

    def square_index(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[tuple, tuple, tuple]:
        """Indicies equivalent to x:x+2,y:y+2,z:z+2"""
        return (
            (x, x + 1, x, x + 1, x, x + 1, x, x + 1),
            (y, y, y + 1, y + 1, y, y, y + 1, y + 1),
            (z, z, z, z, z + 1, z + 1, z + 1, z + 1),
        )

    # these scalings going to change?
    f = data[square_index(xi, yi, zi)]

    dfdx = 0.5 * (
        data[square_index(xi + 1, yi, zi)] - data[square_index(xi - 1, yi, zi)]
    )

    dfdy = 0.5 * (
        data[square_index(xi, yi + 1, zi)] - data[square_index(xi, yi - 1, zi)]
    )
    dfdz = 0.5 * (
        data[square_index(xi, yi, zi + 1)] - data[square_index(xi, yi, zi - 1)]
    )

    d2dxdy = 0.25 * (
        data[square_index(xi + 1, yi + 1, zi)]
        - data[square_index(xi - 1, yi + 1, zi)]
        - data[square_index(xi + 1, yi - 1, zi)]
        + data[square_index(xi - 1, yi - 1, zi)]
    )
    d2dxdz = 0.25 * (
        data[square_index(xi + 1, yi, zi + 1)]
        - data[square_index(xi - 1, yi, zi + 1)]
        - data[square_index(xi + 1, yi, zi - 1)]
        + data[square_index(xi - 1, yi, zi - 1)]
    )
    d2dydz = 0.25 * (
        data[square_index(xi, yi + 1, zi + 1)]
        - data[square_index(xi, yi - 1, zi + 1)]
        - data[square_index(xi, yi + 1, zi - 1)]
        + data[square_index(xi, yi - 1, zi - 1)]
    )
    d3dxdydz = 0.125 * (
        data[square_index(xi + 1, yi + 1, zi + 1)]
        - data[square_index(xi - 1, yi + 1, zi + 1)]
        - data[square_index(xi + 1, yi - 1, zi + 1)]
        + data[square_index(xi - 1, yi - 1, zi + 1)]
        - data[square_index(xi + 1, yi + 1, zi - 1)]
        + data[square_index(xi - 1, yi + 1, zi - 1)]
        + data[square_index(xi + 1, yi - 1, zi - 1)]
        - data[square_index(xi - 1, yi - 1, zi - 1)]
    )
    b = np.concatenate([f, dfdx, dfdy, dfdz, d2dxdy, d2dxdz, d2dydz, d3dxdydz])

    a = np.dot(M, b)
    return a.reshape(4, 4, 4, -1).T.reshape(-1, 64)


def tricubic_interp(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    data: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
):
    """Tri-cubic interpolation of points (x, y, z).
    Based on the implemention by Lekien and Marsden [1].

    Arrays ``x``, ``y`` and ``z`` must be 1d and have the same shape.
    The ``data`` array must be 3d with shape (``xs``, ``ys``, ``zs``).

    Args:
        x: 1d array of x values for interpolation
        y: 1d array of y values for interpolation
        z: 1d array of z values for interpolation
        data: values of the regular mesh
        xs: x values of the mesh
        ys: y values of the mesh
        zs: z values of the mesh

    Returns:
        interpolated values at (x, y, z)

    References:
        .. [1] Lekien, F., & Marsden, J. (2005). Tricubic interpolation in three
            dimensions. International Journal for Numerical Methods in Engineering,
            63(3), 455–471. doi:10.1002/nme.1296
    """
    x, y, z = (
        np.atleast_1d(x).ravel(),
        np.atleast_1d(y).ravel(),
        np.atleast_1d(z).ravel(),
    )

    if x.size != y.size != z.size:
        raise ValueError("x, y and z must have same size")
    if data.shape != (xs.size, ys.size, zs.size):
        raise ValueError("data must have shape (xs, ys, zs)")

    xi = np.searchsorted(xs, x, side="right") - 1
    yi = np.searchsorted(ys, y, side="right") - 1
    zi = np.searchsorted(zs, z, side="right") - 1

    xi = np.clip(xi, 1, xs.size - 3)
    yi = np.clip(yi, 1, ys.size - 3)
    zi = np.clip(zi, 1, zs.size - 3)

    # normalise values to cube
    x = (x - xs[xi]) / (xs[xi + 1] - xs[xi])
    y = (y - ys[yi]) / (ys[yi + 1] - ys[yi])
    z = (z - zs[zi]) / (zs[zi + 1] - zs[zi])

    a = calculate_cubic_coef(data, xi, yi, zi)

    # create arrays of powers
    xp = np.repeat([0, 1, 2, 3], 16)
    yp = np.tile(np.repeat([0, 1, 2, 3], 4), 4)
    zp = np.tile([0, 1, 2, 3], 16)

    val = np.sum(
        a
        * np.power(x[:, None], xp)
        * np.power(y[:, None], yp)
        * np.power(z[:, None], zp),
        axis=1,
    )
    return val
