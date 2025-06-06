import torch
from jaxtyping import Float


def interp1d(
    x: Float[torch.Tensor, " b n"],
    y: Float[torch.Tensor, " b n"],
    xnew: Float[torch.Tensor, " b m"],
    mask_value: float | None = 0.0,
) -> Float[torch.Tensor, " b m"]:
    """Linear interpolation of a 1-D tensor using torch.searchsorted.
    Assumes that x and xnew are sorted in increasing order.

    Args:
        x: The x-coordinates of the data points, shape [batch, N].
        y: The y-coordinates of the data points, shape [batch, N].
        xnew: The x-coordinates of the interpolated points, shape [batch, M].
        mask_value: The value to use for xnew outside the range of x.
    Returns:
        The y-coordinates of the interpolated points, shape [batch, M].
    """
    # Find the indices where xnew should be inserted in sorted_x
    # Given a point xnew[i] in xnew, return j where x[j] is the nearest point in x such that
    # x[j] < xnew[i], except if the nearest point in x has x[j] = xnew[i] then return j - 1.
    indices = torch.searchsorted(x, xnew) - 1

    # We can define a local linear approx of the grad in each interval
    # between two points in x, and we would like to use this to interpolate
    # y at those points in xnew which lie inside the range of x, otherwise
    # interpolated_y is masked for points in xnew outside the range of x.
    # There are len(x) - 1 such intervals between points in x, having indices
    # ranging between 0 and len(x) - 2. Points with xnew < min(x) will be
    # assigned indices of -1 and points with xnew > max(x) will be assigned
    # indices equal to len(x). These are not valid segment indices, but we can
    # clamp them to 0 and len(x) - 2 respectively to avoid breaking the
    # calculation of the slope variable. The nonsense values we obtain outside
    # the range of x will be discarded when masking.
    indices = torch.clamp(indices, 0, x.shape[1] - 1 - 1)

    slopes = (y[:, :-1] - y[:, 1:]) / (x[:, :-1] - x[:, 1:])

    # Interpolate the y-coordinates
    ynew = torch.gather(y, 1, indices) + (
        xnew - torch.gather(x, 1, indices)
    ) * torch.gather(slopes, 1, indices)

    # Mask out the values that are outside the valid range
    mask = (xnew < x[..., 0].reshape(-1, 1)) | (xnew > x[..., -1].reshape(-1, 1))
    ynew[mask] = mask_value

    return ynew


class LatentSpectralGrid(torch.nn.Module):
    def __init__(self, lambda_min: float, resolution: float, num_pixels: int):
        """
        Initialize a latent grid to represent spectra from multiple resolutions.

        Args:
            lambda_min: The minimum wavelength value, in Angstrom.
            resolution: The resolution of the spectra, in Angstrom per pixel.
            num_pixels: The number of pixels in the spectra.

        """
        super().__init__()
        self.register_buffer("lambda_min", torch.tensor(lambda_min))
        self.register_buffer("resolution", torch.tensor(resolution))
        self.register_buffer("length", torch.tensor(num_pixels))
        self.register_buffer(
            "_wavelength",
            (torch.arange(0, num_pixels) * resolution + lambda_min).reshape(
                1, num_pixels
            ),
        )

    @property
    def wavelength(self) -> Float[torch.Tensor, " n"]:
        return self._wavelength.squeeze()

    def to_observed(
        self,
        x_latent: Float[torch.Tensor, " b n"],
        wavelength: Float[torch.Tensor, " b m"],
    ) -> Float[torch.Tensor, " b m"]:
        """Transforms the latent representation to the observed wavelength grid.

        Args:
            x_latent: The latent representation, [batch, self.num_pixels].
            wavelength: The observed wavelength grid, [batch, M].

        Returns:
            The transformed representation on the observed wavelength grid.
        """
        b = x_latent.shape[0]
        return interp1d(self._wavelength.repeat([b, 1]), x_latent, wavelength)

    def to_latent(
        self, x_obs: Float[torch.Tensor, "b m"], wavelength: Float[torch.Tensor, "b m"]
    ) -> Float[torch.Tensor, "b n"]:
        """Transforms the observed representation to the latent wavelength grid.

        Args:
            x_obs: The observed representation, [batch, N].
            wavelength: The wavelength grid, [batch, N].

        Returns:
            The transformed representation on the latent wavelength grid.
        """
        b = x_obs.shape[0]
        return interp1d(wavelength, x_obs, self._wavelength.repeat([b, 1]))
