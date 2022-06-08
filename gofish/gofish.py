import os
import warnings
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from .annulus import annulus
import matplotlib.pyplot as plt

__all__ = ['imagecube']
warnings.filterwarnings('ignore')


class imagecube(object):
    """
    Base class containing all the FITS data. Must be a 3D cube containing two
    spatial and one velocity axis for and spectral shifting. A 2D 'cube' can
    be used to make the most of the deprojection routines. These can easily be
    made from CASA using the ``exportfits()`` command.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (Optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
        v_range (Optional[tuple]): A tuple of minimum and maximum velocities
            to clip the velocity range to.
        verbose (Optional[bool]): Whether to print out warning messages.
        primary_beam (Optional[str]): Path to the primary beam as a FITS file
            to apply the correction.
        bunit (Optional[str]): If no `bunit` header keyword is found, use this
            value, e.g., 'Jy/beam'.
        pixel_scale (Optional[float]): If no axis information is found in the
            header, use this value for the pixel scaling in [arcsec], assuming
            an image centered on 0.0".
    """

    frequency_units = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
    velocity_units = {'km/s': 1e3, 'm/s': 1e0}

    def __init__(self, path, FOV=None, velocity_range=None, verbose=True,
                 clip=None, primary_beam=None, bunit=None, pixel_scale=None):

        # Default parameters for user-defined values.
        self._user_bunit = bunit
        self._user_pixel_scale = pixel_scale
        if self._user_pixel_scale is not None:
            self._user_pixel_scale /= 3600.0

        # Read in the FITS data.
        self._read_FITS(path)
        self.verbose = verbose

        # Primary beam correction.
        self._pb_corrected = False
        if primary_beam is not None:
            self.correct_PB(primary_beam)

        if clip is not None:
            print("WARNING: `clip` is depreciated, use `FOV` instead.")
            FOV = 2.0 * clip

        if FOV is not None:
            self._clip_cube_spatial(FOV/2.0)
        if velocity_range is not None:
            self._clip_cube_velocity(*velocity_range)
        if self.data.ndim == 3:
            self._velax_offset = self._calculate_symmetric_velocity_axis()
        if self.data.ndim != 3 and self.verbose:
            print("WARNING: Provided cube is only 2D. Shifting not available.")

    # -- Fishing Functions -- #

    def average_spectrum(self, r_min=None, r_max=None, dr=None,
                         PA_min=None, PA_max=None, exclude_PA=False,
                         abs_PA=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                         z0=None, psi=None, r_cavity=None, r_taper=None,
                         q_taper=None, z1=None, phi=None, z_func=None,
                         mstar=1.0, dist=100., resample=1, beam_spacing=False,
                         mask_frame='disk', unit='Jy/beam', mask=None,
                         skip_empty_annuli=True,
                         shadowed=False, empirical_uncertainty=False,
                         include_spectral_decorrelation=True,
                         velocity_resolution=1.0):
        """
        Return the averaged spectrum over a given radial range, returning a
        spectrum in units of [Jy/beam] or [K] using the Rayleigh-Jeans
        approximation.

        The `resample` parameter allows you to resample the
        spectrum at a different velocity spacing (by providing a float
        argument) or averaging of an integer number of channels (by providing
        an integer argument). For example, ``resample=3``, will return a
        velocity axis which has been supersampled such that a channel is three
        times as narrow as the intrinsic width. Instead, ``resample=50.0`` will
        result in a velocity axis with a channel spacing of 50 m/s.

        The third variable returned is the standard error on the mean of each
        velocity bin, i.e. the standard deviation of that velocity bin divided
        by the square root of the number of samples.

        Args:
            r_min (Optional[float]): Inner radius in [arcsec] of the region to
                integrate.
            r_max (Optional[float]): Outer radius in [arcsec] of the region to
                integrate.
            dr (Optional[float]): Width of the annuli to split the integrated
                region into in [arcsec]. Default is quater of the beam major
                axis.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mstar (Optional[float]): Stellar mass in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            resample (Optional[float/int]): Resampling parameter for the
                deprojected spectrum. An integer specifies an average of that
                many channels, while a float specifies the desired channel
                width. Default is ``resample=1``.
            beam_spacing(Optional[bool]): When extracting the annuli, whether
                to choose spatially independent pixels or not.
            PA_min (Optional[float]): Minimum polar angle to include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            PA_max (Optional[float]): Maximum polar angleto include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            exclude_PA (Optional[bool]): Whether to exclude pixels where
                ``PA_min <= PA_pix <= PA_max``.
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            unit (Optional[str]): Units for the spectrum, either ``'Jy/beam'``
                or ``'K'``. Note that the conversion to Kelvin assumes the
                Rayleigh-Jeans approximation which is typically invalid at
                sub-mm wavelengths.
            mask (Optional[ndarray]): Either a 2D or 3D mask to use when
                averaging the spectra. This will be used in addition to any
                geometrically defined mask.
            assume_correlated (Optional[bool]): Whether to treat the spectra
                which are stacked as correlated, by default this is
                ``True``. If ``False``, the uncertainty will be estimated using
                Poisson statistics, otherwise the uncertainty is just the
                standard deviation of each velocity bin.
            skip_empty_annuli (Optional[bool]): If ``True``, skip any annuli
                which are empty (i.e. their masks have zero pixels in). If
                ``False``, any empty masks will raise a ``ValueError``.
            empirical_uncertainty (Optional[bool]): If ``True``, use an
                empirical measure of the uncertainty based on an iterative
                sigma clipping described in ``imagecube.estimate_uncertainty``.
            include_spectral_decorrlation (Optional[bool]): If ``True``, take
                account of the spectral decorrelation when estimating
                uncertainties on the average spectrum and return a spectral
                correlation length too. Defaults to ``True``.
            velocity_resolution (Optional[float]): Velocity resolution of the
                data as a fraction of the channel spacing. Defaults to ``1.0``.

        Returns:
            The velocity axis of the spectrum, ``velax``, in [m/s], the
            averaged spectrum, ``spectrum``, and the variance of the velocity
            bin, ``scatter``. The latter two are in units of either [Jy/beam]
            or [K] depending on the ``unit``.
        """
        # Check is cube is 2D.

        self._test_2D()

        # Set the radial sampling. This should try and respect the annuli width
        # of `dr` as best as possible, but more strictly the minimum and
        # maximum radii.

        if dr is None:
            if self.dpix == self.bmaj:
                dr = 2.0 * self.dpix
            else:
                dr = self.bmaj / 4.0
        nbins = max(1, int(np.floor((r_max - r_min) / dr)))
        rbins = np.linspace(r_min, r_max, nbins + 1)
        if rbins.size < 2:
            raise ValueError("Unable to infer suitable `rbins`.")
        _, rvals = self.radial_sampling(rbins=rbins)

        # Keplerian velocity at the annulus centers.

        v_kep = self._keplerian(rpnts=rvals, mstar=mstar, dist=dist, inc=inc,
                                z0=z0, psi=psi, r_cavity=r_cavity,
                                r_taper=r_taper, q_taper=q_taper, z1=z1,
                                phi=phi, z_func=z_func)
        v_kep = np.atleast_1d(v_kep)

        # Output unit.

        unit = unit.lower()
        if unit not in ['mjy/beam', 'jy/beam', 'mk', 'k']:
            raise ValueError("Unknown `unit`.")
        if resample < 1.0 and self.verbose:
            print('WARNING: `resample < 1`, are you sure you want channels '
                  + 'narrower than 1 m/s?')

        # Pseduo masking. Make everything masked a NaN and then revert. The
        # annulus class should be able to remove any NaN values. We include a
        # line to allow for a 2D mask to be broadcast to the full 3D cube. Note
        # that this mask still allows for a mask to be specified with the
        # `PA_min` and `PA_max` arguments. This is combined when calculating
        # the number of pixels by considering any pixels which has at least one
        # finite value in the spectrum.

        if mask is not None:
            if mask.shape != self.data.shape:
                if mask.shape != self.data.shape[1:]:
                    raise ValueError("Unknown mask shape.")
                mask = np.ones(self.data.shape) * mask[None, :, :]
            saved_data = self.data.copy()
            self.data = np.where(mask, self.data, np.nan)
            user_mask = np.any(np.isfinite(self.data), axis=0)
        else:
            user_mask = np.ones(self.data[0].shape)

        # Calculate the number of independent samples for each pixel after
        # accounting for the spectral deprojection of the data. This is based
        # on a similar approach to Yen et al. (2016), but performed
        # numerically such that we can a) take account of any masks and b)
        # estimate the introduced spectral correlation.

        if include_spectral_decorrelation and not empirical_uncertainty:
            v0_map = self.keplerian(
                inc=inc,
                PA=PA,
                mstar=mstar,
                dist=dist,
                x0=x0,
                y0=y0,
                vlsr=0.0,
                z0=z0,
                psi=psi,
                r_cavity=r_cavity,
                r_taper=r_taper,
                q_taper=q_taper,
                z1=z1,
                phi=phi,
                z_func=z_func,
                r_min=r_min,
                r_max=r_max,
                cylindrical=False,
                shadowed=shadowed,
                )

            samples = self._independent_samples(
                v0_map=v0_map,
                mask=user_mask,
                velocity_resolution=velocity_resolution,
                plot=False,
                ignore_spectral_correlation=True,
                )
        else:
            samples = self.beams_per_pix

        # Get the deprojected spectrum for each annulus (or rval). We include
        # an array to describe whether an annulus is included in the average or
        # not in order to rescale the uncertainties.

        x_arr, y_arr, dy_arr, nbeams = [], [], [], []
        included = np.ones(rvals.size).astype('int')

        for ridx in range(rvals.size):
            try:
                annulus = self.get_annulus(rbins[ridx], rbins[ridx+1],
                                           x0=x0, y0=y0, inc=inc, PA=PA,
                                           z0=z0, psi=psi, z1=z1, phi=phi,
                                           r_taper=r_taper, q_taper=q_taper,
                                           r_cavity=r_cavity, z_func=z_func,
                                           beam_spacing=beam_spacing,
                                           PA_min=PA_min, PA_max=PA_max,
                                           exclude_PA=exclude_PA,
                                           abs_PA=abs_PA,
                                           mask_frame=mask_frame,
                                           shadowed=shadowed)

            # Complain if no spectra are found in the annulus.

            except ValueError:
                msg = "No finite pixels found between"
                msg += " {:.2f} and".format(rbins[ridx])
                msg += " {:.2f} arcsec.".format(rbins[ridx+1])
                if not skip_empty_annuli:
                    raise ValueError(msg)
                else:
                    included[ridx] = 0
                    if self.verbose:
                        print("WARNING: " + msg + " Skipping annulus.")
                continue

            # Deproject the spectrum currently using a simple bin average.
            # The try / except loop is that when masking, the spectrum values
            # are converted to NaNs which look like pixels in the calculation
            # of the annulus, but then cannot be binned.
            # TODO: See if there's a better way to combine these two checks.

            try:
                x, y, dy = annulus.deprojected_spectrum(vrot=v_kep[ridx],
                                                        resample=resample,
                                                        scatter=True)
            except ValueError:
                msg = "No finite pixels found between"
                msg += " {:.2f} and".format(rbins[ridx])
                msg += " {:.2f} arcsec.".format(rbins[ridx+1])
                if not skip_empty_annuli:
                    raise ValueError(msg)
                else:
                    included[ridx] = 0
                    if self.verbose:
                        print("WARNING: " + msg + " Skipping annulus.")
                continue

            x_arr += [x]    # velocity axis
            y_arr += [y]    # deprojected spectrum
            dy_arr += [dy]  # error on the mean

            # Calculate the number of pixels within the mask. This is combined
            # with the independent sampels based on the spectral decorrelation.
            # TODO: Check that the user mask is doing what I think it is...

            annulus_mask = self.get_mask(rbins[ridx], rbins[ridx+1],
                                         exclude_r=False, PA_min=PA_min,
                                         PA_max=PA_max, exclude_PA=exclude_PA,
                                         abs_PA=abs_PA, x0=x0, y0=y0, inc=inc,
                                         PA=PA, z0=z0, psi=psi, z1=z1, phi=phi,
                                         r_cavity=r_cavity, r_taper=r_taper,
                                         q_taper=q_taper, z_func=z_func,
                                         mask_frame=mask_frame,
                                         shadowed=shadowed)
            nbeams += [np.sum(annulus_mask * user_mask * samples)]

        # Return the data to it's saved state (i.e. removing the masked NaNs).

        if mask is not None:
            self.data = saved_data

        # Check that the velocity axes are the same. If not, regrid them to the
        # same velocity axis. TODO: Include a flux-preserving algorithm here.

        x = np.median(x_arr, axis=0)
        if not all(np.isclose(np.std(x_arr, axis=0), 0)):
            from scipy.interpolate import interp1d
            y_arr = [interp1d(xx, yy, bounds_error=False)(x)
                     for xx, yy in zip(x_arr, y_arr)]
            dy_arr = [interp1d(xx, dd, bounds_error=False)(x)
                      for xx, dd in zip(x_arr, dy_arr)]

        # Remove any pesky NaNs.

        spectra = np.where(np.isfinite(y_arr), y_arr, 0.0)
        if spectra.size == 0.0:
            raise ValueError("No finite spectra were returned.")

        # Account for the averaging over N independent samples.

        scatter = np.where(np.isfinite(dy_arr), dy_arr, 0.0)
        scatter /= np.sqrt(nbeams)[:, None]
        if spectra.shape != scatter.shape:
            raise ValueError("spectra.shape != scatter.shape")

        # Combine the averages, weighting each annulus based on its area.

        weights = np.pi * (rbins[1:]**2 - rbins[:-1]**2)
        weights = weights[included.astype('bool')]
        weights += 1e-20 * np.random.randn(weights.size)
        if weights.size != spectra.shape[0]:
            raise ValueError("weights.size != spectra.shape[0]")
        spectrum = np.average(spectra, axis=0, weights=weights)

        # Average over the uncertainties. If requested, override the scatter
        # with an emperically derived value.

        if empirical_uncertainty:
            scatter = imagecube.estimate_uncertainty(spectrum)
            scatter *= np.ones(x.size)
        else:
            scatter = np.sum((weights[:, None] * scatter)**2.0, axis=0)**0.5
            scatter /= np.sum(weights)
        if spectrum.shape != scatter.shape:
            print(scatter.shape, spectrum.shape)
            raise ValueError("spectra.shape != scatter.shape")

        # Convert to K using RJ-approximation.

        if unit == 'k':
            spectrum = self.jybeam_to_Tb_RJ(spectrum)
            scatter = self.jybeam_to_Tb_RJ(scatter)
        if unit[0] == 'm':
            spectrum *= 1e3
            scatter *= 1e3

        return x, spectrum, scatter

    def _independent_samples(self, v0_map, mask=None, velocity_resolution=1.0,
                             plot=False, ignore_spectral_correlation=True):
        """
        Args:
            v0_map (arr): Array of the velocities used to decorrelate the data.
            mask (Optional[arr]): 2D boolean mask.
            velocity_resolution (Optional[float]): The velocity resolution as a
                fraction of the channel spacing.
            plot (Optional[bool]): If ``True``, make a diagnostic plot.
            ignore_spectral_correlation (Optional[bool]): If ``True``, return
                only the effective number of pixels in the central channel.

        Returns:
            something
        """

        # Run through each pixel calculating the independent samples.

        _N = np.zeros(v0_map.shape)
        for i in range(self.nxpix):
            for j in range(self.nypix):

                # Skip empty pixels.

                if not np.isfinite(v0_map[j, i]):
                    continue

                _N[j, i] = self._pixel_independent_samples(
                    i,
                    j,
                    v0_map,
                    mask,
                    velocity_resolution,
                    False,
                    True,
                    )

        if ignore_spectral_correlation:
            effective_samples = _N
        else:
            c_idx = abs(self._velax_offset).argmin()
            effective_samples = _N[:, :, c_idx]
        effective_samples = np.clip(effective_samples, a_min=0.0, a_max=1.0)

        # Calculate the spectral correlation.

        if ignore_spectral_correlation:
            spectral_correlation = 0.0
        else:
            _N_collapsed = np.sum(_N, axis=(0, 1))
            _N_collapsed /= np.trapz(_N_collapsed, dx=self.chan)
            _N_collapsed /= _N_collapsed.max()

            _N_cumsum = np.cumsum(_N_collapsed)
            _N_cumsum /= _N_cumsum[-1]

            sig_a = self._velax_offset[abs(_N_cumsum - 0.16).argmin()]
            sig_b = self._velax_offset[abs(_N_cumsum - 0.84).argmin()]
            spectral_correlation = np.mean([abs(sig_a), abs(sig_b)])

        # Diagnostic plots.

        if plot:
            from matplotlib.ticker import MultipleLocator

            fig, ax = plt.subplots()
            _plt = np.where(effective_samples > 0.0, effective_samples, np.nan)
            im = ax.imshow(_plt, origin='lower', extent=self.extent,
                           cmap='turbo', vmin=0.0, vmax=1.0)
            self.plot_beam(ax=ax, color='w')
            cb = plt.colorbar(im, pad=0.03)
            cb.set_label('Independent Samples per Pixel', rotation=270,
                         labelpad=13)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.set_xlabel('Offset (arcsec)')
            ax.set_ylabel('Offset (arcsec)')

        if plot and not ignore_spectral_correlation:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 3.5))

            ax = axs[0]
            ax.axvline(0.0, ls='--', color='0.7')
            ax.step(self._velax_offset / 1e3, _N_collapsed, where='mid')
            ax.set_xlabel('Velocity Offset (km/s)')
            ax.set_ylabel('Relative Correlation')
            ax.set_xlim(-5e-3 * spectral_correlation,
                        5e-3 * spectral_correlation)

            ax = axs[1]
            ax.axvline(0.0, ls='--', color='0.7')
            ax.axvline(sig_a / 1e3, ls=':', color='0.7')
            ax.axvline(sig_b / 1e3, ls=':', color='0.7')
            ax.plot(self._velax_offset / 1e3, _N_cumsum)
            ax.set_xlabel('Velocity Offset (km/s)')
            ax.set_ylabel('Relative Cumulative Correlation')
            ax.set_xlim(-5e-3 * spectral_correlation,
                        5e-3 * spectral_correlation)
            ax.set_ylim(-0.1, 1.1)
            ax.text(sig_a / 1e3, 1.12, r'$-\sigma$', ha='center', va='bottom')
            ax.text(sig_b / 1e3, 1.12, r'$+\sigma$', ha='center', va='bottom')

        if ignore_spectral_correlation:
            return effective_samples
        else:
            return effective_samples, spectral_correlation

    def _pixel_independent_samples(self, x_idx, y_idx, v0_map, mask=None,
                                   velocity_resolution=1.0, plot=False,
                                   ignore_spectral_correlation=True):
        """
        Returns an array of the independent samples per pixel after spectral
        decorrelation. By default, ignores where the spectral correlation goes,
        but can be used to keep track of how pixels become spectrally
        correlated.

        Args:
            x_idx (int): x-axis index of the pixel.
            y_idx (int): y-axis index of the pixel.
            v0_map (arr): Array of the velocities used to decorrelate the data.
            mask (Optional[arr]): 2D boolean mask.
            velocity_resolution (Optional[float]): The velocity resolution as a
                fraction of the channel spacing.
            plot (Optional[bool]): If ``True``, make a diagnostic plot.
            ignore_spectral_correlation (Optional[bool]): If ``True``, return
                only the effective number of pixels in the central channel.

        Returns:
            independent_samples (arr): The numer of effective independent
                samples for the pixel given the provided velocity deprojection.
        """

        # Define the spatial mask. This is a combination of the the beam shape
        # and any user-defined masks. This should result in a 2D image with
        # values of 1.0 or 0.0 (in the mask or out the mask).

        beam_mask = self._beam_mask(self.xaxis[x_idx], self.yaxis[y_idx])
        user_mask = np.ones(beam_mask.shape) if mask is None else mask
        assert beam_mask.shape == user_mask.shape, 'wrong shape for `mask`'
        combined_mask = np.logical_and(beam_mask, user_mask).astype('float')
        total_pixels = combined_mask.sum()

        # Calculate the velocity correlation then shift the beam by this amount
        # assumining a simple Hanning kernel and linear interpolation.

        assert combined_mask.shape == v0_map.shape
        velocity_offset = v0_map - v0_map[y_idx, x_idx]
        if ignore_spectral_correlation:
            channel_offset = np.array([0.0])
        else:
            channel_offset = self._velax_offset.copy()
        offset = 0.5 * self.chan * velocity_resolution

        # If the pixel of choice is masked then return zero.

        if not user_mask[y_idx, x_idx]:
            return np.zeros(channel_offset.shape)

        kernel = channel_offset[:, None, None] - velocity_offset[None, :, :]
        kernel = np.where(abs(kernel) <= offset, 1.0, 0.0)
        kernel *= combined_mask[None, :, :]
        if ignore_spectral_correlation:
            assert kernel[0].shape == self.data[0].shape
        else:
            assert kernel.shape == self.data.shape

        # Calculate the fraction of the beam in each covariance 'channel'.

        total_correlated_pixels = np.sum(kernel, axis=(1, 2))
        correlated_fraction = total_correlated_pixels / total_pixels
        correlated_fraction = np.where(correlated_fraction == 0.0,
                                       1e16, correlated_fraction)
        independent_samples = self.beams_per_pix / correlated_fraction

        # Make diagnostic plots.

        if plot:
            from matplotlib.ticker import MultipleLocator
            fig, axs = plt.subplots(figsize=(7, 4.5), constrained_layout=True,
                                    gridspec_kw=dict(height_ratios=(0.07, 1)),
                                    ncols=2, nrows=2,)

            ax = axs[1, 0]
            ax.patch.set_facecolor(plt.get_cmap('viridis')(0.0))
            im = ax.imshow(abs(velocity_offset) / self.chan, origin='lower',
                           extent=self.extent, cmap='viridis_r', vmax=1.0)
            ax.set_xlim(self.xaxis[x_idx] + 1.5 * self.bmaj,
                        self.xaxis[x_idx] - 1.5 * self.bmaj)
            ax.set_ylim(self.yaxis[y_idx] - 1.5 * self.bmaj,
                        self.yaxis[y_idx] + 1.5 * self.bmaj)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.set_xlabel('Offset (arcsec)')
            ax.set_ylabel('Offset (arcsec)')

            self.plot_beam(ax=ax, x0=0.5, y0=0.5, color='k')
            if not np.all(user_mask == 1):
                ax.contourf(self.xaxis, self.yaxis, np.where(user_mask, 1, 0),
                            [0.0, 0.5], colors='w', alpha=0.5)
                ax.contour(self.xaxis, self.yaxis, np.where(user_mask, 1, 0),
                           [0.5], linewidths=1.0, colors='w')

            cb = plt.colorbar(im, cax=axs[0, 0], extend='max',
                              orientation='horizontal')
            cb.set_label('Velocity Difference / Channel Spacing', labelpad=13)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

            axs[0, 1].axis('off')

            if ignore_spectral_correlation:
                correlated_pixels = kernel[0]
            else:
                correlated_pixels = kernel[abs(channel_offset).argmin()]

            ax = axs[1, 1]
            ax.imshow(np.where(beam_mask, 0.1, np.nan), vmin=0, vmax=1,
                      origin='lower', extent=self.extent, cmap='binary')
            ax.imshow(np.where(combined_mask, 0.5, np.nan), vmin=0, vmax=1,
                      origin='lower', extent=self.extent, cmap='binary')
            ax.imshow(np.where(correlated_pixels > 0.5, 0.8, np.nan), vmin=0,
                      vmax=1, origin='lower', extent=self.extent, cmap='Reds')

            if mask is None:
                if ignore_spectral_correlation:
                    pcnt = correlated_fraction[0]
                else:
                    pcnt = correlated_fraction[abs(channel_offset).argmin()]
                pcnt *= 1e2
                ax.text(0.05, 0.96,
                        '{:.0f}% of beam correlated'.format(pcnt),
                        color=plt.get_cmap('Reds')(0.8), ha='left', va='top',
                        transform=ax.transAxes)
            else:
                pcnt = 1e2 * total_pixels / beam_mask.sum()
                ax.text(0.05, 0.96,
                        'masked beam is {:.0f}% of total beam'.format(pcnt),
                        color='0.3', ha='left', va='top',
                        transform=ax.transAxes)

                if ignore_spectral_correlation:
                    pcnt = correlated_fraction[0]
                else:
                    pcnt = correlated_fraction[abs(channel_offset).argmin()]
                pcnt *= 1e2
                ax.text(0.05, 0.89,
                        '{:.0f}% of masked beam correlated'.format(pcnt),
                        color=plt.get_cmap('Reds')(0.8), ha='left', va='top',
                        transform=ax.transAxes)

            ax.set_xlim(self.xaxis[x_idx] + 1.0 * self.bmaj,
                        self.xaxis[x_idx] - 1.0 * self.bmaj)
            ax.set_ylim(self.yaxis[y_idx] - 1.0 * self.bmaj,
                        self.yaxis[y_idx] + 1.0 * self.bmaj)
            ax.tick_params(which='both', left=0, bottom=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        return np.where(independent_samples < 1e-10, 0.0, independent_samples)

    def _beam_mask(self, x, y, threshold=0.5, stretch=1.0, response=False):
        """
        Returns a 2D Gaussian mask based on the attached beam centered at
        (x, y) on the sky.

        Args:
            x (flaot): RA offset of the center of the beam.
            y (float): Dec offset of the center of the beam.
            threshold (Optional[float]): Threshold beam power to consider a
                pixel within the beam. Default is 0.5.
            stretch (Optional[float]): Stretch the beam by this factor. A
                `stretch=2` will result in a beam mask that is twice as large
                as the attached beam.
            response (Optional[bool]): If ``True``, return the beam response
                function rather than a boolean mask.

        Returns:
            beammask (arr): 2D boolean array of pixels covered by the beam if
            ``response=False``, the default, otherwise a 2D array of the beam
            response function centered at that location.
        """
        xx, yy = np.meshgrid(self.xaxis - x, self.yaxis - y)
        theta = -np.radians(self.bpa)
        std_x = 0.5 * stretch * self.bmin / np.sqrt(np.log(2.0))
        std_y = 0.5 * stretch * self.bmaj / np.sqrt(np.log(2.0))
        a = np.cos(theta)**2 / std_x**2 + np.sin(theta)**2 / std_y**2
        b = np.sin(2*theta) / std_x**2 - np.sin(2*theta) / std_y**2
        c = np.sin(theta)**2 / std_x**2 + np.cos(theta)**2 / std_y**2
        f = np.exp(-(a*xx**2 + b*xx*yy + c*yy**2))
        return f if response else f >= threshold

    @staticmethod
    def estimate_uncertainty(a, nsigma=3.0, niter=20):
        """
        Estimate the noise by iteratively sigma-clipping. For each iteraction
        the ``a`` array is masked above ``abs(a) > nsigma * std`` and the
        standard deviation, ``std`` calculated. This is repeated until either
        convergence or for ``niter`` iteractions. In some cases, usually with
        low ``nsigma`` values, the ``std`` will approach zero and all ``a``
        values are masked, resulting in an NaN. In this case, the function will
        return the last finite value.

        Args:
            a (array): Array of data from which to estimate the uncertainty.
            nsigma (Optional[float]): Factor of the standard devitation above
                which to mask ``a`` values.
            niter (Optional[int]): Number of iterations to halt after if
                convergence is not reached.

        Returns:
            std (float): Standard deviation of the sigma-clipped data.
        """
        if niter < 1:
            raise ValueError("Must have at least one iteration.")
        nonzero = a != 0.0
        std = np.nanmax(a)
        for _ in range(niter):
            std_new = np.nanstd(a[(abs(a) <= nsigma * std) & nonzero])
            if std_new == std or np.isnan(std_new) or std_new == 0.0:
                return std
            std = std_new
        return std

    def integrated_spectrum(self, r_min=None, r_max=None, dr=None, x0=0.0,
                            y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                            r_cavity=None, r_taper=None, q_taper=None, z1=None,
                            phi=None, z_func=None, mstar=1.0, dist=100.,
                            resample=1, beam_spacing=False, PA_min=None,
                            PA_max=None, exclude_PA=False, abs_PA=False,
                            mask=None, mask_frame='disk',
                            empirical_uncertainty=False,
                            skip_empty_annuli=True,
                            shadowed=False, velocity_resolution=1.0,
                            include_spectral_decorrelation=True):
        """
        Return the integrated spectrum over a given radial range, returning a
        spectrum in units of [Jy].

        The `resample` parameter allows you to resample the
        spectrum at a different velocity spacing (by providing a float
        argument) or averaging of an integer number of channels (by providing
        an integer argument). For example, ``resample=3``, will return a
        velocity axis which has been supersampled such that a channel is three
        times as narrow as the intrinsic width. Instead, ``resample=50.0`` will
        result in a velocity axis with a channel spacing of 50 m/s.

        .. note::

            The third variable returned is the scatter in each velocity bin and
            not the uncertainty on the bin mean as the data is not strictly
            independent due to spectral and spatial correlations in the data.
            If you want to assume uncorrelated data to get a better estimate of
            the uncertainty, set ``assumed_correlated=False``.

        Args:
            r_min (Optional[float]): Inner radius in [arcsec] of the region to
                integrate.
            r_max (Optional[float]): Outer radius in [arcsec] of the region to
                integrate.
            dr (Optional[float]): Width of the annuli to split the
                integrated region into in [arcsec]. Default is quater of the
                beam major axis.
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mstar (Optional[float]): Stellar mass in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            resample(Optional[float/int]): Resampling parameter for the
                deprojected spectrum. An integer specifies an average of that
                many channels, while a float specifies the desired channel
                width. Default is ``resample=1``.
            beam_spacing(Optional[bool]): When extracting the annuli, whether
                to choose spatially independent pixels or not.
            PA_min (Optional[float]): Minimum polar angle to include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            PA_max (Optional[float]): Maximum polar angleto include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            exclude_PA (Optional[bool]): Whether to exclude pixels where
                ``PA_min <= PA_pix <= PA_max``.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask (Optional[ndarray]): Either a 2D or 3D mask to use when
                averaging the spectra. This will be used in addition to any
                geometrically defined mask.
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            assume_correlated (Optional[bool]): Whether to treat the spectra
                which are stacked as correlated, by default this is
                ``True``. If ``False``, the uncertainty will be estimated using
                Poisson statistics, otherwise the uncertainty is just the
                standard deviation of each velocity bin.
            skip_empty_annuli (Optional[bool]): If ``True``, skip any annuli
                which are empty (i.e. their masks have zero pixels in). If
                ``False``, any empty masks will raise a ``ValueError``.

        Returns:
            The velocity axis of the spectrum, ``velax``, in [m/s], the
            integrated spectrum, ``spectrum``, and the variance of the velocity
            bin, ``scatter``. The latter two are in units of [Jy].

        """
        # Check is cube is 2D.

        self._test_2D()

        # Get average spectrum for the desired region.

        x, y, dy = self.average_spectrum(
            r_min=r_min,
            r_max=r_max,
            dr=dr,
            x0=x0,
            y0=y0,
            inc=inc,
            PA=PA,
            z0=z0,
            psi=psi,
            z1=z1,
            phi=phi,
            r_cavity=r_cavity,
            r_taper=r_taper,
            q_taper=q_taper,
            z_func=z_func,
            mstar=mstar,
            dist=dist,
            resample=resample,
            unit='jy/beam',
            beam_spacing=beam_spacing,
            PA_min=PA_min,
            PA_max=PA_max,
            exclude_PA=exclude_PA,
            abs_PA=abs_PA,
            mask=mask,
            mask_frame=mask_frame,
            skip_empty_annuli=skip_empty_annuli,
            empirical_uncertainty=empirical_uncertainty,
            velocity_resolution=velocity_resolution,
            include_spectral_decorrelation=include_spectral_decorrelation,
            shadowed=shadowed,
            )

        # Calculate the area of the integration region.

        if mask is not None:
            if mask.shape != self.data.shape:
                if mask.shape != self.data.shape[1:]:
                    raise ValueError("Unknown mask shape.")
                mask = np.ones(self.data.shape) * mask[None, :, :]
            _mask_A = np.any(mask, axis=0)
        else:
            _mask_A = np.ones(self.data[0].shape)

        _mask_B = self.get_mask(
            r_min,
            r_max,
            exclude_r=False,
            PA_min=PA_min,
            PA_max=PA_max,
            exclude_PA=exclude_PA,
            abs_PA=abs_PA,
            x0=x0,
            y0=y0,
            inc=inc,
            PA=PA,
            z0=z0,
            psi=psi,
            z1=z1,
            phi=phi,
            r_cavity=r_cavity,
            r_taper=r_taper,
            q_taper=q_taper,
            z_func=z_func,
            mask_frame=mask_frame,
            shadowed=shadowed,
            )

        # Rescale from Jy/beam to Jy and return.

        nbeams = self.beams_per_pix * np.sum(_mask_A * _mask_B)
        y *= nbeams
        if empirical_uncertainty:
            dy = imagecube.estimate_uncertainty(y) * np.ones(y.size)
        else:
            dy *= nbeams
        return x, y, dy

    def radial_spectra(self, rvals=None, rbins=None, dr=None, x0=0.0, y0=0.0,
                       inc=0.0, PA=0.0, z0=None, psi=None, r_cavity=None,
                       r_taper=None, q_taper=None, z1=None, phi=None,
                       z_func=None, mstar=1.0, dist=100., resample=1,
                       beam_spacing=False, r_min=None, r_max=None,
                       PA_min=None, PA_max=None, exclude_PA=None, abs_PA=False,
                       mask_frame='disk', mask=None, unit='Jy/beam',
                       shadowed=False, skip_empty_annuli=True,
                       empirical_uncertainty=False):
        """
        Return shifted and stacked spectra, over a given spatial region in the
        disk. The averaged spectra can be rescaled by using the ``unit``
        argument for which the possible units are:

            - ``'mJy/beam'``
            - ``'Jy/beam'``
            - ``'mK'``
            - ``'K'``
            - ``'mJy'``
            - ``'Jy'``

        where ``'mJy'`` or ``'Jy'`` will integrate the emission over the
        defined spatial range assuming that the averaged spectrum is the same
        for all pixels in that region.

        In addition to a spatial region specified by the usual geometrical mask
        properties (``r_min``, ``r_max``, ``PA_min``, ``PA_max``), a
        user-defined can be included, either as a 2D array (such that it is
        constant as a function of velocity), or as a 3D array, for example a
        CLEAN mask, for velocity-specific masks.

        There are two ways to return the uncertainties for the spectra. The
        default are the straight `statistical` uncertainties which are
        propagated through from the ``annulus`` class. Sometimes these can
        appear to give a poor description of the true variance of the data. In
        this case the user can use ``empirical_uncertainty=True`` which will
        use an iterative sigma-clipping approach to estimate the uncertainty
        by the variance in line-free regions of the spectrum.

        .. note::
            Calculation of the empirircal uncertainty is experimental. Use
            the results with caution and if anything looks suspicious, please
            contact me.

        Args:
            rvals (Optional[floats]): Array of bin centres for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            rbins (Optional[floats]): Array of bin edges for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            dr (Optional[float]): Width of the radial bins in [arcsec] to
                use if neither ``rvals`` nor ``rbins`` is set. Default is 1/4
                of the beam major axis.
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mstar (Optional[float]): Stellar mass in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            resample(Optional[float/int]): Resampling parameter for the
                deprojected spectrum. An integer specifies an average of that
                many channels, while a float specifies the desired channel
                width. Default is ``resample=1``.
            beam_spacing(Optional[bool]): When extracting the annuli, whether
                to choose spatially independent pixels or not. Default it not.
            r_min (Optional[float]): Inner radius in [arcsec] of the region to
                integrate. The value used will be greater than or equal to
                ``r_min``.
            r_max (Optional[float]): Outer radius in [arcsec] of the region to
                integrate. The value used will be less than or equal to
                ``r_max``.
            PA_min (Optional[float]): Minimum polar angle to include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            PA_max (Optional[float]): Maximum polar angleto include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            exclude_PA (Optional[bool]): Whether to exclude pixels where
                ``PA_min <= PA_pix <= PA_max``. Default is ``False``.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            mask (Optional[arr]): A 2D or 3D mask to include in addition to the
                geometrical mask. If 3D, must match the shape of the ``data``
                array, while a 2D mask will be interpretted as a velocity
                independent mask and must match ``data[0].shape``.
            unit (Optional[str]): Desired unit of the output spectra.
            shadowed (Optional[bool]): If ``True``, use a slower but more
                accurate deprojection algorithm which will handle shadowed
                pixels due to sub-structure.
            skip_empty_annuli (Optional[bool]): If ``True``, skip any annuli
                which are found to have no finite spectra in them, returning
                ``NaN`` for that annlus. Otherwise raise a ``ValueError``.
            empirical_uncertainty (Optional[bool]): Whether to calculate the
                uncertainty on the spectra empirically (``True``) or using the
                statistical uncertainties.

        Returns:
            Four arrays. An array of the bin centers, ``rvals``, an array of
            the velocity axis, ``velax``, and the averaged spectra, ``spectra``
            and their associated uncertainties, ``scatter``. The latter two
            will have shapes of ``(rvals.size, velax.size)`` and wil be in
            units given by the ``unit`` argument.
        """

        # Check is cube is 2D.

        self._test_2D()

        # Radial sampling with radial boundaries.

        rbins, rvals = self.radial_sampling(rbins=rbins, rvals=rvals, dr=dr)
        r_min = rbins[0] if r_min is None else r_min
        r_max = rbins[-1] if r_max is None else r_max
        rbins = rbins[np.logical_and(rbins >= r_min, rbins <= r_max)]
        rbins, rvals = self.radial_sampling(rbins=rbins)
        if rbins.size < 2:
            raise ValueError("Unable to infer suitable `rbins`.")

        # Goign to copy a lot of the code from `average_spectrum`, but this is
        # because we want to fix the radial gridding and it is unclear if that
        # will be fully respected with `average_spectrum`.

        # Define the velocity at the annulus centers.

        v_kep = self._keplerian(rpnts=rvals, mstar=mstar, dist=dist, inc=inc,
                                z0=z0, psi=psi, z1=z1, phi=phi,
                                r_cavity=r_cavity, r_taper=r_taper,
                                q_taper=q_taper, z_func=z_func)
        v_kep = np.atleast_1d(v_kep)

        # Output unit.

        unit = unit.lower()
        if unit not in ['mjy/beam', 'jy/beam', 'jy', 'mjy', 'mk', 'k']:
            raise ValueError("Unknown `unit`.")
        if resample < 1.0 and self.verbose:
            print('WARNING: `resample < 1`, are you sure you want channels '
                  + 'narrower than 1 m/s?')

        # Pseudo masking - see `average_spectrum` for more.

        if mask is not None:
            if mask.shape != self.data.shape:
                if mask.shape != self.data.shape[1:]:
                    raise ValueError("Unknown mask shape.")
                mask = np.ones(self.data.shape) * mask[None, :, :]
            saved_data = self.data.copy()
            self.data = np.where(mask, self.data, np.nan)
            user_mask = np.any(np.isfinite(self.data), axis=0)
        else:
            user_mask = np.ones(self.data[0].shape)

        # Get the deprojected spectrum for each annulus (or rval). We include
        # an array to describe whether an annulus is included in the average or
        # not in order to rescale the uncertainties.

        x_arr, y_arr, dy_arr, npix_arr = [], [], [], []

        for ridx in range(rvals.size):
            try:
                annulus = self.get_annulus(rbins[ridx], rbins[ridx+1],
                                           x0=x0, y0=y0, inc=inc, PA=PA,
                                           z0=z0, psi=psi, z1=z1, phi=phi,
                                           r_cavity=r_cavity, r_taper=r_taper,
                                           q_taper=q_taper, z_func=z_func,
                                           beam_spacing=beam_spacing,
                                           PA_min=PA_min, PA_max=PA_max,
                                           exclude_PA=exclude_PA,
                                           abs_PA=abs_PA,
                                           mask_frame=mask_frame,
                                           shadowed=shadowed)

                x, y, dy = annulus.deprojected_spectrum(vrot=v_kep[ridx],
                                                        resample=resample,
                                                        scatter=True)

            # Complain if no spectra are found in the annulus.

            except ValueError:
                msg = "No pixels found between {:.2f} ".format(rbins[ridx])
                msg += "and {:.2f} arcsec.".format(rbins[ridx+1])
                if not skip_empty_annuli:
                    raise ValueError(msg)
                elif self.verbose:
                    print("WARNING: " + msg + " Skipping annulus.")
                x, y, dy = [np.nan], [np.nan], [np.nan]

            # Deproject the spectrum currently using a simple bin average.

            x_arr += [x]    # velocity axis
            y_arr += [y]    # deprojected spectrum
            dy_arr += [dy]  # error on the mean

            # Calculate the number of pixels within the mask. This is combined
            # with the `user_mask` which is the 2D projection of the user
            # provided mask which any spectrum contains a finite value.

            npix = self.get_mask(rbins[ridx], rbins[ridx+1], exclude_r=False,
                                 PA_min=PA_min, PA_max=PA_max,
                                 exclude_PA=exclude_PA, abs_PA=abs_PA, x0=x0,
                                 y0=y0, inc=inc, PA=PA, z0=z0, psi=psi, z1=z1,
                                 phi=phi, r_cavity=r_cavity, r_taper=r_taper,
                                 q_taper=q_taper, z_func=z_func,
                                 mask_frame=mask_frame, shadowed=shadowed)
            npix_arr += [np.sum(npix * user_mask)]

        # Sort through all the spectra and resize all the NaN columns where
        # there were not finite pixels found. This is a bit of a circular
        # approach because a priori we do not know the size of the (x, y, dy)
        # lists. We loop through all the finite values to calculate who long
        # these lists are, then we replace all rows in that list with an
        # appropriate length NaN.

        size = np.squeeze([y for y in y_arr if any(np.isfinite(y))])
        if size.size == 0:
            raise ValueError("No finite spectra were returned.")
        size = np.atleast_2d(size).shape[1]

        x_arr_tmp = []
        y_arr_tmp = []
        dy_arr_tmp = []
        for idx in range(len(y_arr)):
            if all(np.isnan(y_arr[idx])):
                x_arr_tmp += [np.ones(size) * np.nan]
                y_arr_tmp += [np.ones(size) * np.nan]
                dy_arr_tmp += [np.ones(size) * np.nan]
            else:
                x_arr_tmp += [x_arr[idx]]
                y_arr_tmp += [y_arr[idx]]
                dy_arr_tmp += [dy_arr[idx]]

        x_arr = np.squeeze(x_arr_tmp)
        y_arr = np.squeeze(y_arr_tmp)
        dy_arr = np.squeeze(dy_arr_tmp)

        # Return the data to it's saved state (i.e. removing the masked NaNs).

        if mask is not None:
            self.data = saved_data

        # Check that the velocity axes are the same. If not, regrid them to the
        # same velocity axis. TODO: Include a flux-preserving algorithm here.

        velax = np.nanmedian(x_arr, axis=0)
        if not all(np.isclose(np.nanstd(x_arr, axis=0), 0)):
            from scipy.interpolate import interp1d
            y_arr = [interp1d(xx, yy, bounds_error=False)(velax)
                     for xx, yy in zip(x_arr, y_arr)]
            dy_arr = [interp1d(xx, dd, bounds_error=False)(velax)
                      for xx, dd in zip(x_arr, dy_arr)]

        # Calculate the number of independent beams in each spectra. This will
        # take into account both the user-provided mask and any PA ranges set
        # by the user.

        N = np.squeeze(npix_arr) * self.dpix**2 / self.beamarea_arcsec

        # Note here we don't want to remove any NaNs such that they're not
        # included in the averaging process.

        spectra = np.squeeze(y_arr)
        scatter = np.squeeze(dy_arr)

        # Replace scatter with empirical uncertainties if requested.
        # Note that this is just a single value for each spectrum and so we
        # need to broadcast it to the same shape as scatter. Otherwise just
        # rescale the uncertainties by sqrt(N) where N is the number of
        # independent beams.

        if empirical_uncertainty:
            scatter = [imagecube.estimate_uncertainty(y) for y in spectra]
            scatter = np.squeeze(scatter)[:, None] * np.ones(spectra.shape)
        else:
            scatter /= np.sqrt(N)[:, None]

        # Apply the rescaling of the pixels. Currently all the spectra are in
        # Jy/beam units. Conversion to K will use the RJ assumption to avoid
        # any non-linearities around zero.

        unit = unit.lower()
        if 'k' in unit:
            spectra = self.jybeam_to_Tb_RJ(spectra)
            scatter = self.jybeam_to_Tb_RJ(scatter)
        elif 'beam' not in unit:
            spectra *= N[:, None]
            scatter *= N[:, None]
        if unit[0] == 'm':
            spectra *= 1e3
            scatter *= 1e3

        # Final check things are how we expect them to be, then return.

        if spectra.shape != (rvals.shape[0], velax.shape[0]):
            raise ValueError("Mismatch between spectra, rvals and velax.")
        if spectra.shape != scatter.shape:
            raise ValueError("Mismatch between spectra and scatter.")
        return rvals, velax, spectra, scatter

    def radial_profile(self, rvals=None, rbins=None, dr=None,
                       unit='Jy/beam m/s', x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                       z0=None, psi=None, r_cavity=None, r_taper=None,
                       q_taper=None, z1=None, phi=None, z_func=None,
                       mstar=0.0, dist=100., resample=1, beam_spacing=False,
                       r_min=None, r_max=None, PA_min=None, PA_max=None,
                       exclude_PA=False, abs_PA=False, mask_frame='disk',
                       mask=None, velo_range=None, assume_correlated=True,
                       shadowed=False):
        """
        Generate a radial profile from shifted and stacked spectra. There are
        different ways to collapse the spectrum into a single value using the
        ``unit`` argument. Possible units for the flux (density) are:

            - ``'mJy/beam'``
            - ``'Jy/beam'``
            - ``'mK'``
            - ``'K'``
            - ``'mJy'``
            - ``'Jy'``

        where ``'/beam'`` is equivalent to ``'/pix'`` if not beam information
        is found in the FITS header. For the velocity we have:

            - ``'m/s'``
            - ``'km/s'``
            - ``''``

        with the empty string resulting in the peak value of the spectrum. For
        example, ``unit='K'`` will return the peak brightness temperature as a
        function of radius, while ``unit='K km/s'`` will return the velocity
        integrated brightness temperature as a function of radius.

        All conversions from [Jy/beam] to [K] are performed using the
        Rayleigh-Jeans approximation. For other units, or to develop more
        sophisticated statistics for the collapsed line profiles, use the
        ``radial_spectra`` function which will return the shifted and stacked
        line profiles.

        .. note:

            The shifting and stacking is only available for 3D cubes. If a 2D
            cube (quadrilateral?) is detected, this function will revert to a
            standard azimuthal average without the capability of transforming
            units.

        Args:
            rvals (Optional[floats]): Array of bin centres for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            rbins (Optional[floats]): Array of bin edges for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            dr (Optional[float]): Width of the radial bins in [arcsec] to
                use if neither ``rvals`` nor ``rbins`` is set. Default is 1/4
                of the beam major axis.
            unit (Optional[str]): Unit for the y-axis of the profile, as
                in the function description.
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mstar (Optional[float]): Stellar mass in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            resample(Optional[float/int]): Resampling parameter for the
                deprojected spectrum. An integer specifies an average of that
                many channels, while a float specifies the desired channel
                width. Default is ``resample=1``.
            beam_spacing(Optional[bool]): When extracting the annuli, whether
                to choose spatially independent pixels or not.
            r_min (Optional[float]): Inner radius in [arcsec] of the region to
                integrate. The value used will be greater than or equal to
                ``r_min``.
            r_max (Optional[float]): Outer radius in [arcsec] of the region to
                integrate. The value used will be less than or equal to
                ``r_max``.
            PA_min (Optional[float]): Minimum polar angle to include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            PA_max (Optional[float]): Maximum polar angleto include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            exclude_PA (Optional[bool]): Whether to exclude pixels where
                ``PA_min <= PA_pix <= PA_max``.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            mask (Optional[array]): A user-specified 2D or 3D array to mask the
                data with prior to shifting and stacking.
            velo_range (Optional[tuple]): A tuple containing the spectral
                range to integrate if required for the provided ``unit``. Can
                either be a string, including units, or as channel integers.
            shadowed (Optional[bool]): If ``True``, use a slower algorithm for
                deprojecting the pixel coordinates into disk-center coordiantes
                which can handle shadowed pixels.

        Returns:
            Arrays of the bin centers in [arcsec], the profile value in the
            requested units and the associated uncertainties.
        """

        # If the data is only 2D, we assume this is either a moment map or a
        # continuum image and so can revert to the 2D approach.

        if self.data.ndim == 2:
            return self._radial_profile_2D(rvals=rvals, rbins=rbins, dr=dr,
                                           x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi,
                                           r_cavity=r_cavity, r_taper=r_taper,
                                           q_taper=q_taper,
                                           z_func=z_func, r_min=r_min,
                                           r_max=r_max, PA_min=PA_min,
                                           PA_max=PA_max, abs_PA=abs_PA,
                                           exclude_PA=exclude_PA,
                                           mask_frame=mask_frame,
                                           assume_correlated=assume_correlated,
                                           shadowed=shadowed)

        # Otherwise we want to grab the shifted and stacked spectra over the
        # given range. We parse the desired unit into a flux component and a
        # spectral component. The latter is only needed for any integration.

        _flux_unit, _velo_unit = imagecube._parse_unit(unit)

        out = self.radial_spectra(rvals=rvals, rbins=rbins, dr=dr,
                                  x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                  z1=z1, phi=phi, r_cavity=r_cavity,
                                  r_taper=r_taper, q_taper=q_taper,
                                  z_func=z_func, mstar=mstar,
                                  dist=dist, resample=resample,
                                  beam_spacing=beam_spacing, r_min=r_min,
                                  r_max=r_max, PA_min=PA_min, PA_max=PA_max,
                                  exclude_PA=exclude_PA, abs_PA=abs_PA,
                                  mask_frame=mask_frame, mask=mask,
                                  unit=_flux_unit, shadowed=shadowed)
        rvals, velax, spectra, scatter = out

        # Cut down the spectra to the correct velocity range. Find the channels
        # closest to the requested values but extend the range to make sure
        # those channels are included.

        va, vb = self._parse_channel(velo_range)
        velax = velax[va:vb+1]
        spectra = spectra[:, va:vb+1]
        scatter = scatter[:, va:vb+1]

        # Collapse the spectra to a radial profile. The returned `spectra`
        # should be already in units of 'Jy/beam', 'Jy' or 'K' with associated
        # uncertainties in `scatter`. Note this for-loop approach was used as
        # `np.trapz` cannot handle NaNs. We could circumvent this by setting
        # all NaNs to 0, however those pixels will still be included in the
        # error propagation.

        if _velo_unit is not None:
            scale = 1e0 if _velo_unit == 'm/s' else 1e-3
            v_tmp = velax * scale
            profile = []
            for s_tmp in spectra:
                mask = np.isfinite(s_tmp)
                profile += [np.trapz(s_tmp[mask], v_tmp[mask])]
            profile = np.squeeze(profile)
        else:
            profile = np.nanmax(spectra, axis=1)

        if profile.size != rvals.size:
            raise ValueError("Mismatch in x and y values.")

        # Basic approximation of uncertainty. If the profile is a maximum, then
        # we just take the uncertainty at that position. Otherwise a simple
        # integration is applied.

        if _velo_unit is not None:
            sigma = scatter * np.diff(velax).mean() * scale
            sigma = np.nansum(sigma**2, axis=1)**0.5
        else:
            sigma = [s[i] for s, i in zip(scatter, np.argmax(spectra, axis=1))]
            sigma = np.squeeze(sigma)

        # Check the shapes of the arrays and return.

        if rvals.shape != profile.shape:
            raise ValueError("Mismatch in rvals and profile shape.")
        if profile.shape != sigma.shape:
            raise ValueError("Mismatch in profile and sigma shape.")
        return rvals, profile, sigma

    def _parse_channel(self, unit):
        """Return the channel numbers based on the provided input."""
        _units = []
        if unit is None:
            return 0, self.velax.size-1
        for u in unit:
            if isinstance(u, int):
                _units += [u]
            elif isinstance(u, float):
                _units += [abs(self.velax - u).argmin()]
            elif isinstance(u, str):
                u = u.lower()
                if 'km/s' in u:
                    _velo = float(u.replace('km/s', '')) * 1e3
                    _units += [abs(self.velax - _velo).argmin()]
                elif 'm/s' in u:
                    _velo = float(u.replace('m/s', ''))
                    _units += [abs(self.velax - _velo).argmin()]
            else:
                raise ValueError("Unrecognised `velo_range` unit.")
        return int(min(_units)), int(max(_units))

    @staticmethod
    def _parse_unit(unit):
        """Return the flux and velocity units for integrating spectra."""
        try:
            flux, velo = unit.split(' ')
            velo = velo.lower()
        except ValueError:
            flux, velo = unit, None
        flux = flux.lower().replace('/pix', '/beam')
        if flux.lower() not in ['mjy', 'jy', 'mk', 'k', 'mjy/beam', 'jy/beam']:
            raise ValueError("Unknown flux unit: {}.".format(flux))
        if velo is not None and velo.lower() not in ['m/s', 'km/s']:
            raise ValueError("Unknown velocity unit: {}".format(velo))
        if velo is not None:
            velo = velo.lower()
        return flux.lower(), velo

    def _test_bunit(self):
        """Test to make sure the attached BUNIT value is valid."""
        try:
            bunit, vunit = self.bunit.lower().split(' ')
        except ValueError:
            bunit, vunit = self.bunit.lower(), ''
        bunit = bunit.replace('/pix', '/beam')
        if bunit not in ['jy/beam', 'mjy/beam', 'k', 'mk', 'm/s', 'km/s']:
            raise ValueError("Unknown 'BUNIT' value " +
                             "{}. ".format(self.header['bunit']) +
                             "Please provide `bunit`.")
        if vunit not in ['', 'km/s', 'm/s']:
            raise ValueError("Unknown 'BUNIT' value " +
                             "{}. ".format(self.header['bunit']) +
                             "Please provide `bunit`.")
        return bunit, vunit

    def _test_2D(self):
        """Check to see if the cube is 3D and can use shifting functions."""
        if self.data.ndim == 2:
            raise ValueError("Cube is only 2D. Shifting not available.")

    def _radial_profile_2D(self, rvals=None, rbins=None, dr=None, x0=0.0,
                           y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                           r_cavity=None, r_taper=None, q_taper=None, z1=None,
                           phi=None, z_func=None, r_min=None, r_max=None,
                           PA_min=None, PA_max=None, exclude_PA=False,
                           abs_PA=False, mask_frame='disk',
                           assume_correlated=False, percentiles=False,
                           shadowed=False):
        """
        Returns the radial profile if `self.data.ndim == 2`, i.e., if shifting
        cannot be performed. Uses all the same parameters, but does not do any
        of the shifting and ignores the units. If this is called directly then
        there are several specific options.

        Args:
            percentiles (Optional[bool]): Use the 16th to 84th percentil of the
                bin distribution to estimate uncertainties rather than the
                standard deviation.

        Returns:
            Three 1D arrays with ``x``, ``y`` and ``dy`` for plotting.
        """

        # Warning message.
        if self.verbose:
            print("WARNING: Attached data is not 3D, so shifting cannot be " +
                  "applied.\n\t Reverting to standard azimuthal averaging; " +
                  "will ignore `unit` argument.")

        # Calculate the mask.
        rbins, rvals = self.radial_sampling(rbins=rbins, rvals=rvals, dr=dr)
        mask = self.get_mask(r_min=rbins[0], r_max=rbins[-1], PA_min=PA_min,
                             PA_max=PA_max, exclude_PA=exclude_PA,
                             abs_PA=abs_PA, mask_frame=mask_frame, x0=x0,
                             y0=y0, inc=inc, PA=PA, z0=z0, psi=psi, z1=z1,
                             phi=phi, r_cavity=r_cavity, r_taper=r_taper,
                             q_taper=q_taper, z_func=z_func, shadowed=shadowed)
        if mask.shape != self.data.shape:
            raise ValueError("Mismatch in mask and data shape.")
        mask = mask.flatten()

        # Deprojection of the disk coordinates.
        rpnts = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                 z1=z1, phi=phi, r_cavity=r_cavity,
                                 r_taper=r_taper, q_taper=q_taper,
                                 z_func=z_func, shadowed=shadowed)[0]
        if rpnts.shape != self.data.shape:
            raise ValueError("Mismatch in rvals and data shape.")
        rpnts = rpnts.flatten()

        # Account for the number of independent beams.
        if assume_correlated:
            nbeams = 2.0 * np.pi * rvals / self.bmaj
        else:
            nbeams = 1.0

        # Radial binning.
        rpnts = rpnts[mask]
        toavg = self.data.flatten()[mask]
        ridxs = np.digitize(rpnts, rbins)

        # Averaging.
        if percentiles:
            rstat = np.array([np.percentile(toavg[ridxs == r], [16, 50, 84])
                              for r in range(1, rbins.size)]).T
            ravgs = rstat[1]
            rstds = np.array([rstat[1] - rstat[0], rstat[2] - rstat[1]])
            rstds /= np.sqrt(nbeams)[None, :]
        else:
            ravgs = np.array([np.mean(toavg[ridxs == r])
                              for r in range(1, rbins.size)])
            rstds = np.array([np.std(toavg[ridxs == r])
                              for r in range(1, rbins.size)])
            rstds /= np.sqrt(nbeams)

        return rvals, ravgs, rstds

    def shifted_cube(self, inc, PA, x0=0.0, y0=0.0, z0=None, psi=None,
                     r_cavity=None, r_taper=None, q_taper=None, z1=None,
                     phi=None, z_func=None, mstar=None, dist=None, vmod=None,
                     r_min=None, r_max=None, fill_val=np.nan, save=False,
                     shadowed=False):
        """
        Apply the velocity shift to each pixel and return the cube, or save as
        as new FITS file. This would be useful if you want to create moment
        maps of the data where you want to integrate over a specific velocity
        range without having to worry about the Keplerian rotation in the disk.

        Args:
            inc (float): Inclination of the disk in [degrees].
            PA (float): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[callable]): A function which returns the emission
                height in [arcsec] for a radius given in [arcsec].
            mstar (Optional[float]): Stellar mass in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            v0 (Optional[callable]): A function which returns the projected
                line of sight velocity in [m/s] for a radius given in [m/s].
            r_min (Optional[float]): The inner radius in [arcsec] to shift.
            r_max (Optional[float]): The outer radius in [arcsec] to shift.

        Returns:
            The shifted data cube.
        """

        # Radial positions.
        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                           z0=z0, psi=psi, z1=z1, phi=phi,
                                           r_cavity=r_cavity, r_taper=r_taper,
                                           q_taper=q_taper, z_func=z_func,
                                           shadowed=shadowed)
        r_min = 0.0 if r_min is None else r_min
        r_max = rvals.max() if r_max is None else r_max
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)

        # Projected velocity.
        if (mstar is not None) and (vmod is not None):
            raise ValueError("Only specify `mstar` and `dist` or `vmod`.")
        if mstar is not None:
            if dist is None:
                raise ValueError("Must specify `dist` with `mstar`.")
            vmod = self._keplerian(rpnts=rvals, mstar=mstar, dist=dist,
                                   inc=inc, z0=z0, psi=psi, z1=z1, phi=phi,
                                   r_cavity=r_cavity, r_taper=r_taper,
                                   q_taper=q_taper, z_func=z_func)
            vmod *= np.cos(tvals)
        elif vmod is None:
            raise ValueError("Must specify `mstar` and `dist` or `vmod`.")
        if vmod.shape != rvals.shape:
            raise ValueError("Velocity map incorrect shape.")

        # Shift each pixel.
        from scipy.interpolate import interp1d
        shifted = np.empty(self.data.shape)
        for y in range(self.nypix):
            for x in range(self.nxpix):
                if mask[y, x]:
                    shifted[:, y, x] = interp1d(self.velax - vmod[y, x],
                                                self.data[:, y, x],
                                                bounds_error=False)(self.velax)
        assert shifted.shape == self.data.shape, "Wrong shape of shifted cube."

        if save:
            if isinstance(save, str):
                output = save
            else:
                output = self.path.replace('.fits', '_shifted.fits')
            fits.writeto(output, shifted.astype('float32'), self.header)

        return shifted

    def keplerian(self, inc, PA, mstar, dist, x0=0.0, y0=0.0, vlsr=0.0,
                  z0=None, psi=None, r_cavity=None, r_taper=None, q_taper=None,
                  z1=None, phi=None, z_func=None, r_min=None, r_max=None,
                  cylindrical=False, shadowed=False):
        """
        Projected Keplerian velocity profile in [m/s]. For positions outside
        ``r_min`` and ``r_max`` the values will be set to NaN.

        Args:
            inc (float): Inclination of the disk in [degrees].
            PA (float): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            mstar (float): Stellar mass in [Msun].
            dist (float): Distance to the source in [pc].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            r_cavity (Optional[float]): Edge of the inner cavity for the
                emission surface in [arcsec].
            r_taper (Optional[float]): Characteristic radius in [arcsec] of the
                exponential taper to the emission surface.
            q_taper (Optional[float]): Exponent of the exponential taper of the
                emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[callable]): A function which returns the emission
                height in [arcsec] for a radius given in [arcsec].
            r_min (Optional[float]): The inner radius in [arcsec] to model.
            r_max (Optional[float]): The outer radius in [arcsec] to model.
            cylindrical (Optional[bool]): If ``True``, force cylindrical
                rotation, i.e. ignore the height in calculating the velocity.
            shadowed (Optional[bool]): If ``True``, use a slower algorithm for
                deprojecting the pixel coordinates into disk-center coordiantes
                which can handle shadowed pixels.
        """
        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                               z0=z0, psi=psi, z1=z1, phi=phi,
                                               r_cavity=r_cavity,
                                               r_taper=r_taper,
                                               q_taper=q_taper,
                                               z_func=z_func,
                                               shadowed=shadowed)
        r_min = 0.0 if r_min is None else r_min
        r_max = np.nanmax(rvals) if r_max is None else r_max
        assert r_min < r_max, 'r_min >= r_max'
        r_m = rvals * dist * sc.au
        z_m = 0.0 if cylindrical else zvals * dist * sc.au
        vkep = sc.G * mstar * 1.988e30 * np.power(r_m, 2.0)
        vkep = np.sqrt(vkep / np.power(np.hypot(r_m, z_m), 3.0))
        vkep = vkep * np.sin(abs(np.radians(inc))) * np.cos(tvals) + vlsr
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        return np.where(mask, vkep, np.nan)

    def find_center(self, dx=None, dy=None, Nx=None, Ny=None, mask=None,
                    v_min=None, v_max=None, spectrum='avg', SNR='peak',
                    normalize=True, **kwargs):
        """
        Find the source center (assuming the disk is azimuthally symmetric) by
        calculating the SNR of the averaged spectrum by varying the source
        center, ``(x0, y0)``.

        Args:
            dx (Optional[float]): Maximum offset to consider in the `x`
                direction in [arcsec]. Default is one beam FWHM.
            dy (Optional[float]): Maximum offset to consider in the `y`
                direction in [arcsec]. Default is one beam FWHM.
            Nx (Optional[int]): Number of samples to take along the `x`
                direction. Default results in roughtly pixel spacing.
            Ny (Optional[int]): Number of samples to take along the `y`
                direction. Default results in roughtly pixel spacing.
            mask (Optional[array]): Boolean mask of channels to use when
                calculating the integrated flux or RMS noise.
            v_min (Optional[float]): Minimum velocity in [m/s] to consider if
                an explicit mask is not provided.
            v_max (Optional[float]): Maximum velocity in [m/s] to consider if
                an explicit mask is not provided.
            spectrum (Optional[str]): Type of spectrum to consider, either the
                integrated spectrum with ``'int'``, or the average spectrum
                with ``'avg'``.
            SNR (Optional[str]): Type of signal-to-noise definition to use.
                Either ``'int'`` to use the integrated flux density as the
                signal, or ``'peak'`` to use the maximum value.
            normalize (Optional[bool]): Whether to normalize the SNR map
                relative to the SNR at ``(x0, y0) = (0, 0)``.

        Returns:
            The axes of the grid search, ``x0s`` and ``y0s``, as well as the
            2D array of SNR values, ``SNR``.
        """

        # Verify the SNR method.
        spectrum = spectrum.lower()
        if spectrum == 'avg':
            spectrum_func = self.average_spectrum
        elif spectrum == 'int':
            spectrum_func = self.integrated_spectrum
        else:
            raise ValueError("`spectrum` must be 'avg' or 'int'.")
        SNR = SNR.lower()
        if SNR == 'int':
            SNR_func = self._integrated_SNR
        elif SNR == 'peak':
            SNR_func = self._peak_SNR
        else:
            raise ValueError("`SNR` must be 'int' or 'peak'.")

        # Define the axes to search.
        dx = self.bmaj if dx is None else dx
        dy = self.bmaj if dy is None else dy
        Nx = int(2. * dx / self.dpix) if Nx is None else Nx
        Ny = int(2. * dy / self.dpix) if Ny is None else Ny
        x0s = np.linspace(-dx, dx, Nx)
        y0s = np.linspace(-dy, dy, Ny)
        SNR = np.zeros((Ny, Nx))

        # Empty the kwargs.
        _ = kwargs.pop('x0', np.nan)
        _ = kwargs.pop('y0', np.nan)

        # Define the mask.
        x, _, _ = spectrum_func(**kwargs)
        v_min = np.percentile(x, [40.])[0] if v_min is None else v_min
        v_max = np.percentile(x, [60.])[0] if v_max is None else v_max
        mask = np.logical_and(x >= v_min, x <= v_max) if mask is None else mask
        if mask.size != x.size:
            raise ValueError("`mask` is not the same size as expected v-axis.")

        # Loop through the possible centers.
        for i, x0 in enumerate(x0s):
            for j, y0 in enumerate(y0s):
                try:
                    x, y, dy = spectrum_func(x0=x0, y0=y0, **kwargs)
                    SNR[j, i] = SNR_func(x, y, dy, mask)
                except ValueError:
                    SNR[j, i] = np.nan

        # Determine the optimum position.
        self._plot_center(x0s, y0s, SNR, normalize)
        return x0s, y0s, SNR

    def _integrated_SNR(self, x, y, dy, mask):
        """SNR based on the integrated spectrum."""
        y_tmp = np.where(np.logical_and(mask, np.isfinite(y)), y, 0.0)
        return np.trapz(y_tmp, x=x) / np.nanmean(dy[~mask])

    def _peak_SNR(self, x, y, dy, mask):
        """SNR based on the peak of the spectrum."""
        y_tmp = np.where(np.logical_and(mask, np.isfinite(y)), y, 0.0)
        return np.max(y_tmp) / np.nanmean(dy[~mask])

    def _keplerian(self, rpnts, mstar=1.0, dist=100., inc=90.0, z0=0.0,
                   psi=1.0, z1=0.0, phi=1.0, r_cavity=0.0, r_taper=np.inf,
                   q_taper=1.0, z_func=None):
        """
        Return a Keplerian rotation profile [m/s] at rpnts [arcsec].

        Args:
            rpnts (ndarray/float): Radial locations in [arcsec] to calculate
                the Keplerian rotation curve at.
            mstar (float): Mass of the central star in [Msun].
            dist (float): Distance to the source in [pc].
            inc (Optional[float]): Inclination of the source in [deg]. If not
                provided, will return the unprojected value.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.

        Returns:
            vkep (ndarray/float): Keplerian rotation curve [m/s] at the
                specified radial locations.
        """

        # Set the defaults.
        z0 = 0.0 if z0 is None else z0
        psi = 1.0 if psi is None else psi
        z1 = 0.0 if z1 is None else z1
        phi = 1.0 if phi is None else phi
        r_taper = np.inf if r_taper is None else r_taper
        q_taper = 1.0 if q_taper is None else q_taper
        r_cavity = 0.0 if r_cavity is None else r_cavity
        rpnts = np.squeeze(rpnts)
        if z_func is None:
            def z_func(r_in):
                r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
                z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
                return z * np.exp(-np.power(r / r_taper, q_taper))
        zvals = z_func(rpnts)
        r_m, z_m = rpnts * dist * sc.au, zvals * dist * sc.au
        vkep = sc.G * mstar * 1.988e30 * np.power(r_m, 2.0)
        vkep = np.sqrt(vkep / np.power(np.hypot(r_m, z_m), 3.0))
        return vkep * np.sin(abs(np.radians(inc)))

    # -- Inferring Velocity Profiles -- #

    def get_vlos(self, rvals=None, rbins=None, r_min=None, r_max=None,
                 PA_min=None, PA_max=None, exclude_PA=False, abs_PA=False,
                 x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                 r_cavity=None, r_taper=None, q_taper=None, z1=None, phi=None,
                 z_func=None, mstar=None, dist=None, mask=None,
                 mask_frame='disk', beam_spacing=True, fit_vrad=True,
                 annulus_kwargs=None, get_vlos_kwargs=None, shadowed=False):
        """
        Infer the rotational and radial velocity profiles from the data
        following the approach described in `Teague et al. (2018)`_. The cube
        will be split into annuli, with the `get_vlos()` function from
        `annulus` being used to infer the rotational (and radial) velocities.

        .. _Teague et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...868..113T/abstract

        Args:
            rvals (Optional[floats]): Array of bin centres for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            rbins (Optional[floats]): Array of bin edges for the profile in
                [arcsec]. You need only specify one of ``rvals`` and ``rbins``.
            r_min (float): Minimum midplane radius of the annuli in [arcsec].
            r_max (float): Maximum midplane radius of the annuli in [arcsec].
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            z_func (Optional[function]): A function which provides z(r). Note
                that no checking will occur to make sure this is a valid
                function.
            mstar (Optional[float]): Stellar mass of the central star in [Msun]
                to calculate the  starting positions for the MCMC. If
                specified, must also provide ``dist``.
            dist (Optional[float]): Source distance in [pc] to use to calculate
                the starting positions for the MCMC. If specified, must also
                provide ``mstar``.
            mask (Optional[ndarray]): A 2D array mask which matches the shape
                of the data.
            mask_frame (Optional[str]): Coordinate frame for the mask. Either
                ``'disk'`` or ``'sky'``. If ``disk`` coordinates are used then
                the inclination and position angle of the mask are set to zero.
            beam_spacing (Optional[bool/float]): If True, randomly sample the
                annulus such that each pixel is at least a beam FWHM apart. A
                number can also be used in place of a boolean which will
                describe the number of beam FWHMs to separate each sample by.
            fit_vrad (Optional[bool]): Whether to include radial velocities in
                the optimization.
            annulus_kwargs (Optional[dict]): Kwargs to pass to ``get_annulus``.
            get_vlos_kwargs (Optional[dict]): Kwargs to pass to
                ``annulus.get_vlos``.

        Returns:
            The radial sampling of the annuli, ``rpnts``, and a list of the
            values returned from ``annulus.get_vlos``.
        """

        # Check the input variables.
        if mstar is not None and dist is None:
            raise ValueError("Must specify both `mstar` and `dist`.")
        if dist is not None and mstar is None:
            raise ValueError("Must specify both `mstar` and `dist`.")

        # Get the radial sampling.
        rbins, rpnts = self.radial_sampling(rbins=rbins, rvals=rvals)
        r_min = rbins[0] if r_min is None else r_min
        r_max = rbins[-1] if r_max is None else r_max
        r_mask = np.logical_and(rbins >= r_min, rbins <= r_max)
        rbins, rpnts = self.radial_sampling(rbins=rbins[r_mask])

        # Set the defauls for the get_vlos() function.
        get_vlos_kwargs = {} if get_vlos_kwargs is None else get_vlos_kwargs
        get_vlos_kwargs['plots'] = get_vlos_kwargs.pop('plots', 'none')

        returns = []
        for ridx in range(rpnts.size):
            annulus = self.get_annulus(r_min=rbins[ridx], r_max=rbins[ridx+1],
                                       PA_min=PA_min, PA_max=PA_max,
                                       exclude_PA=exclude_PA, abs_PA=abs_PA,
                                       x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                       psi=psi, z1=z1, phi=phi,
                                       r_cavity=r_cavity, r_taper=r_taper,
                                       q_taper=q_taper, z_func=z_func,
                                       mask=mask, beam_spacing=beam_spacing,
                                       shadowed=shadowed)

            # Starting positions if a velocity profile is given.
            if mstar is not None:
                vrot = self._keplerian(rpnts=rpnts[ridx], mstar=mstar,
                                       dist=dist, inc=inc, z0=z0, psi=psi,
                                       z1=z1, phi=phi, r_cavity=r_cavity,
                                       r_taper=r_taper, q_taper=q_taper,
                                       z_func=z_func)
                vrad = 0.0
                rms = self.estimate_RMS()
                ln_sig = np.log(np.std(annulus.spectra))
                ln_rho = np.log(150.)
                if fit_vrad:
                    get_vlos_kwargs['p0'] = [vrot, vrad, rms, ln_sig, ln_rho]
                else:
                    get_vlos_kwargs['p0'] = [vrot, rms, ln_sig, ln_rho]
            else:
                get_vlos_kwargs['p0'] = None
                get_vlos_kwargs['fit_vrad'] = fit_vrad
            returns += [annulus.get_vlos(**get_vlos_kwargs)]
        return rpnts, np.squeeze(returns)

    # -- Annulus Masking Functions -- #

    def get_annulus(self, r_min, r_max, PA_min=None, PA_max=None,
                    exclude_PA=False, abs_PA=False, x0=0.0, y0=0.0, inc=0.0,
                    PA=0.0, z0=None, psi=None, r_cavity=None, r_taper=None,
                    q_taper=None, z1=None, phi=None, z_func=None,
                    mask=None, mask_frame='disk', beam_spacing=True,
                    annulus_kwargs=None, shadowed=False):
        """
        Return an annulus (or section of), of spectra and their polar angles.
        Can select spatially independent pixels within the annulus, however as
        this is random, each draw will be different.

        Args:
            r_min (float): Minimum midplane radius of the annulus in [arcsec].
            r_max (float): Maximum midplane radius of the annulus in [arcsec].
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            z_func (Optional[function]): A function which provides z(r). Note
                that no checking will occur to make sure this is a valid
                function.
            mask (Optional[ndarray]): A 2D array mask which matches the shape
                of the data.
            mask_frame (Optional[str]): Coordinate frame for the mask. Either
                ``'disk'`` or ``'sky'``. If ``disk`` coordinates are used then
                the inclination and position angle of the mask are set to zero.
            beam_spacing (Optional[bool/float]): If True, randomly sample the
                annulus such that each pixel is at least a beam FWHM apart. A
                number can also be used in place of a boolean which will
                describe the number of beam FWHMs to separate each sample by.

        Returns:
            If ``annulus=True``, will return an ``eddy.annulus`` instance,
            otherwise will be an array containing the polar angles of each
            spectrum in [degrees] and the array of spectra, ordered in
            increasing polar angle.
        """
        # Check is cube is 2D.
        self._test_2D()

        # Generate the mask and check it is the correct shape.
        if mask is None:
            mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=False,
                                 PA_min=PA_min, PA_max=PA_max,
                                 exclude_PA=exclude_PA, abs_PA=abs_PA, x0=x0,
                                 y0=y0, inc=inc, PA=PA, z0=z0, psi=psi, z1=z1,
                                 phi=phi, r_cavity=r_cavity, r_taper=r_taper,
                                 q_taper=q_taper, z_func=z_func,
                                 mask_frame=mask_frame, shadowed=shadowed)
        if mask.shape != self.data[0].shape:
            raise ValueError("`mask` is incorrect shape.")
        mask = mask.flatten()

        # Flatten the data and get deprojected pixel coordinates.
        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        rvals, tvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, z1=z1, phi=phi,
                                        r_cavity=r_cavity, r_taper=r_taper,
                                        q_taper=q_taper, z_func=z_func,
                                        shadowed=shadowed)[:2]
        rvals, tvals = rvals.flatten(), tvals.flatten()
        dvals, rvals, tvals = dvals[:, mask].T, rvals[mask], tvals[mask]

        # Apply the beam sampling.
        if beam_spacing:

            # Order the data in increase position angle.
            idxs = np.argsort(tvals)
            dvals, tvals = dvals[idxs], tvals[idxs]

            # Calculate the sampling rate.
            sampling = float(beam_spacing) * self.bmaj
            sampling /= np.mean(rvals) * np.median(np.diff(tvals))
            sampling = np.floor(sampling).astype('int')

            # If the sampling rate is above 1, start at a random location in
            # the array and sample at this rate, otherwise don't sample. This
            # happens at small radii, for example.
            if sampling > 1:
                start = np.random.randint(0, tvals.size)
                tvals = np.concatenate([tvals[start:], tvals[:start]])
                dvals = np.vstack([dvals[start:], dvals[:start]])
                tvals, dvals = tvals[::sampling], dvals[::sampling]

        # Return the annulus.
        annulus_kwargs = {} if annulus_kwargs is None else annulus_kwargs
        return annulus(spectra=dvals, theta=tvals, velax=self.velax,
                       **annulus_kwargs)

    def get_mask(self, r_min=None, r_max=None, exclude_r=False, PA_min=None,
                 PA_max=None, exclude_PA=False, abs_PA=False,
                 mask_frame='disk', x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None,
                 psi=None, r_cavity=None, r_taper=None, q_taper=None, z1=None,
                 phi=None, z_func=None, shadowed=False):
        """
        Returns a 2D mask for pixels in the given region. The mask can be
        specified in either disk-centric coordinates, ``mask_frame='disk'``,
        or on the sky, ``mask_frame='sky'``. If sky-frame coordinates are
        requested, the geometrical parameters (``inc``, ``PA``, ``z0``, etc.)
        are ignored, however the source offsets, ``x0``, ``y0``, are still
        considered.

        Args:
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.

        Returns:
            A 2D array mask matching the shape of a channel.
        """

        # Check the requested frame.
        mask_frame = mask_frame.lower()
        if mask_frame not in ['disk', 'sky']:
            raise ValueError("mask_frame must be 'disk' or 'sky'.")

        # Remove coordinates if in sky-frame.
        if mask_frame == 'sky':
            r_cavity = 0.0
            inc, PA = 0.0, 0.0
            z0, psi = 0.0, 1.0
            z1, phi = 0.0, 1.0
            z_func = None

        # Calculate pixel coordaintes.
        rvals, tvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, z1=z1, phi=phi,
                                        r_cavity=r_cavity, r_taper=r_taper,
                                        q_taper=q_taper, z_func=z_func,
                                        frame='cylindrical',
                                        shadowed=shadowed)[:2]
        tvals = abs(tvals) if abs_PA else tvals

        # Radial mask.
        r_min = np.nanmin(rvals) if r_min is None else r_min
        r_max = np.nanmax(rvals) if r_max is None else r_max
        if r_min >= r_max:
            raise ValueError("`r_min` must be smaller than `r_max`.")
        r_mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        r_mask = ~r_mask if exclude_r else r_mask

        # Azimuthal mask.
        PA_min = np.nanmin(tvals) if PA_min is None else np.radians(PA_min)
        PA_max = np.nanmax(tvals) if PA_max is None else np.radians(PA_max)
        if PA_min >= PA_max:
            raise ValueError("`PA_min` must be smaller than `PA_max`.")
        PA_mask = np.logical_and(tvals >= PA_min, tvals <= PA_max)
        PA_mask = ~PA_mask if exclude_PA else PA_mask

        # Combine and return.
        mask = r_mask * PA_mask
        if np.sum(mask) == 0:
            raise ValueError("There are zero pixels in the mask.")
        return mask

    def radial_sampling(self, rbins=None, rvals=None, dr=None):
        """
        Return bins and bin center values. If the desired bin edges are known,
        will return the bin edges and vice versa. If neither are known will
        return default binning with the desired spacing.

        Args:
            rbins (Optional[list]): List of bin edges.
            rvals (Optional[list]): List of bin centers.
            dr (Optional[float]): Spacing of bin centers in [arcsec]. Defaults
                to a quarter of the beam major axis.

        Returns:
            rbins (list): List of bin edges.
            rpnts (list): List of bin centres.
        """
        if rbins is not None and rvals is not None:
            raise ValueError("Specify only 'rbins' or 'rvals', not both.")
        if rvals is not None:
            try:
                dr = np.diff(rvals)[0] * 0.5
            except IndexError:
                if self.dpix == self.bmaj:
                    dr = 2.0 * self.dpix
                else:
                    dr = self.bmaj / 4.0
            rbins = np.linspace(rvals[0] - dr, rvals[-1] + dr, len(rvals) + 1)
        if rbins is not None:
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            if dr is None:
                if self.dpix == self.bmaj:
                    dr = 2.0 * self.dpix
                else:
                    dr = self.bmaj / 4.0
            rbins = np.arange(0, self.xaxis.max(), dr)
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        return rbins, rvals

    def background_residual(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None,
                            psi=None, r_cavity=None, r_taper=None,
                            q_taper=None, z1=None, phi=None, z_func=None,
                            r_min=None, r_max=None, PA_min=None, PA_max=None,
                            exclude_PA=False, abs_PA=False, mask_frame='disk',
                            interp1d_kw=None, background_only=False,
                            shadowed=False):
        """
        Return the residual from an azimuthally avearged background. This is
        most appropriate for exploring azimuthally asymmetric emission in
        either the zeroth moment (integrated intensity) or the peak brightness
        temperature maps. As such, this function only works for 2D data.

        The coordinates provided will be used to both build the azimuthally
        averaged profile (using the ``radial_profile`` function) and then
        project this onto the sky-plane. Any masking parameters used here will
        only be used when creating the azimuthally spectrum, but the residual
        with cover the entire data.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            z_func (Optional[function]): A function which provides z(r). Note
                that no checking will occur to make sure this is a valid
                function.
            r_min (Optional[float]): Inner radius in [arcsec] of the region to
                integrate. The value used will be greater than or equal to
                ``r_min``.
            r_max (Optional[float]): Outer radius in [arcsec] of the region to
                integrate. The value used will be less than or equal to
                ``r_max``.
            PA_min (Optional[float]): Minimum polar angle to include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            PA_max (Optional[float]): Maximum polar angleto include in the
                annulus in [degrees]. Note that the polar angle is measured in
                the disk-frame, unlike the position angle which is measured in
                the sky-plane.
            exclude_PA (Optional[bool]): Whether to exclude pixels where
                ``PA_min <= PA_pix <= PA_max``.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            interp1d_kw (Optional[dict]): Kwargs to pass to
                ``scipy.interpolate.interp1d``.
            background_only (Optional[bool]): If True, return only the
                azimuthally averaged background rather than the residual.
                Default is ``False``.

        Returns:
            residual (array): Residual between the data and the azimuthally
                averaged background. If ``background_only = True`` then this is
                just the azimuthally averaged background.
        """
        # Check if the attached data is 2D.
        if self.data.ndim != 2:
            raise ValueError("Cannot azimuthally average a 3D cube.")

        # Make the azimuthal profile.
        x, y, _ = self.radial_profile(x0=x0, y0=y0, inc=inc, PA=PA,
                                      z0=z0, phi=phi, z1=z1, psi=psi,
                                      r_cavity=r_cavity, r_taper=r_taper,
                                      q_taper=q_taper,
                                      z_func=z_func, r_min=r_min, r_max=r_max,
                                      PA_min=PA_min, PA_max=PA_max,
                                      exclude_PA=exclude_PA, abs_PA=abs_PA,
                                      mask_frame=mask_frame, shadowed=shadowed)

        # Build the interpolation function.
        from scipy.interpolate import interp1d
        interp1d_kw = interp1d_kw if interp1d_kw is not None else {}
        interp1d_kw['bounds_error'] = interp1d_kw.pop('bounds_error', False)
        interp1d_kw['fill_value'] = interp1d_kw.pop('fill_value', np.nan)
        avg = interp1d(x, y, **interp1d_kw)

        # Return the residual (or background if requested).
        background = avg(self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                          z0=z0, phi=phi, z1=z1, psi=psi,
                                          r_cavity=r_cavity, r_taper=r_taper,
                                          q_taper=q_taper,
                                          z_func=z_func, shadowed=shadowed)[0])
        return background if background_only else self.data - background

    # -- Deprojection Functions -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                    r_cavity=None, r_taper=None, q_taper=None, z1=None,
                    phi=None, z_func=None, force_positive_surface=False,
                    force_negative_surface=False, frame='cylindrical',
                    shadowed=False, extend=2.0, oversample=2.0,
                    griddata_kw=None):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is most simply described as a
        power law profile,

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi}

        where ``z0`` and ``psi`` can be provided by the user. With the increase
        in spatial resolution afforded by interferometers such as ALMA there
        are a couple of modifications that can be used to provide a better
        match to the data.

        An inner cavity can be included with the ``r_cavity`` argument which
        makes the transformation:

        .. math::

            \tilde{r} = {\rm max}(0, r - r_{\rm cavity})

        Note that the inclusion of a cavity will mean that other parameters,
        such as ``z0``, would need to change as the radial axis has effectively
        been shifted.

        To account for the drop in emission surface in the outer disk where the
        gas surface density decreases there are two descriptions. The preferred
        way is to include an exponential taper to the power law profile,

        .. math::

            z_{\rm tapered}(r) = z(r) \times \exp\left( -\left[
            \frac{r}{r_{\rm taper}} \right]^{q_{\rm taper}} \right)

        where both ``r_taper`` and ``q_taper`` values must be set.
        Alternatively you can use a second power law profile,

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi} +
            z_1 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\varphi}

        again where both ``z1`` and ``phi`` must be specified. While it is
        possible to combine the double power law profile with the exponential
        taper, this is not advised due to the large degeneracy between some of
        the arguments.

        It is also possible to override this parameterization and directly
        provide a user-defined ``z_func``. This allow for highly complex
        surfaces to be included. If this is provided, the other height
        parameters are ignored.

        In some cases, the projection of the emission surface can lead to
        regions of the disk being 'shadowed' by itself along the line of sight
        to the observer. The default method for calculating the disk
        coordinates does not take this into account in preference to speed. If
        you work with some emission surface that suffers from this, you can use
        the ``shadowed=True`` argument which will use a more precise method to
        calculate the disk coordinates. This can be much slower for large cubes
        as it requires rotating large grids.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Edge of the inner cavity for the
                emission surface in [arcsec].
            r_taper (Optional[float]): Characteristic radius in [arcsec] of the
                exponential taper to the emission surface.
            q_taper (Optional[float]): Exponent of the exponential taper of the
                emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to ``z0``.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            force_positive_surface (Optional[bool]): Force the emission surface
                to be positive, default is ``False``.
            force_negative_surface (Optioanl[bool]): Force the emission surface
                to be negative, default is ``False``.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either ``'polar'`` or ``'cartesian'``.
            shadowed (Optional[bool]): Use a slower, but more precise, method
                for deprojecting the pixel coordinates if the emission surface
                is shadowed.

        Returns:
            Three coordinate arrays, either the cylindrical coordaintes,
            ``(r, theta, z)`` or cartestian coordinates, ``(x, y, z)``,
            depending on ``frame``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Apply the inclination concention.
        inc = inc if inc < 90.0 else inc - 180.0

        # Check that the necessary pairs are provided.
        msg = "Must specify either both or neither of `{}` and `{}`."
        if (z0 is not None) != (psi is not None):
            raise ValueError(msg.format('z0', 'psi'))
        if (z1 is not None) != (phi is not None):
            raise ValueError(msg.format('z1', 'phi'))
        if (r_taper is not None) != (q_taper is not None):
            raise ValueError(msg.format('r_taper', 'q_taper'))
        if (z1 is not None) and (r_taper is not None) and self.verbose:
            print("WARNING: Use a double power law with a tapered edge is not "
                  + "advised due to large degeneracies.")

        # Set the defaults.
        z0 = 0.0 if z0 is None else z0
        psi = 1.0 if psi is None else psi
        z1 = 0.0 if z1 is None else z1
        phi = 1.0 if phi is None else phi
        r_taper = np.inf if r_taper is None else r_taper
        q_taper = 1.0 if q_taper is None else q_taper
        r_cavity = 0.0 if r_cavity is None else r_cavity

        # Faster deprojection for no emission surface.
        if z0 == 0.0 and z_func is None:
            r, t = self._get_midplane_polar_coords(x0, y0, inc, PA)
            z = np.zeros(r.shape)
            if frame == 'cylindrical':
                return r, t, z
            return r * np.cos(t), r * np.sin(t), z

        if force_positive_surface and force_negative_surface:
            raise ValueError("Cannot force positive and negative surface.")
        if force_positive_surface:
            z_min, z_max = 0.0, 1e10
        elif force_negative_surface:
            z_min, z_max = -1e10, 0.0
        else:
            z_min, z_max = -1e10, 1e10

        # Define the emission surface function.
        if z_func is None:
            def z_func(r_in):
                r = np.clip(r_in - r_cavity, a_min=0.0, a_max=None)
                z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
                z *= np.exp(-np.power(r / r_taper, q_taper))
                return np.clip(z, a_min=z_min, a_max=z_max)

        # Calculate the pixel values.
        if shadowed:
            griddata_kw = {} if griddata_kw is None else griddata_kw
            r, t, z = self._get_shadowed_coords(x0, y0, inc, PA, z_func,
                                                extend, oversample,
                                                **griddata_kw)
        else:
            r, t, z = self._get_flared_coords(x0, y0, inc, PA, z_func)

        # Transform coordinate system.
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = imagecube._rotate_coords(x_sky, y_sky, PA)
        return imagecube._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func):
        """Return cylindrical coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(10):
            y_tmp = y_mid + z_func(r_tmp) * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    def _get_shadowed_coords(self, x0, y0, inc, PA, z_func, extend=2.0,
                             oversample=2.0, griddata_kw=None):
        """
        Return cyclindrical coords of surface in [arcsec, rad, arcsec] but
        using a slightly slower method that deals better with large gradients
        in the emission surface.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            extend (Optional[float]): When calculating the disk-frame
                coordinates, extend the axes by this amount. A larger area is
                necessary as for large inclinations as the projected distance
                along the minor axis will be compressed.
            oversample (Optional[float]): Oversample the axes by this factor. A
                larger value will give a more precise deprojection at the cost
                of computation time.
            griddata_kw (Optional[dict]): Any kwargs to be passed to
                ``griddata`` which performs the interpolation.

        Returns:
            Three coordinate arrays, either the cylindrical coordaintes,
            ``(r, theta, z)`` or cartestian coordinates, ``(x, y, z)``,
            depending on ``frame``.
        """

        # Make the disk-frame coordinates.
        diskframe_coords = self._get_diskframe_coords(extend, oversample)
        xdisk, ydisk, rdisk, tdisk = diskframe_coords
        zdisk = z_func(rdisk)

        # Incline the disk.
        inc = np.radians(inc)
        x_dep = xdisk
        y_dep = ydisk * np.cos(inc) - zdisk * np.sin(inc)

        # Remove shadowed pixels.
        if inc < 0.0:
            y_dep = np.maximum.accumulate(y_dep, axis=0)
        else:
            y_dep = np.minimum.accumulate(y_dep[::-1], axis=0)[::-1]

        # Rotate and the disk.
        x_rot, y_rot = imagecube._rotate_coords(x_dep, y_dep, PA)
        x_rot, y_rot = x_rot + x0, y_rot + y0

        # Grid the disk.
        from scipy.interpolate import griddata
        disk = (x_rot.flatten(), y_rot.flatten())
        grid = (self.xaxis[None, :], self.yaxis[:, None])
        griddata_kw = {} if griddata_kw is None else griddata_kw
        griddata_kw['method'] = griddata_kw.get('method', 'nearest')
        r_obs = griddata(disk, rdisk.flatten(), grid, **griddata_kw)
        t_obs = griddata(disk, tdisk.flatten(), grid, **griddata_kw)
        return r_obs, t_obs, z_func(r_obs)

    def _get_diskframe_coords(self, extend=2.0, oversample=0.5):
        """Disk-frame coordinates based on the cube axes."""
        x_disk = np.linspace(extend * self.xaxis[0], extend * self.xaxis[-1],
                             int(self.nxpix * oversample))[::-1]
        y_disk = np.linspace(extend * self.yaxis[0], extend * self.yaxis[-1],
                             int(self.nypix * oversample))
        x_disk, y_disk = np.meshgrid(x_disk, y_disk)
        r_disk = np.hypot(x_disk, y_disk)
        t_disk = np.arctan2(y_disk, x_disk)
        return x_disk, y_disk, r_disk, t_disk

    def sky_to_disk(self, coords, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                    coord_type='cartesian', coord_type_out='cartesian'):
        """
        Deproject the sky plane coordinates into midplane disk-frame
        coordinates. Accounting for non-zero emission heights in progress.

        Args:
            coords (tuple): A tuple of the sky-frame coordinates to transform.
                Must be either cartestian or cylindrical frames specified by
                the ``coord_type`` argument. All spatial coordinates should be
                given in [arcsec], while all angular coordinates should be
                given in [radians].
            x0 (Optional[float]): Source right ascension offset in [arcsec].
            y0 (Optional[float]): Source declination offset in [arcsec].
            inc (float): Inclination of the disk in [deg].
            PA (float): Position angle of the disk, measured Eastwards to the
                red-shifted major axis from North in [deg].
            coord_type (Optional[str]): The coordinate type of the sky-frame
                coordinates, either ``'cartesian'`` or ``'cylindrical'``.
            coord_type_out (Optional[str]): The coordinate type of the returned
                disk-frame coordinates, either ``'cartesian'`` or
                ``'cylindrical'``.

        Returns:
            Two arrays representing the deprojection of the input coordinates
            into the disk-frame, ``x_disk`` and ``y_disk`` if
            ``coord_type_out='cartesian'`` otherwise ``r_disk`` and ``t_disk``.
        """

        c1, c2 = np.squeeze(coords[0]), np.squeeze(coords[1])
        if coord_type.lower() == 'cartesian':
            x_sky, y_sky = c1, c2
        elif coord_type.lower() == 'cylindrical':
            x_sky = c1 * np.cos(c2)
            y_sky = c1 * np.sin(c2)
        else:
            message = "`coord_type` must be 'cartesian' or 'cylindrical'."
            raise ValueError(message)

        x_rot, y_rot = imagecube._rotate_coords(x_sky-x0, y_sky-y0, PA)
        x_disk, y_disk = imagecube._deproject_coords(x_rot, y_rot, inc)

        if coord_type_out.lower() == 'cartesian':
            return x_disk, y_disk
        elif coord_type_out.lower() == 'cylindrical':
            return np.hypot(x_disk, y_disk), np.arctan2(y_disk, x_disk)
        else:
            message = "`coord_type_out` must be 'cartesian' or 'cylindrical'."
            raise ValueError(message)

    def disk_to_sky(self, coords, inc, PA, x0=0.0, y0=0.0,
                    coord_type='cartesian', return_idx=False):
        """
        Project disk-frame coordinates onto the sky plane. Can return either
        the coordinates (the default return), useful for plotting, or the
        pixel indices (if ``return_idx=True``) which can be used to extract a
        spectrum at a particular location.

        Args:
            coords (tuple): A tuple of the disk-frame coordinates to transform.
                Must be either cartestian, cylindrical or spherical frames,
                specified by the ``frame`` argument. If only two coordinates
                are given, the input is assumed to be 2D. All spatial
                coordinates should be given in [arcsec], while all angular
                coordinates should be given in [radians].
            inc (float): Inclination of the disk in [deg].
            PA (float): Position angle of the disk, measured Eastwards to the
                red-shifted major axis from North in [deg].
            x0 (Optional[float]): Source right ascension offset in [arcsec].
            y0 (Optional[float]): Source declination offset in [arcsec].
            coord_type (Optional[str]): Coordinate system used for the disk
                coordinates, either ``'cartesian'``, ``'cylindrical'`` or
                ``'spherical'``.
            return_idx (Optional[bool]): If true, return the index of the
                nearest pixel to each on-sky position.

        Returns:
            Two arrays representing the projection of the input coordinates
            onto the sky, ``x_sky`` and ``y_sky``, unless ``return_idx`` is
            ``True``, in which case the arrays are the indices of the nearest
            pixels on the sky.
        """
        c1, c2 = np.squeeze(coords[0]), np.squeeze(coords[1])
        try:
            c3 = np.squeeze(coords[2])
        except IndexError:
            c3 = np.zeros(c1.size)
        if coord_type.lower() == 'cartesian':
            x, y, z = c1, c2, c3
        elif coord_type.lower() == 'cylindrical':
            x = c1 * np.cos(c2)
            y = c1 * np.sin(c2)
            z = c3
        elif coord_type.lower() == 'spherical':
            x = c1 * np.cos(c2) * np.sin(c3)
            y = c1 * np.sin(c2) * np.sin(c3)
            z = c1 * np.cos(c3)
        else:
            raise ValueError("frame_in must be 'cartestian'," +
                             " 'cylindrical' or 'spherical'.")
        inc = np.radians(inc)
        PA = np.radians(PA - 90.0)
        y_roll = np.cos(inc) * y - np.sin(inc) * z
        x_sky = np.cos(PA) * x - np.sin(PA) * y_roll + x0
        y_sky = -np.sin(PA) * x - np.cos(PA) * y_roll + y0
        if not return_idx:
            return x_sky, y_sky
        x_pix = np.array([abs(self.xaxis - xx).argmin() for xx in x_sky])
        y_pix = np.array([abs(self.yaxis - yy).argmin() for yy in y_sky])
        return x_pix, y_pix

    # -- Spectral Axis Manipulation -- #

    def velocity_to_restframe_frequency(self, velax=None, vlsr=0.0):
        """Return restframe frequency [Hz] of the given velocity [m/s]."""
        velax = self.velax if velax is None else np.squeeze(velax)
        return self.nu0 * (1. - (velax - vlsr) / 2.998e8)

    def restframe_frequency_to_velocity(self, nu, vlsr=0.0):
        """Return velocity [m/s] of the given restframe frequency [Hz]."""
        return 2.998e8 * (1. - nu / self.nu0) + vlsr

    def spectral_resolution(self, dV=None):
        """Convert velocity resolution in [m/s] to [Hz]."""
        dV = dV if dV is not None else self.chan
        nu = self.velocity_to_restframe_frequency(velax=[-dV, 0.0, dV])
        return np.mean([abs(nu[1] - nu[0]), abs(nu[2] - nu[1])])

    def velocity_resolution(self, dnu):
        """Convert spectral resolution in [Hz] to [m/s]."""
        v0 = self.restframe_frequency_to_velocity(self.nu0)
        v1 = self.restframe_frequency_to_velocity(self.nu0 + dnu)
        vA = max(v0, v1) - min(v0, v1)
        v1 = self.restframe_frequency_to_velocity(self.nu0 - dnu)
        vB = max(v0, v1) - min(v0, v1)
        return np.mean([vA, vB])

    # -- Masking Functions -- #

    def keplerian_mask(self, inc, PA, dist, mstar, vlsr, x0=0.0, y0=0.0,
                       z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
                       q_taper=None, dV0=300.0, dVq=-0.5, r_min=0.0, r_max=4.0,
                       nbeams=None, tolerance=0.01, restfreqs=None,
                       max_dz0=0.2, return_type='float'):
        """
        Generate a make based on a Keplerian velocity model. Original code from
        ``https://github.com/richteague/keplerian_mask``. Unlike with the
        original code, the mask will be built on the same cube grid as the
        attached data. Multiple lines can be considered at once by providing a
        list of the rest frequencies of the line.

        Unlike other functions, this does not accept ``z_func``.

        Args:
            inc (float): Inclination of the disk in [deg].
            PA (float): Position angle of the disk, measured Eastwards to the
                red-shifted major axis from North in [deg].
            dist (float): Distance to source in [pc].
            mstar (float): Stellar mass in [Msun].
            vlsr (float): Systemic velocity in [m/s].
            x0 (Optional[float]): Source right ascension offset in [arcsec].
            y0 (Optional[float]): Source declination offset in [arcsec].
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Edge of the inner cavity for the
                emission surface in [arcsec].
            r_taper (Optional[float]): Characteristic radius in [arcsec] of the
                exponential taper to the emission surface.
            q_taper (Optional[float]): Exponent of the exponential taper of the
                emission surface.
            dV0 (Optional[float]): Line Doppler width at 1" in [m/s].
            dVq (Optional[float]): Powerlaw exponent for the Doppler-width
                dependence.
            r_min (Optional[float]): Inner radius to consider in [arcsec].
            r_max (Optional[float]): Outer radius to consider in [arcsec].
            nbeams (Optional[float]): Size of convolution kernel to smooth the
                mask by.
            tolerance (Optional[float]): After smoothing, the limit used to
                decide if a pixel is masked or not. Lower values will include
                more pixels.
            restfreqs (Optional[list]): Rest frequency (or list of rest
                frequencies) in [Hz] to allow for multiple (hyper-)fine
                components.
            max_dz0 (Optional[float]): The maximum step size between different
                ``z0`` values used for the different emission heights.
            return_type (Optional[str]): The value type used for the returned
                mask, the default is ``'float'``.

        Returns:
            ndarry:
                The Keplerian mask with the desired value type.
        """

        # Define the radial line width profile.

        def dV(r):
            return dV0 * r**dVq

        # Calculate the different heights that we'll have to use. For this we
        # use steps of `max_dz0`. The use of `linspace` at the end is to ensure
        # that the steps in z0 are equal.

        if z0 != 0.0:
            z0s = np.arange(0.0, z0, max_dz0)
            z0s = np.append(z0s, z0) if z0s[-1] != z0 else z0s
            z0s = np.concatenate([-z0s[1:][::-1], z0s])
            z0s = np.linspace(z0s[0], z0s[-1], z0s.size)
        else:
            z0s = np.zeros(1)

        # For each line center we need to loop through the different emission
        # heights. Each mask is where the line center, ``v_kep``, is within the
        # local linewidth, ``dV``, and half a channel. We then collapse all the
        # masks down to a single mask.

        masks = []
        for _z0 in z0s:
            for restfreq in np.atleast_1d(restfreqs):
                offset = self.restframe_frequency_to_velocity(restfreq)
                rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                         z0=_z0, psi=psi, r_cavity=r_cavity,
                                         r_taper=r_taper, q_taper=q_taper)[0]
                v_kep = self.keplerian(x0=x0, y0=y0, inc=inc, PA=PA,
                                       mstar=mstar, dist=dist,
                                       vlsr=vlsr+offset, z0=_z0,
                                       psi=psi, r_cavity=r_cavity,
                                       r_taper=r_taper, q_taper=q_taper,
                                       r_min=r_min, r_max=r_max)
                mask = abs(self.velax[:, None, None] - v_kep)
                masks += [mask < dV(rvals) + self.chan]
        mask = np.any(masks, axis=0).astype('float')
        assert mask.shape == self.data.shape, "wrong mask shape"

        # Apply smoothing to the mask to broaden or soften the edges. Anything
        # that results in a value above ``tolerance`` is assumed to be within
        # the mask.

        if nbeams:
            mask = self.convolve_with_beam(mask, scale=float(nbeams))
        return np.where(mask >= tolerance, 1.0, 0.0).astype(return_type)

    def _string_to_Hz(self, string):
        """
        Convert a string to a frequency in [Hz].
        """
        if isinstance(string, float):
            return string
        if isinstance(string, int):
            return string
        factor = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
        for key in ['GHz', 'MHz', 'kHz', 'Hz']:
            if key in string:
                return float(string.replace(key, '')) * factor[key]

    # -- FITS I/O -- #

    def _read_FITS(self, path):
        """Reads the data from the FITS file."""

        # File names.
        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]

        # FITS data.
        self.header = fits.getheader(path)
        self.data = np.squeeze(fits.getdata(self.path))
        self.data = np.where(np.isfinite(self.data), self.data, 0.0)
        try:
            self.bunit = self.header['bunit']
        except KeyError:
            if self._user_bunit is not None:
                self.bunit = self._user_bunit
            else:
                print("WARNING: Not `bunit` header keyword found.")
                self.bunit = input("\t Enter brightness unit: ")

        # Position axes.
        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)
        self.dpix = np.mean([abs(np.diff(self.xaxis))])
        self.xaxis -= 0.5*self.dpix
        self.yaxis -= 0.5*self.dpix
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size

        # Spectral axis.
        self.nu0 = self._readrestfreq()
        try:
            self.velax = self._readvelocityaxis()
            if self.velax.size > 1:
                self.chan = np.mean(np.diff(self.velax))
            else:
                self.chan = np.nan
            self.freqax = self._readfrequencyaxis()
            if self.chan < 0.0:
                self.data = self.data[::-1]
                self.velax = self.velax[::-1]
                self.freqax = self.freqax[::-1]
                self.chan *= -1.0
        except KeyError:
            self.velax = None
            self.chan = None
            self.freqax = None
        try:
            self.channels = np.arange(self.velax.size)
        except AttributeError:
            self.channels = [0]

        # Check that the data is saved such that increasing indices in x are
        # decreasing in offset counter to the yaxis.
        if np.diff(self.xaxis).mean() > 0.0:
            self.xaxis = self.xaxis[::-1]
            self.data = self.data[:, ::-1]

        # Beam.
        self._read_beam()

    def _read_beam(self):
        """Reads the beam properties from the header."""
        try:
            if self.header.get('CASAMBM', False):
                beam = fits.open(self.path)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                self.bmaj, self.bmin, self.bpa = beam
            else:
                self.bmaj = self.header['bmaj'] * 3600.
                self.bmin = self.header['bmin'] * 3600.
                self.bpa = self.header['bpa']
            self.beamarea_arcsec = self._calculate_beam_area_arcsec()
            self.beamarea_str = self._calculate_beam_area_str()
        except Exception:
            print("WARNING: No beam values found. Assuming pixel as beam.")
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea_arcsec = self.dpix**2.0
            self.beamarea_str = np.radians(self.dpix / 3600.)**2.0
        self.bpa %= 180.0

    def print_beam(self):
        """Print the beam properties."""
        print('{:.2f}" x {:.2f}" at {:.1f} deg'.format(*self.beam))

    @property
    def beam(self):
        return self.bmaj, self.bmin, self.bpa

    @property
    def beams_per_pix(self):
        """Number of beams per pixel."""
        return self.dpix**2.0 / self.beamarea_arcsec

    @property
    def pix_per_beam(self):
        """Number of pixels in a beam."""
        return self.beamarea_arcsec / self.dpix**2.0

    @property
    def FOV(self):
        """Field of view."""
        return self.xaxis.max() - self.xaxis.min()

    def _clip_cube_velocity(self, v_min=None, v_max=None):
        """Clip the cube to within ``vmin`` and ``vmax``."""
        v_min = self.velax[0] if v_min is None else v_min
        v_max = self.velax[-1] if v_max is None else v_max
        i = abs(self.velax - v_min).argmin()
        i += 1 if self.velax[i] < v_min else 0
        j = abs(self.velax - v_max).argmin()
        j -= 1 if self.velax[j] > v_max else 0
        self.velax = self.velax[i:j+1]
        self.data = self.data[i:j+1]

    def _clip_cube_spatial(self, radius):
        """Clip the cube plus or minus clip arcseconds from the origin."""
        if radius > min(self.xaxis.max(), self.yaxis.max()):
            if self.verbose:
                print('WARNING: FOV = {:.1f}" larger than '.format(radius * 2)
                      + 'FOV of cube: {:.1f}".'.format(self.xaxis.max() * 2))
        else:
            xa = abs(self.xaxis - radius).argmin()
            if self.xaxis[xa] < radius:
                xa -= 1
            xb = abs(self.xaxis + radius).argmin()
            if -self.xaxis[xb] < radius:
                xb += 1
            xb += 1
            ya = abs(self.yaxis + radius).argmin()
            if -self.yaxis[ya] < radius:
                ya -= 1
            yb = abs(self.yaxis - radius).argmin()
            if self.yaxis[yb] < radius:
                yb += 1
            yb += 1
            if self.data.ndim == 3:
                self.data = self.data[:, ya:yb, xa:xb]
            else:
                self.data = self.data[ya:yb, xa:xb]
            self.xaxis = self.xaxis[xa:xb]
            self.yaxis = self.yaxis[ya:yb]
            self.nxpix = self.xaxis.size
            self.nypix = self.yaxis.size

    def _readspectralaxis(self, a):
        """Returns the spectral axis in [Hz] or [m/s]."""
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        return a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del

    def _readpositionaxis(self, a=1):
        """Returns the position axis in [arcseconds]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [1, 2].")
        try:
            a_len = self.header['naxis%d' % a]
            a_del = self.header['cdelt%d' % a]
            a_pix = self.header['crpix%d' % a]
        except KeyError:
            if self._user_pixel_scale is None:
                print('WARNING: No axis information found.')
                _input = input("\t Enter pixel scale size in [arcsec]: ")
                self._user_pixel_scale = float(_input) / 3600.0
            a_len = self.data.shape[-1] if a == 1 else self.data.shape[-2]
            if a == 1:
                a_del = -1.0 * self._user_pixel_scale
            else:
                a_del = 1.0 * self._user_pixel_scale
            a_pix = a_len / 2.0 + 0.5
        axis = (np.arange(a_len) - a_pix + 1.0) * a_del
        return 3600 * axis

    def _readrestfreq(self):
        """Read the rest frequency."""
        try:
            nu = self.header['restfreq']
        except KeyError:
            try:
                nu = self.header['restfrq']
            except KeyError:
                try:
                    nu = self.header['crval3']
                except KeyError:
                    nu = np.nan
        return nu

    def _readvelocityaxis(self):
        """Wrapper for _velocityaxis and _spectralaxis."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype%d' % a].lower():
            specax = self._readspectralaxis(a)
            velax = (self.nu0 - specax) * sc.c
            velax /= self.nu0
        else:
            velax = self._readspectralaxis(a)
        return velax

    def _readfrequencyaxis(self):
        """Returns the frequency axis in [Hz]."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype3'].lower():
            return self._readspectralaxis(a)
        return self._readrestfreq() * (1.0 - self._readvelocityaxis() / sc.c)

    def _calculate_symmetric_velocity_axis(self):
        """Returns a symmetric velocity axis for decorrelation functions."""
        try:
            velax_symmetric = np.arange(self.velax.size).astype('float')
        except AttributeError:
            return np.array([0.0])
        velax_symmetric -= velax_symmetric.max() / 2
        if abs(velax_symmetric).min() > 0.0:
            velax_symmetric -= abs(velax_symmetric).min()
        if abs(velax_symmetric[0]) < abs(velax_symmetric[-1]):
            velax_symmetric = velax_symmetric[:-1]
        elif abs(velax_symmetric[0]) > abs(velax_symmetric[-1]):
            velax_symmetric = velax_symmetric[1:]
        velax_symmetric *= self.chan
        return velax_symmetric

    def frequency(self, vlsr=0.0, unit='GHz'):
        """
        A `velocity_to_restframe_frequency` wrapper with unit conversion.

        Args:
            vlsr (optional[float]): Sytemic velocity in [m/s].
            unit (optional[str]): Unit for the output axis.

        Returns:
            1D array of frequency values.
        """
        return self.frequency_offset(nu0=0.0, vlsr=vlsr, unit=unit)

    def frequency_offset(self, nu0=None, vlsr=0.0, unit='MHz'):
        """
        Return the frequency offset relative to `nu0` for easier plotting.

        Args:
            nu0 (optional[float]): Reference restframe frequency in [Hz].
            vlsr (optional[float]): Sytemic velocity in [m/s].
            unit (optional[str]): Unit for the output axis.

        Returns:
            1D array of frequency values.
        """
        nu0 = self.nu0 if nu0 is None else nu0
        nu = self.velocity_to_restframe_frequency(vlsr=vlsr)
        return (nu - nu0) / imagecube.frequency_units[unit]

    # -- Unit Conversions -- #

    def jybeam_to_Tb_RJ(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using Rayleigh-Jeans approximation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return jy2k * data / self._calculate_beam_area_str()

    def jybeam_to_Tb(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Tb = 1e-26 * abs(data) / self._calculate_beam_area_str()
        Tb = 2.0 * sc.h * nu**3 / Tb / sc.c**2
        Tb = sc.h * nu / sc.k / np.log(Tb + 1.0)
        return np.where(data >= 0.0, Tb, -Tb)

    def Tb_to_jybeam_RJ(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using Rayleigh-Jeans approxmation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return data * self._calculate_beam_area_str() / jy2k

    def Tb_to_jybeam(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Fnu = 2. * sc.h * nu**3 / sc.c**2
        Fnu /= np.exp(sc.h * nu / sc.k / abs(data)) - 1.0
        Fnu *= self._calculate_beam_area_str() / 1e-26
        return np.where(data >= 0.0, Fnu, -Fnu)

    def _calculate_beam_area_arcsec(self):
        """Beam area in square arcseconds."""
        omega = self.bmin * self.bmaj
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _calculate_beam_area_str(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    # -- Utilities -- #

    def estimate_RMS(self, N=5, r_in=0.0, r_out=1e10):
        """
        Estimate RMS of the cube based on first and last `N` channels and a
        circular area described by an inner and outer radius.

        Args:
            N (int): Number of edge channels to include.
            r_in (float): Inner edge of pixels to consider in [arcsec].
            r_out (float): Outer edge of pixels to consider in [arcsec].

        Returns:
            RMS (float): The RMS based on the requested pixel range.
        """
        r_dep = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        rmask = np.logical_and(r_dep >= r_in, r_dep <= r_out)
        rms = np.concatenate([self.data[:int(N)], self.data[-int(N):]])
        rms = np.where(rmask[None, :, :], rms, np.nan)
        return np.sqrt(np.nansum(rms**2) / np.sum(np.isfinite(rms)))

    def print_RMS(self, N=5, r_in=0.0, r_out=1e10):
        """Print the estimated RMS in Jy/beam and K (using RJ approx.)."""
        rms = self.estimate_RMS(N, r_in, r_out)
        rms_K = self.jybeam_to_Tb_RJ(rms)
        print('{:.2f} mJy/beam ({:.2f} K)'.format(rms * 1e3, rms_K))

    def correct_PB(self, path):
        """Correct for the primary beam given by ``path``."""
        if self._pb_corrected:
            raise ValueError("This data has already been PB corrected.")
        pb = np.squeeze(fits.getdata(path))
        if pb.shape == self.data.shape:
            self.data /= pb
        else:
            self.data /= pb[None, :, :]
        self._pb_corrected = True

    def shift_image(self, x0=0.0, y0=0.0, data=None):
        """
        Shift the source center of the provided data by ``d0`` [arcsec] and
        ``y0`` [arcsec] in the x- and y-directions, respectively. The shifting
        is performed with ``scipy.ndimage.shift`` which uses a third-order
        spline interpolation.

        Args:
            x0 (Optional[float]): Shfit along the x-axis in [arcsec].
            y0 (Optional[float]): Shifta long the y-axis in [arcsec].
            data (Optional[ndarray]): Data to shift.
        Returns:
            ndarray:
                The shifted array.
        """
        from scipy.ndimage import shift
        data = data.copy() if data is not None else self.data.copy()
        data = np.where(np.isfinite(data), data, 0.0)
        y0, x0 = -y0 / self.dpix, x0 / self.dpix
        if y0 == x0 == 0.0:
            return data
        if data.ndim == 3:
            from tqdm import trange
            shifted = []
            for cidx in trange(data.shape[0]):
                shifted += [shift(data[cidx], [y0, x0])]
            return np.squeeze(shifted)
        if data.ndim == 2:
            return shift(data, [y0, x0])

    def rotate_image(self, PA, data=None):
        """
        Rotate the image such that the red-shifted axis aligns with the x-axis.

        Args:
            PA (float): Position angle of the disk, measured to the red-shifted
                major axis of the disk, anti-clockwise from North, in [deg].
            data (Optional[ndarray]): Data to rotate if not the attached data.

        Returns:
            ndarray:
                The rotated array.
        """
        from scipy.ndimage import rotate
        data = data if data is not None else self.data
        data = np.where(np.isfinite(data), data, 0.0)
        if data.ndim == 3:
            from tqdm import trange
            shifted = []
            for cidx in trange(data.shape[0]):
                shifted += [rotate(data[cidx], PA - 90.0, reshape=False)]
            return np.squeeze(shifted)
        if data.ndim == 2:
            return rotate(data, PA - 90.0, reshape=False)

    @property
    def rms(self):
        """RMS of the cube based on the first and last 5 channels."""
        return self.estimate_RMS(N=5)

    @property
    def extent(self):
        """Cube field of view for use with Matplotlib's ``imshow``."""
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    def get_spectrum(self, coords, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                     frame='sky', coord_type='cartesian', area=0.0,
                     beam_weighting=False, return_mask=False):
        """
        Return a spectrum at a position defined by a coordinates given either
        in sky-frame position (``frame='sky'``) or a disk-frame location
        (``frame='disk'``). The coordinates can be either in cartesian or
        cylindrical frames set by ``coord_type``.

        By default the returned spectrum is extracted at the pixel closest to
        the provided coordinates. If ``area`` is set to a positive value, then
        a beam-shaped area is averaged over, where ``area`` sets the size of
        this region in number of beams. For example ``area=2.0`` will result
        in an average over an area twice the size of the beam.

        If an average is averaged over, you can also weight the pixels by the
        beam response with ``beam_weighting=True``. This will reduce the weight
        of pixels that are further away from the beam center.

         Finally, to check that you're extracting what you think you are, you
         can return the mask (and weights) used for the extraction with
         ``return_mask=True``. Note that if ``beam_weighting=False`` then all
         ``weights`` will be 1.

         TODO: Check that the returned uncertainties are reasonable.

        Args:
            coords (tuple): The coordinates from where you want to extract a
                spectrum. Must be a length 2 tuple.
            x0 (Optional[float]): RA offset in [arcsec].
            y0 (Optional[float]): Dec offset in [arcsec].
            inc (Optional[float]): Inclination of source in [deg]. Only
                required for ``frame='disk'``.
            PA (Optional[float]): Position angle of source in [deg]. Only
                required for ``frame='disk'``.
            frame (Optional[str]): The frame that the ``coords`` are given.
                Either ``'disk'`` or ``'sky'``.
            coord_type (Optional[str]): The type of coordinates given, either
                ``'cartesian'`` or ``'cylindrical'``.
            area (Optional[float]): The area to average over in units of the
                beam area. Note that this take into account the beam aspect
                ratio and position angle. For a single pixel extraction use
                ``area=0.0``.
            beam_weighting (Optional[bool]): Whether to use the beam response
                function to weight the averaging of the spectrum.
            return_mask (Optional[bool]): Whether to return the mask and
                weights used to extract the spectrum.

        Retuns (if ``return_mask=False``):
            x, y, dy (arrays): The velocity axis, extracted spectrum and
            associated uncertainties.
        (if ``return_mask=True``):
            mask, weights (arrays): Arrays of the mask used to extract the
            spectrum and the weighted used for the averaging.
        """

        # Convert the input coordinate into on-sky cartesian coordinates
        # relative to the center of the image.

        if frame.lower() == 'sky':
            if inc != 0.0 or PA != 0.0:
                message = "WARNING: You shouldn't need to specify `inc` or "
                message += "`PA` when using `frame='sky'`."
                print(message)
            c1 = np.squeeze(coords[0])
            c2 = np.squeeze(coords[1])
            if coord_type.lower() == 'cartesian':
                x, y = c1 + x0, c2 + y0
            elif coord_type.lower() == 'cylindrical':
                x = x0 + c1 * np.cos(c2 - np.radians(90.0))
                y = y0 - c1 * np.sin(c2 - np.radians(90.0))
        elif frame.lower() == 'disk':
            x, y = self.disk_to_sky(coords=coords, coord_type=coord_type,
                                    inc=inc, PA=PA, x0=x0, y0=y0)
        assert x.size == y.size == 1

        # Define the area to average over.

        if area == 0.0:
            x_pix = abs(self.xaxis - x).argmin()
            y_pix = abs(self.yaxis - y).argmin()
            mask = np.zeros(self.data[0].shape)
            weights = np.zeros(mask.shape)
            mask[y_pix, x_pix] = 1
            weights[y_pix, x_pix] = 1
        elif area > 0.0:
            mask = self._beam_mask(x, y, stretch=area)
            weights = self._beam_mask(x, y, stretch=area, response=True)
        else:
            raise ValueError("`area` must be a non-negative value.")
        weights = weights if beam_weighting else mask

        # If requested, return the mask and the weighting instead.

        if return_mask:
            return mask, weights

        # Otherwise, extract the spectrum and average it.

        y = [np.average(c * mask, weights=weights) for c in self.data]
        dy = max(1.0, mask.sum() * self.beams_per_pix)**-0.5 * self.rms
        return self.velax, np.array(y), np.array([dy for _ in y])

    def convolve_with_beam(self, data, scale=1.0, circular=False,
                           convolve_kwargs=None):
        """
        Convolve the attached data with a 2D Gaussian kernel matching the
        synthesized beam. This can be scaled with ``scale``, or forced to be
        circular (taking the major axis as the radius of the beam).

        Args:
            data (ndarray): The data to convolve. Must be either 2D or 3D.
            scale (Optional[float]): Factor to scale the synthesized beam by.
            circular (Optional[bool]): Force a cicular kernel. If ``True1``,
                the kernel will adopt the scaled major axis of the beam to use
                as the radius.
            convolve_kwargs (Optional[dict]): Keyword arguments to pass to
                ``astropy.convolution.convolve``.

        Returns:
            ndarray:
                Data convolved with the requested kernel.
        """
        from astropy.convolution import convolve, Gaussian2DKernel
        kw = {} if convolve_kwargs is None else convolve_kwargs
        kw['preserve_nan'] = kw.pop('preserve_nan', True)
        kw['boundary'] = kw.pop('boundary', 'fill')
        bmaj = scale * self.bmaj / self.dpix / 2.355
        bmin = scale * self.bmin / self.dpix / 2.355
        kernel = Gaussian2DKernel(x_stddev=bmaj if circular else bmin,
                                  y_stddev=bmaj, theta=self.bpa)
        if data.ndim == 3:
            from tqdm import trange
            convolved = []
            for cidx in trange(data.shape[0]):
                convolved += [convolve(data[cidx], kernel, **kw)]
            return np.squeeze(convolved)
        elif data.ndim == 2:
            return convolve(data, kernel, **kw)
        else:
            raise ValueError("`data` must be 2 or 3 dimensional.")

    def cross_section(self, x0=0.0, y0=0.0, PA=0.0, mstar=1.0, dist=100.,
                      vlsr=None, grid=True, grid_spacing=None, downsample=1,
                      cylindrical_rotation=False, clip_noise=True, min_npnts=5,
                      statistic='mean', mask_velocities=None):
        """
        Return the cross section of the data following Dutrey et al. (2017).
        This yields ``I_nu(r, z)``. If ``grid=True`` then this will be gridded
        using ``scipy.interpolate.griddata`` onto axes with the same pixel
        spacing as the attached data.

        Reference:
            Dutrey et al. (2017):
                https://ui.adsabs.harvard.edu/abs/2017A%26A...607A.130D

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            PA (Optional[float]): Position angle of the disk in [deg].
            mstar (Optional[float]): Mass of the central star in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            vlsr (Optional[float]): Systemic velocity in [m/s]. If ``None``,
                assumes the central velocity.
            grid (Optional[bool]): Whether to grid the coordinates to a regular
                grid. Default is ``True``.
            grid_spacing (Optional[float]): The spacing, in [arcsec], for the R
                and Z grids. If ``None`` is provided, will use pixel spacing.
            downsample (Optional[int]): If provided, downsample the coordinates
                to grid by this factor to speed up the interpolation for large
                datasets. Default is ``1``.
            cylindrical_rotation (Optional[bool]): If ``True``, assume that the
                Keplerian rotation decreases with height above the midplane.
            clip_noise (Optional[bool]): If ``True``, remove all pixels which
                fall below 3 times the standard deviation of the two edge
                channels. If the argument is a ``float``, use this as the clip
                level.
            min_npnts (Optional[int]): Number of minimum points in each bin for
                the average. Default is 5.
            statistic (Optional[str]): Statistic to calculate for each bin.
                Note that the uncertainty returned will only make sense with
                ``'max'``, ``'mean'`` or ``'median'``.
            mask_velocities (Optional[list of tuples]): List of
                ``(v_min, v_max)`` tuples to mask (i.e. remove from the
                averaging).
        Returns:
            ndarray: Either two 1D arrays containing ``(r, z, I_nu)``, or, if
            ``grid=True``, two 1D arrays with the ``r`` and ``z`` axes and
            two 2D array of ``I_nu`` and ``dI_nu``.
        """

        # Pixel coordinates.

        v_sky = np.ones(self.data.shape) * self.velax[:, None, None]
        x_sky = np.ones(self.data.shape) * self.xaxis[None, None, :]
        y_sky = np.ones(self.data.shape) * self.yaxis[None, :, None]
        v_sky -= np.median(self.velax) if vlsr is None else vlsr
        x_sky *= dist * sc.au
        y_sky *= dist * sc.au

        # Shift the emission distribution and rotate if the major axis is not
        # aligned with the x-axis.

        if (x0 == 0.0) & (y0 == 0.0):
            intensity = self.data.copy()
        else:
            intensity = self.shift_center(x0, y0, data=self.data)
        if not ((PA == 90.0) or PA == (270.0)):
            intensity = self.rotate_image(PA, data=intensity)

        # Remove the maked velocities pixels.
        # TODO - Include here the Keplerian masks.

        if mask_velocities:
            mask = [np.logical_and(self.velax <= v_min, self.velax >= v_max)
                    for v_min, v_max in np.atleast_2d(mask_velocities)]
            mask = ~np.any(mask, axis=0)
            mask = mask[:, None, None] * np.ones(v_sky.shape).astype(bool)
            x_sky, y_sky, v_sky = x_sky[mask], y_sky[mask], v_sky[mask]

        # Transformation assuming cylindrical rotation.

        R = np.power(sc.G * 1.9891e30 * mstar * (x_sky / v_sky)**2, 2./3.)
        if not cylindrical_rotation:
            R -= y_sky**2
        R = np.sqrt(R) / sc.au / dist
        Z = y_sky / sc.au / dist

        # Return the raw pixel values if necessary.

        if not grid:
            return R, Z, I

        # Flatten the data and remove NaNs.

        R, Z, I = R.flatten(), Z.flatten(), I.flatten()
        mask = np.isfinite(I) & np.isfinite(R)
        R, Z, I = R[mask], Z[mask], I[mask]

        # Remove noise.
        if clip_noise:
            if isinstance(clip_noise, bool):
                clip_noise = 3.0 * np.nanstd([cube.data[0], cube.data[-1]])
            mask = I >= clip_noise
            R, Z, I = R[mask], Z[mask], I[mask]

        # Downsample the data to speed the averaging.
        # TODO: Check if this is actually necessary.

        if downsample > 1:
            downsample = int(downsample)
            R, Z, I = R[::downsample], Z[::downsample], I[::downsample]

        # Define the grids.

        R_grid = cube.xaxis.copy()[cube.xaxis >= 0.0]
        R_grid = R_grid[np.argsort(R_grid)]
        if grid_spacing is not None:
            R_grid = np.arange(0.0, R_grid[-1], grid_spacing)
        R_bins = cube.radial_sampling(rvals=R_grid)[0]
        Z_grid = cube.yaxis.copy()
        if grid_spacing is not None:
            Z_grid = np.arange(Z_grid[0], Z_grid[-1], grid_spacing)
        Z_bins = cube.radial_sampling(rvals=Z_grid)[0]

        # Grid the data.

        from scipy.stats import binned_statistic_2d
        I_grid = binned_statistic_2d(R, Z, I, bins=[R_bins, Z_bins],
                                     statistic=statistic)[0]
        dI_grid = binned_statistic_2d(R, Z, I, bins=[R_bins, Z_bins],
                                      statistic='std')[0]
        N_pnts = binned_statistic_2d(R, Z, I, bins=[R_bins, Z_bins],
                                     statistic='count')[0]
        I_grid = np.where(N_pnts >= min_npnts, I_grid, np.nan)
        dI_grid = np.where(N_pnts >= min_npnts, dI_grid / N_pnts**0.5, np.nan)
        return R_grid, Z_grid, I_grid.T, dI_grid.T

    # -- Spiral Functions -- #

    def spiral_coords(self, r_p, t_p, m=None, r_min=None, r_max=None,
                      mstar=1.0, T0=20.0, Tq=-0.5, dist=100., clockwise=True,
                      frame_out='cartesian'):
        """
        Spiral coordinates from Bae & Zhaohuan (2018a). In order to recover the
        linear spirals from Rafikov (2002), use m >> 1.
        Args:
            r_p (float): Orbital radius of the planet in [arcsec].
            t_p (float): Polar angle of planet relative to the red-shifted
                major axis of the disk in [radians].
            m (optional[int]): Azimuthal wavenumber of the spiral. If not
                specified, will assume the dominant term based on the rotation
                and temperature profiles.
            r_min (optional[float]): Inner radius of the spiral in [arcsec].
            r_max (optional[float]): Outer radius of the spiral in [arcsec].
            mstar (optioanl[float]): Stellar mass of the central star in [Msun]
                to calculate the rotation profile.
            T0 (optional[float]): Gas temperature in [K] at 1 arcsec.
            Tq (optional[float]): Exoponent of the radial gas temperature
                profile.
            dist (optional[float]): Source distance in [pc] used to scale
                [arcsec] to [au] in the calculation of the rotation profile.
            clockwise (optional[bool]): Direction of the spiral.
            frame_out (optional[str]): Coordinate frame of the returned values,
                either 'cartesian' or 'cylindrical'.
        Returns:
            ndarray:
                Coordinates of the spiral in either cartestian or cylindrical
                frame.
        """

        # Define the radial grid in [arcsec].
        r_min = 0.1 if r_min is None else r_min
        r_max = self.xaxis.max() if r_max is None else r_max
        rvals = np.arange(r_min, r_max, 0.1 * self.dpix)
        clockwise = -1.0 if clockwise is True else 1.0

        # Define the physical properties as a function of radius. SI units.
        omega = np.sqrt(sc.G * mstar * 1.988e30 * (rvals * sc.au * dist)**-3)
        tgas = T0 * np.power(rvals, Tq)
        cs = np.sqrt(sc.k * tgas / 2.37 / sc.m_p)
        H = cs / omega

        # Define the dominant wave number if not defined.
        if m is None:
            m = 0.5 * (r_p * dist * sc.au / H)[abs(rvals - r_p).argmin()]
        m = np.round(m)
        rmn = r_p * dist * sc.au * (1.0 - 1.0 / m)**(2./3.)
        rmp = r_p * dist * sc.au * (1.0 + 1.0 / m)**(2./3.)

        # Integrate the equation numerically.
        x = rvals * dist * sc.au
        y = omega * np.sqrt(abs((1 - (rvals / r_p)**(3./2.))**2 - m**-2.)) / cs
        idx_n = abs(rvals * sc.au * dist - rmn).argmin()
        idx_p = abs(rvals * sc.au * dist - rmp).argmin()
        phi = np.ones(rvals.size) * t_p

        for i, r in enumerate(x):
            phi[i] = t_p - np.sign(r - r_p) * np.pi / 4. / m
            if r <= rmn:
                phi[i] -= clockwise * np.trapz(y[i:idx_n+1][::-1],
                                               x=x[i:idx_n+1][::-1])
            elif r >= rmp:
                phi[i] -= clockwise * np.trapz(y[idx_p:i+1],
                                               x=x[idx_p:i+1])
            else:
                phi[i] = np.nan

        # Return the spirals.
        if frame_out == 'cylindrical':
            return rvals, phi
        return rvals * np.cos(phi), rvals * np.sin(phi)

    # -- Plotting Functions -- #

    def plot_center(self, x0s, y0s, SNR, normalize=True):
        """Plot the array of SNR values."""

        # Find the center.
        SNR_mask = np.isfinite(SNR)
        SNR = np.where(SNR_mask, SNR, -1e10)
        yidx, xidx = np.unravel_index(SNR.argmax(), SNR.shape)
        SNR = np.where(SNR_mask, SNR, np.nan)
        x0, y0 = x0s[xidx], y0s[yidx]
        print('Peak SNR at (x0, y0) = ({:.2f}", {:.2f}").'.format(x0, y0))

        # Define the levels.
        if normalize:
            SNR /= SNR[abs(y0s).argmin(), abs(x0s).argmin()]
            vmax = abs(SNR - 1.0).max()
            vmin = 1.0 - vmax
            vmax = 1.0 + vmax
            cb_label = 'Normalized SNR'
        else:
            vmax = SNR.min()
            vmin = SNR.max()
            cb_label = 'SNR'

        # Plot the figure.
        _, ax = plt.subplots()
        im = ax.imshow(SNR, extent=[x0s[0], x0s[-1], y0s[0], y0s[-1]],
                       origin='lower', vmax=vmax, vmin=vmin, cmap='RdGy_r')
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label(cb_label, rotation=270, labelpad=13)
        ax.scatter(x0, y0, marker='x', s=60, c='w', linewidths=2.)
        ax.scatter(x0, y0, marker='x', s=30, c='k', linewidths=1.)
        ax.set_aspect(1)
        ax.set_xlim(x0s[-1], x0s[0])
        ax.set_ylim(y0s[0], y0s[-1])
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        self._plot_beam(ax=ax)

    def plot_teardrop(self, inc, PA, mstar, dist, ax=None, rvals=None,
                      rbins=None, dr=None, x0=0.0, y0=0.0, z0=None, psi=None,
                      r_cavity=None, r_taper=None, q_taper=None, z1=None,
                      phi=None, z_func=None, resample=1, beam_spacing=False,
                      r_min=None, r_max=None, PA_min=None, PA_max=None,
                      exclude_PA=False, abs_PA=False, mask_frame='disk',
                      mask=None, unit='Jy/beam', pcolormesh_kwargs=None,
                      shadowed=False):
        """
        Make a `teardrop` plot. For argument descriptions see
        ``radial_spectra``. For all properties related to ``pcolormesh``,
        include them in ``pcolormesh_kwargs`` as a dictionary, e.g.

            pcolormesh_kwargs = dict(cmap='inferno', vmin=0.0, vmax=1.0)

        This will override any of the default style parameters.
        """

        # Grab the spectra.
        out = self.radial_spectra(rbins=rbins, rvals=rvals, dr=dr, x0=x0,
                                  y0=y0, inc=inc, PA=PA, z0=z0, psi=psi, z1=z1,
                                  phi=phi, r_cavity=r_cavity, r_taper=r_taper,
                                  q_taper=q_taper, z_func=z_func, mstar=mstar,
                                  dist=dist, resample=resample,
                                  beam_spacing=beam_spacing, r_min=r_min,
                                  r_max=r_max, PA_min=PA_min, PA_max=PA_max,
                                  exclude_PA=exclude_PA, abs_PA=abs_PA,
                                  mask_frame=mask_frame, mask=mask, unit=unit,
                                  shadowed=shadowed)
        rvals, velax, spectra, scatter = out

        # Generate the axes.
        if ax is None:
            fig, ax = plt.subplots()

        # Plot the figure.
        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}
        pcolormesh_kwargs['cmap'] = pcolormesh_kwargs.pop('cmap', 'inferno')
        im = ax.pcolormesh(velax / 1e3, rvals, spectra, **pcolormesh_kwargs)
        cb = plt.colorbar(im, pad=0.02)
        cb.set_label('({})'.format(unit), rotation=270, labelpad=13)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Radius (arcsec)')
        return ax

    def plot_beam(self, ax, x0=0.1, y0=0.1, **kwargs):
        """
        Plot the sythensized beam on the provided axes.

        Args:
            ax (matplotlib axes instance): Axes to plot the FWHM.
            x0 (float): Relative x-location of the marker.
            y0 (float): Relative y-location of the marker.
            kwargs (dic): Additional kwargs for the style of the plotting.
        """
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((x0, y0)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.pop('hatch', '////////'),
                       lw=kwargs.pop('linewidth', kwargs.pop('lw', 1)),
                       color=kwargs.pop('color', kwargs.pop('c', 'k')),
                       zorder=kwargs.pop('zorder', 1000), **kwargs)
        ax.add_patch(beam)

    def plot_surface(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=None, psi=None,
                     r_cavity=None, r_taper=None, q_taper=None, z_func=None,
                     shadowed=False, r_max=None, fill=None, ax=None,
                     contour_kwargs=None, imshow_kwargs=None, return_fig=True):
        """
        Overplot the assumed emission surface.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Edge of the inner cavity for the
                emission surface in [arcsec].
            r_taper (Optional[float]): Characteristic radius in [arcsec] of the
                exponential taper to the emission surface.
            q_taper (Optional[float]): Exponent of the exponential taper of the
                emission surface.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            shadowed (Optional[bool]): If ``True``, use the slower, but more
                robust method for deprojecting pixel values.
            r_max (Optional[float]): Maximum radius in [arcsec] to plot the
                emission surface out to.
            fill (Optional[str]): A string to execute (be careful!) to fill in
                the emission surface using ``rvals``, ``tvals`` and ``zvals``
                as returned by ``disk_coords``. For example, to plot the radial
                values use ``fill='rvals'``. To plot the projection of
                rotational velocities, use ``fill='rvals * np.cos(tvals)'``.
            ax (Optional[matplotlib axis instance]): Axis to plot onto.
                contour_kwargs (Optional[dict]): Kwargs to pass to
                ``matplolib.contour`` to overplot the mesh.
            return_fig (Optional[bool]): If ``True``, return the figure for
                additional plotting.

        Returns:
            The ``ax`` instance.
        """

        # Generate the axes.

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        # Get the disk-frame coordinates.

        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0,
                                               inc=inc, PA=PA,
                                               z0=z0, psi=psi,
                                               r_cavity=r_cavity,
                                               r_taper=r_taper,
                                               q_taper=q_taper,
                                               z_func=z_func,
                                               shadowed=shadowed)

        # Mask the data based on r_max.

        r_max = np.nanmax(rvals) if r_max is None else r_max
        zvals = np.where(rvals <= r_max, zvals, np.nan)
        tvals = np.where(rvals <= r_max, tvals, np.nan)
        tvals = np.where(rvals >= 0.5 * self.bmaj, tvals, np.nan)
        rvals = np.where(rvals <= r_max, rvals, np.nan)

        # Fill in the background.

        if fill is not None:
            kw = {} if imshow_kwargs is None else imshow_kwargs
            kw['origin'] = 'lower'
            kw['extent'] = self.extent
            ax.imshow(eval(fill), **kw)

        # Draw the contours. The azimuthal angles are drawn on individually to
        # avoid having overlapping lines about the +\- pi boundary making a
        # particularly thick line.

        kw = {} if contour_kwargs is None else contour_kwargs
        kw['levels'] = kw.pop('levels', np.arange(0.5 * self.bmaj,
                                                  0.99 * r_max,
                                                  0.5 * self.bmaj))
        kw['levels'] = np.append(kw['levels'], 0.99 * r_max)
        kw['linewidths'] = kw.pop('linewidths', 1.0)
        kw['colors'] = kw.pop('colors', 'k')
        ax.contour(self.xaxis, self.yaxis, rvals, **kw)

        kw['levels'] = [0.0]
        for t in np.arange(-np.pi, np.pi, np.pi / 8.0):
            if t - 0.1 < -np.pi:
                a = np.where(abs(tvals - t) <= 0.1,
                             tvals - t, np.nan)
                b = np.where(abs(tvals - 2.0 * np.pi - t) <= 0.1,
                             tvals - 2.0 * np.pi - t, np.nan)
                amask = np.where(np.isfinite(a), 1, -1)
                bmask = np.where(np.isfinite(b), 1, -1)
                a = np.where(np.isfinite(a), a, 0.0)
                b = np.where(np.isfinite(b), b, 0.0)
                ttmp = np.where(amask * bmask < 1, a + b, np.nan)
            elif t + 0.1 > np.pi:
                a = np.where(abs(tvals - t) <= 0.1,
                             tvals - t, np.nan)
                b = np.where(abs(tvals + 2.0 * np.pi - t) <= 0.1,
                             tvals + 2.0 * np.pi - t, np.nan)
                amask = np.where(np.isfinite(a), 1, -1)
                bmask = np.where(np.isfinite(b), 1, -1)
                a = np.where(np.isfinite(a), a, 0.0)
                b = np.where(np.isfinite(b), b, 0.0)
                ttmp = np.where(amask * bmask < 1, a + b, np.nan)
            else:
                ttmp = np.where(abs(tvals - t) <= 0.1, tvals - t, np.nan)
            ax.contour(self.xaxis, self.yaxis, ttmp, **kw)
        ax.set_xlim(max(ax.get_xlim()), min(ax.get_xlim()))
        ax.set_aspect(1)

        if return_fig:
            return fig

    def plot_maximum(self, ax=None, imshow_kwargs=None):
        """
        Plot the maximum along the spectral axis.

        Args:
            ax (Optional[matplotlib axis instance]): Axis to use for plotting.
            imshow_kwargs (Optional[dict]): Kwargs to pass to imshow.

        Return:
            The axis with the maximum plotted.
        """
        from matplotlib.ticker import MultipleLocator
        ax = plt.subplots()[1] if ax is None else ax
        kw = {} if imshow_kwargs is None else imshow_kwargs
        kw['origin'] = 'lower'
        kw['extent'] = self.extent
        im = ax.imshow(np.nanmax(self.data, axis=0), **kw)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('Peak Intensity', rotation=270, labelpad=13)
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        return ax

    def plot_mask(self, ax, r_min=None, r_max=None, exclude_r=False,
                  PA_min=None, PA_max=None, exclude_PA=False, abs_PA=False,
                  mask_frame='disk', mask=None, x0=0.0, y0=0.0, inc=0.0,
                  PA=0.0, z0=None, psi=None, r_cavity=None, r_taper=None,
                  q_taper=None, z1=None, phi=None, z_func=None,
                  mask_color='k', mask_alpha=0.5, contour_kwargs=None,
                  contourf_kwargs=None, shadowed=False):
        """
        Plot the boolean mask on the provided axis to check that it makes
        sense.

        Args:
            ax (matplotib axis instance): Axis to plot the mask.
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z1 (Optional[float]): Correction to emission height at 1" in
                [arcsec].
            phi (Optional[float]): Flaring angle correction term.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mask_color (Optional[str]): Color used for the mask lines.
            mask_alpha (Optional[float]): The alpha value of the filled contour
                of the masked regions. Setting ``mask_alpha=0.0`` will remove
                the filling.

            contour_kwargs (Optional[dict]): Kwargs to pass to contour for
                drawing the mask.

        Returns:
            ax : The matplotlib axis instance.
        """
        # Grab the mask.
        if mask is None:
            mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=exclude_r,
                                 PA_min=PA_min, PA_max=PA_max,
                                 exclude_PA=exclude_PA, abs_PA=abs_PA,
                                 mask_frame=mask_frame, x0=x0, y0=y0, inc=inc,
                                 PA=PA, z0=z0, psi=psi, z1=z1, phi=phi,
                                 r_cavity=r_cavity, r_taper=r_taper,
                                 q_taper=q_taper, z_func=z_func,
                                 shadowed=shadowed)
        assert mask.shape[0] == self.yaxis.size, "Wrong y-axis shape for mask."
        assert mask.shape[1] == self.xaxis.size, "Wrong x-axis shape for mask."

        # Set the default plotting style.
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', mask_color)
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 1.0)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '-')
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        contourf_kwargs['alpha'] = contourf_kwargs.pop('alpha', mask_alpha)
        contourf_kwargs['colors'] = contourf_kwargs.pop('colors', mask_color)

        # Plot the contour and return the figure.
        ax.contourf(self.xaxis, self.yaxis, mask, [-.5, .5], **contourf_kwargs)
        ax.contour(self.xaxis, self.yaxis, mask, 1, **contour_kwargs)
