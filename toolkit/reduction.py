from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.utils.console import ProgressBar
from astropy.modeling import models, fitting
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, centroid_com, centroid_1dg

from .star_selection import init_centroids
from .photometry_results import PhotometryResults

__all__ = ['photometry']


def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]//binning_factor, a.shape[1]//binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(sh).sum(-1).sum(1)


def photometry(image_paths, master_dark_path, master_flat_path, star_positions,
               aperture_radii, centroid_stamp_half_width, psf_stddev_init,
               aperture_annulus_radius, output_path):
    """
    Parameters
    ----------
    master_dark_path : str
        Path to master dark frame
    master_flat_path :str
        Path to master flat field
    target_centroid : `~numpy.ndarray`
        position of centroid, with shape (2, 1)
    comparison_flux_threshold : float
        Minimum fraction of the target star flux required to accept for a
        comparison star to be included
    aperture_radii : `~numpy.ndarray`
        Range of aperture radii to use
    centroid_stamp_half_width : int
        Centroiding is done within image stamps centered on the stars. This
        parameter sets the half-width of the image stamps.
    psf_stddev_init : float
        Initial guess for the width of the PSF stddev parameter, used for
        fitting 2D Gaussian kernels to the target star's PSF.
    aperture_annulus_radius : int
        For each aperture in ``aperture_radii``, measure the background in an
        annulus ``aperture_annulus_radius`` pixels bigger than the aperture
        radius
    output_path : str
        Path to where outputs will be saved.
    """
    master_dark = fits.getdata(master_dark_path)
    master_flat = fits.getdata(master_flat_path)

    master_flat[(master_flat < 0.1) | np.isnan(master_flat)] = 1.0 # tmp

    target_centroid = star_positions#[:, 0]
    # star_positions = init_centroids(image_paths[0:3], master_flat, master_dark,
    #                                 target_centroid, plots=True).T #,
    #                                 #min_flux=comparison_flux_threshold).T

    # Initialize some empty arrays to fill with data:
    times = np.zeros(len(image_paths))
    fluxes = np.zeros((len(image_paths), len(star_positions),
                       len(aperture_radii)))
    errors = np.zeros((len(image_paths), len(star_positions),
                       len(aperture_radii)))
    xcentroids = np.zeros((len(image_paths), len(star_positions)))
    ycentroids = np.zeros((len(image_paths), len(star_positions)))
    airmass = np.zeros(len(image_paths))
    airpress = np.zeros(len(image_paths))
    humidity = np.zeros(len(image_paths))
    telfocus = np.zeros(len(image_paths))
    psf_stddev = np.zeros(len(image_paths))

    medians = np.zeros(len(image_paths))

    with ProgressBar(len(image_paths)) as bar:
        for i in range(len(image_paths)):
            bar.update()

            # Subtract image by the dark frame, normalize by flat field
            imagedata = (fits.getdata(image_paths[i]) - master_dark) / master_flat

            # Collect information from the header
            imageheader = fits.getheader(image_paths[i])
            exposure_duration = imageheader['EXPTIME']
            times[i] = Time(imageheader['DATE-OBS'], format='isot').jd # scale=imageheader['TIMESYS'].lower()
            medians[i] = np.median(imagedata)
            airmass[i] = imageheader['AIRMASS']
            # airpress[i] = imageheader['AIRPRESS']
            # humidity[i] = imageheader['HUMIDITY']
            # telfocus[i] = imageheader['TELFOCUS']

            # Initial guess for each stellar centroid informed by previous centroid
            for j in range(len(star_positions)):
                if i == 0:
                    init_x = star_positions[j][0]
                    init_y = star_positions[j][1]
                else:
                    init_x = ycentroids[i-1][j]
                    init_y = xcentroids[i-1][j]

                # # override for first 2019 observations
                # init_x = star_positions[j][0]
                # init_y = star_positions[j][1]

                # Cut out a stamp of the full image centered on the star
                image_stamp = imagedata[int(init_y) - centroid_stamp_half_width:
                                        int(init_y) + centroid_stamp_half_width,
                                        int(init_x) - centroid_stamp_half_width:
                                        int(init_x) + centroid_stamp_half_width]
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(image_stamp)
                # plt.show()

                # Measure stellar centroid with 2D gaussian fit
                x_stamp_centroid, y_stamp_centroid = centroid_com(image_stamp - np.nanmedian(image_stamp))
                # if (x_stamp_centroid < 0 or y_stamp_centroid < 0 or
                #         x_stamp_centroid > 2*centroid_stamp_half_width or
                #         y_stamp_centroid > 2*centroid_stamp_half_width):
                #     x_stamp_centroid = centroid_stamp_half_width
                #     y_stamp_centroid = centroid_stamp_half_width
                y_centroid = x_stamp_centroid + init_x - centroid_stamp_half_width
                x_centroid = y_stamp_centroid + init_y - centroid_stamp_half_width

                xcentroids[i, j] = x_centroid
                ycentroids[i, j] = y_centroid

                # For the target star, measure PSF:
                if j == 0:
                    psf_model_init = models.Gaussian2D(amplitude=np.max(image_stamp),
                                                       x_mean=centroid_stamp_half_width,
                                                       y_mean=centroid_stamp_half_width,
                                                       x_stddev=psf_stddev_init,
                                                       y_stddev=psf_stddev_init)

                    fit_p = fitting.LevMarLSQFitter()
                    y, x = np.mgrid[:image_stamp.shape[0], :image_stamp.shape[1]]
                    best_psf_model = fit_p(psf_model_init, x, y, image_stamp -
                                           np.median(image_stamp))
                    psf_stddev[i] = 0.5*(best_psf_model.x_stddev.value +
                                          best_psf_model.y_stddev.value)

            positions = np.vstack([ycentroids[i, :], xcentroids[i, :]])

            for k, aperture_radius in enumerate(aperture_radii):
                target_apertures = CircularAperture(positions, aperture_radius)

                flux = aperture_photometry(imagedata,
                                           target_apertures)['aperture_sum'].data
                # print(flux)
                # import matplotlib.pyplot as plt
                # plt.imshow(imagedata)
                # target_apertures.plot(color='w')
                # plt.show()

                background_subtracted_flux = flux

                fluxes[i, :, k] = background_subtracted_flux/exposure_duration
                errors[i, :, k] = np.sqrt(flux)

    ## Save some values
    results = PhotometryResults(times, fluxes, errors, xcentroids, ycentroids,
                                airmass, airpress, humidity, medians,
                                psf_stddev, aperture_radii)
    results.save(output_path)
    return results
