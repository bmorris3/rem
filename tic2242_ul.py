import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import mad_std

from toolkit import (photometry, transit_model_b,
                     PhotometryResults, PCA_light_curve, params_b)

# Image paths
image_paths = sorted(glob('/Users/brettmorris/data/rem/20190829/IMG*UL*.fits'))
master_flat_path = 'outputs/master_flat_s_201906_UL_i_1s_norm.fits'
master_dark_path = 'outputs/masterdark.fits'

# dark = np.zeros_like(fits.getdata(image_paths[0]))
# fits.writeto(master_dark_path, dark)

# Photometry settings
aperture_radii = np.arange(15, 30)
centroid_stamp_half_width = 10
psf_stddev_init = 2
aperture_annulus_radius = 50
transit_parameters = params_b
star_positions = [[539, 665],
                  [268, 621]]
output_path = 'outputs/20190829ul.npz'
force_recompute_photometry = False #True

# Do photometry:

if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              star_positions,
                              aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path)

else:
    phot_results = PhotometryResults.load(output_path)

# print('Calculating PCA...')

lcs = (phot_results.fluxes[:, 0, :]/phot_results.fluxes[:, 1, :])
norm_lcs = lcs / np.median(lcs, axis=0)
min_std = np.argmin(mad_std(norm_lcs, axis=0))

# plt.plot(phot_results.times, lcs[:, min_std], '.')
# plt.show()


regressors = np.vstack([phot_results.fluxes[:, 1, min_std],
                        phot_results.xcentroids[:, 0] - phot_results.xcentroids[:, 0].mean(),
                        phot_results.ycentroids[:, 0] - phot_results.ycentroids[:, 0].mean(),
                        phot_results.airmass,
                        phot_results.background_median]).T

target_lc = phot_results.fluxes[:, 0, min_std]

y = np.linalg.lstsq(regressors, target_lc)[0]
comp_lc = regressors @ y

np.save('outputs/20190829_i.npy', target_lc / comp_lc)

fig, ax = plt.subplots(4, 1, figsize=(10, 5))
ax[0].plot(phot_results.times, target_lc / comp_lc, '.')
ax[1].set_ylabel('X')
ax[1].plot(phot_results.times, phot_results.xcentroids[:, 0], '.')
ax[1].plot(phot_results.times, phot_results.xcentroids[:, 1], '.')

ax[2].set_ylabel('Y')
ax[2].plot(phot_results.times, phot_results.ycentroids[:, 0], '.')
ax[2].plot(phot_results.times, phot_results.ycentroids[:, 1], '.')

ax[3].set_ylabel('Flux')
ax[3].plot(phot_results.times, phot_results.fluxes[:, 0, min_std], '.')
ax[3].plot(phot_results.times, phot_results.fluxes[:, 1, min_std], '.')
plt.show()

