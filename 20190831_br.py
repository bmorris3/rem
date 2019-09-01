import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import mad_std

from toolkit import (photometry, transit_model_b,
                     PhotometryResults, PCA_light_curve, params_b)

# Image paths
image_paths = sorted(glob('/Users/brettmorris/data/rem/20190831/IMG*BR*.fits'))[:-3]
master_flat_path = 'outputs/master_flat_s_201906_BR_r_1s_norm.fits'
master_dark_path = 'outputs/masterdark.fits'

# dark = np.zeros_like(fits.getdata(image_paths[0]))
# fits.writeto(master_dark_path, dark)

# Photometry settings
aperture_radii = np.arange(15, 30)
centroid_stamp_half_width = 10
psf_stddev_init = 2
aperture_annulus_radius = 30
star_positions = [[458, 531],
                  [708, 519]]
output_path = 'outputs/20190831br.npz'
force_recompute_photometry = False

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

stds = []
lcs = []
for ap in range(phot_results.fluxes.shape[2]):
    regressors = np.vstack([phot_results.fluxes[:, 1, ap],
                            phot_results.xcentroids[:, 0] - phot_results.xcentroids[:, 0].mean(),
                            phot_results.ycentroids[:, 0] - phot_results.ycentroids[:, 0].mean(),
                            phot_results.airmass,
                            phot_results.background_median]).T

    target_lc = phot_results.fluxes[:, 0, ap]

    y = np.linalg.lstsq(regressors, target_lc, rcond=None)[0]
    comp_lc = regressors @ y
    lc = target_lc/comp_lc
    stds.append(mad_std(lc))
    lcs.append(lc)

best_ap = np.argmin(stds)
best_lc = lcs[best_ap]

np.save('outputs/20190831_r.npy', best_lc)

fig, ax = plt.subplots(4, 1, figsize=(10, 5))
ax[0].plot(phot_results.times, best_lc, '.')
ax[1].set_ylabel('X')
ax[1].plot(phot_results.times, phot_results.xcentroids[:, 0], '.')
ax[1].plot(phot_results.times, phot_results.xcentroids[:, 1], '.')

ax[2].set_ylabel('Y')
ax[2].plot(phot_results.times, phot_results.ycentroids[:, 0], '.')
ax[2].plot(phot_results.times, phot_results.ycentroids[:, 1], '.')

ax[3].set_ylabel('Flux')
ax[3].plot(phot_results.times, phot_results.fluxes[:, 0, best_ap], '.')
ax[3].plot(phot_results.times, phot_results.fluxes[:, 1, best_ap], '.')
# ax[3].plot(phot_results.times, phot_results.background_median, '.')
plt.show()
