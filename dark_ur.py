import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import mad_std

from toolkit import (photometry, transit_model_b,
                     PhotometryResults, PCA_light_curve, params_b)
from astropy.io import fits

image_paths = sorted(glob('/Users/brettmorris/data/rem/20190829/IMG*UR*.fits'))[:-3]

images = []

for p in image_paths:
    images.append(fits.getdata(p))

fits.writeto('outputs/masterdark_ur.fits', np.nanmedian(images, axis=0), overwrite=True)