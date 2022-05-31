#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import mccd


# Class definition
class Glob2CCD(object):
    r""" Get the CCD ID number from the global coordinate position.

    The Loc2Glob() object as input is the one that defines the instrument's
    geometry.

    Parameters
    ----------
    loc2glob: Loc2Glob object
        Object with the desired focal plane geometry.
    with_gaps: bool
        If add the gaps to the CCD area.
    """
    def __init__(self, loc2glob, with_gaps=True):
        # Save loc2glob object
        self.loc2glob = loc2glob
        self.with_gaps = with_gaps
        self.ccd_list = np.arange(self.loc2glob.ccd_tot)
        # Init edges defininf the CCDs
        self.edge_x_list, self.edge_y_list = self.build_all_edges()

    def build_all_edges(self):
        """ Build the edges for all the CCDs in the focal plane.
        """
        edge_x_list = np.array([
            self.build_edge(ccd_n)[0] for ccd_n in self.ccd_list
        ])
        edge_y_list = np.array([
            self.build_edge(ccd_n)[1] for ccd_n in self.ccd_list
        ])

        return edge_x_list, edge_y_list

    def build_edge(self, ccd_n):
        """ Build the edges of the `ccd_n` in global coordinates.
        """
        if self.with_gaps:
            corners = np.array([
                [
                    -self.loc2glob.x_gap / 2,
                    -self.loc2glob.y_gap / 2
                ],
                [
                    self.loc2glob.x_npix + self.loc2glob.x_gap / 2,
                    -self.loc2glob.y_gap / 2
                ],
                [
                    -self.loc2glob.x_gap / 2,
                    self.loc2glob.y_npix + self.loc2glob.y_gap / 2
                ],
                [
                    self.loc2glob.x_npix + self.loc2glob.x_gap / 2,
                    self.loc2glob.y_npix + self.loc2glob.y_gap / 2,
                ],
            ])
        else:
            corners = np.array([
                [0, 0],
                [self.loc2glob.x_npix, 0],
                [0, self.loc2glob.y_npix],
                [self.loc2glob.x_npix, self.loc2glob.y_npix],
            ])

        glob_corners = np.array([
            self.loc2glob.loc2glob_img_coord(ccd_n, pos[0], pos[1])
            for pos in corners
        ])

        edge_x = np.array([
            np.min(glob_corners[:, 0]), np.max(glob_corners[:, 0])
        ])
        edge_y = np.array([
            np.min(glob_corners[:, 1]), np.max(glob_corners[:, 1])
        ])

        return edge_x, edge_y

    def is_inside(self, x, y, edge_x, edge_y):
        """ Is the position inside the edges.

        Return True if the position is within the rectangle
        defined by the edges.

        Parameters
        ----------
        x: float
            Horizontal position in global coordinate system.
        y: float
            Vertical position in global coordinate system.
        edge_x: np.ndarray
            Edge defined as `np.array([min_x, max_x])`.
        edge_y: np.ndarray
            Edge defined as `np.array([min_y, max_y])`.
        """
        if (x > edge_x[0]) and (x < edge_x[1]) and \
           (y > edge_y[0]) and (y < edge_y[1]):
            return True
        else:
            return False

    def get_ccd_n(self, x, y):
        """ Returns the CCD number from the position `(x, y)`.

        Returns `None` if the position is not found.
        """
        bool_list = np.array([
            self.is_inside(x, y, edge_x, edge_y)
            for edge_x, edge_y in zip(self.edge_x_list, self.edge_y_list)
        ])

        try:
            return self.ccd_list[bool_list][0]
        except Exception:
            return None


# Starting the script
data_dir = '/n05data/tliaudat/CFIS_shapepipe_v1/data/'
original_cat = 'unions_shapepipe_psf_2022_v1.0.1.fits'
new_cat = 'unions_shapepipe_psf_2022_v1.0.2-0000000.fits'

starcat = fits.open(data_dir + original_cat, memmap=True)
print(starcat[1].columns)

# Build a mask to remove objects with problems
# These are objects with DEC coordinate equal to zero
mask_good_dec = ~(starcat[1].data['DEC'] == 0.)

loc2glob = mccd.mccd_utils.Loc2Glob()
glob2ccd = Glob2CCD(loc2glob)

ccd_list = np.array([
    glob2ccd.get_ccd_n(x, y) for x, y in zip(
        starcat[1].data['X'][mask_good_dec],
        starcat[1].data['Y'][mask_good_dec]
    )
])

# Collect columns
data = {
    'X': starcat[1].data['X'][mask_good_dec],
    'Y': starcat[1].data['Y'][mask_good_dec],
    'RA': starcat[1].data['RA'][mask_good_dec],
    'DEC': starcat[1].data['DEC'][mask_good_dec],
    'E1_PSF_HSM': starcat[1].data['E1_PSF_HSM'][mask_good_dec],
    'E2_PSF_HSM': starcat[1].data['E2_PSF_HSM'][mask_good_dec],
    'SIGMA_PSF_HSM': starcat[1].data['SIGMA_PSF_HSM'][mask_good_dec],
    'E1_STAR_HSM': starcat[1].data['E1_STAR_HSM'][mask_good_dec],
    'E2_STAR_HSM': starcat[1].data['E2_STAR_HSM'][mask_good_dec],
    'SIGMA_STAR_HSM': starcat[1].data['SIGMA_STAR_HSM'][mask_good_dec],
    'FLAG_PSF_HSM': starcat[1].data['FLAG_PSF_HSM'][mask_good_dec],
    'FLAG_STAR_HSM': starcat[1].data['FLAG_STAR_HSM'][mask_good_dec],
    'CCD_NB': ccd_list.astype(float),
}

save_path = data_dir + new_cat
mccd.mccd_utils.save_to_fits(data, save_path)
