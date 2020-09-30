# -*- coding: utf-8 -*-

""" MCCD UTILS

These functions  include several functions needed by the MCCD

"""

from __future__ import absolute_import, print_function
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import mccd_rca.utils as utils
from scipy.interpolate import Rbf
# import utils as utils
from shapepipe.pipeline import file_io
import galsim.hsm as hsm
from galsim import Image


def find_ccd_idx(ccd_id,ccd_list):
    idx = np.array([ccd_id == ccd_i for ccd_i in ccd_list])
    return np.argwhere(idx)[0][0]


class Loc2Glob(object):
    """        'COMMENT Unique detector IDs for MegaCam (North on top, East to the left)',
           'COMMENT    --------------------------',
           'COMMENT    ba ba ba ba ba ba ba ba ba',
           'COMMENT    00 01 02 03 04 05 06 07 08',
           'COMMENT --------------------------------',
           'COMMENT ba ba ba ba ba ba ba ba ba ba ba',
           'COMMENT 36 09 10 11 12 13 14 15 16 17 37',
           'COMMENT --------------*-----------------',
           'COMMENT 38 18 19 20 21 22 23 24 25 26 39',
           'COMMENT ab ab ab ab ab ab ab ab ab ab ab',
           'COMMENT --------------------------------',
           'COMMENT    27 28 29 30 31 32 33 34 35',
           'COMMENT    ab ab ab ab ab ab ab ab ab',
           'COMMENT    __________________________'
    """

    def __init__(self, x_gap=70, y_gap=425, x_npix=2048, y_npix=4612):
        self.x_gap = x_gap
        self.y_gap = y_gap
        self.x_npix = x_npix
        self.y_npix = y_npix

    def loc2glob_img_coord(self, ccd_n, x_coor, y_coor):
        """ Go from the local to the global img (pixel) coordinate system.
        Global system with (0,0) in the intersection of ccds [12,13,21,22]
        """

        # Flip axes
        x_coor, y_coor = self.flip_coord(ccd_n, x_coor, y_coor)

        # Calculate the shift
        x_shift, y_shift = self.shift_coord(ccd_n)

        # Return new coordinates
        return x_coor + x_shift, y_coor + y_shift

    def flip_coord(self, ccd_n, x_coor, y_coor):
        """ So the origin is on the south-west corner.
        Positive: South to North ; West to East
        FIXED! """

        if ccd_n < 18 or ccd_n in [36, 37]:
             x_coor = self.x_npix - x_coor + 1
             y_coor = self.y_npix - y_coor + 1
        else:
             pass

        return x_coor, y_coor

    def shift_coord(self, ccd_n):
        """ Coordinate shift: from local to global.
        """
        if ccd_n < 9:
            # first row
            x_shift = (ccd_n - 4) * (self.x_gap + self.x_npix)
            y_shift = self.y_gap + self.y_npix
            return x_shift, y_shift

        elif ccd_n < 18:
            # second row, non-ears
            x_shift = (ccd_n - 13) * (self.x_gap + self.x_npix)
            y_shift = 0.
            return x_shift, y_shift

        elif ccd_n < 27:
            # third row non-ears
            x_shift = (ccd_n - 22) * (self.x_gap + self.x_npix)
            y_shift = -1. * (self.y_gap + self.y_npix)
            return x_shift, y_shift

        elif ccd_n < 36:
            # fourth row
            x_shift = (ccd_n - 31) * (self.x_gap + self.x_npix)
            y_shift = -2. * (self.y_gap + self.y_npix)
            return x_shift, y_shift

        elif ccd_n < 37:
            # ccd= 36 ears, second row
            x_shift = (-5.) * (self.x_gap + self.x_npix)
            y_shift = 0.
            return x_shift, y_shift

        elif ccd_n < 38:
            # ccd= 37 ears, second row
            x_shift = (5.) * (self.x_gap + self.x_npix)
            y_shift = 0.
            return x_shift, y_shift

        elif ccd_n < 39:
            # ccd= 38 ears, third row
            x_shift = (-5.) * (self.x_gap + self.x_npix)
            y_shift = -1. * (self.y_gap + self.y_npix)
            return x_shift, y_shift

        elif ccd_n < 40:
            # ccd= 39 ears, third row
            x_shift = (5.) * (self.x_gap + self.x_npix)
            y_shift = -1. * (self.y_gap + self.y_npix)
            return x_shift, y_shift

class MccdInputs(object):
    """ Handle inputs for the multi-ccd algorithm.
    """

    def __init__(self, separator='-', coord_x_descriptor='XWIN_IMAGE', coord_y_descriptor='YWIN_IMAGE',
                 mask_thresh=-1e5, save_masks = True):
        self.separator = separator
        self.coord_x_descriptor = coord_x_descriptor
        self.coord_y_descriptor = coord_y_descriptor
        self.mask_thresh = mask_thresh
        self.loc2glob = Loc2Glob()

        self.save_masks = save_masks

        self.SNR_list = None
        self.star_list = None
        self.position_list = None
        self.mask_list = None
        self.ccd_list = None
        self.RA_list = None
        self.DEC_list = None

        self.num_catalogs = None
        self.catalog_ids = None
        self.starcat_list = None

    def handle_mask(self, stars, thresh, apply_to_stars=False):
        """ Reads SExtracted star stamps, generates RCA-compatible masks (that is, binary weights),
        and replaces bad pixels with 0s - they will not be used by RCA, but the ridiculous numerical
        values can otherwise still lead to problems because of convolutions."""
        mask = np.ones(stars.shape)
        mask[stars < thresh] = 0
        if apply_to_stars:
            stars[stars < thresh] = 0
        return mask

    def parse_path(self, path):
        """ Parse one path and extract info.
        """
        # Remove the .fits
        my_path = path.split('.')[0]

        # split the path to get the info
        splitted_str = my_path.split(self.separator)

        # Extract info
        ccd_n = splitted_str[-1]
        starcat_id = splitted_str[-2]

        return starcat_id, ccd_n

    def parse_folder(self, folder_path, pattern):
        """ Parse a folder. Separate into a list of np.array each one for each
        starcat_id present in the folder.
        Each np.array in the list contains (starcat_id, ccd_n, path)
        """
        file_paths = glob.glob(folder_path + pattern)
        file_paths.sort()

        complete_list = np.array([[self.parse_path(path)[0], self.parse_path(path)[1], path]
                                  for path in file_paths])

        starcat_unique_ids = np.unique(complete_list[:, 0])

        # Create  list for each star catalog including all the CCDs
        starcat_list = []

        for it in range(starcat_unique_ids.shape[0]):
            st_id = starcat_unique_ids[it]

            indexes = (complete_list[:, 0] == str(st_id))
            starcat_list.append(complete_list[indexes, :])

        # Save the ids
        self.num_catalogs = len(starcat_list)
        self.catalog_ids = starcat_unique_ids

        # Save the complete list
        self.starcat_list = starcat_list

    def parse_pipeline_input_list(self,input_list,element_position):
        """ Parse a pipeline input file list. Separate into a list of np.array
        for each starcat_id present in the folder.
        Each np.array in the list contains (starcat_id, ccd_n, path)
        If there are multiple elements, ex: train/test the element_position
        determines which one will be used.
        """
        complete_list = np.array([[self.parse_path(path[element_position])[0],
                                   self.parse_path(path[element_position])[1],
                                   path[element_position]]
                                   for path in input_list])
        starcat_unique_ids = np.unique(complete_list[:, 0])

        # Create  list for each star catalog including all the CCDs
        starcat_list = []

        for it in range(starcat_unique_ids.shape[0]):
            st_id = starcat_unique_ids[it]

            indexes = (complete_list[:, 0] == str(st_id))
            starcat_list.append(complete_list[indexes, :])

        # Save the ids
        self.num_catalogs = len(starcat_list)
        self.catalog_ids = starcat_unique_ids

        # Save the complete list
        self.starcat_list = starcat_list

    def prep_mccd_inputs(self, starcat_array):
        """ Prepare the inputs for mccd algorithm:
        - Correct the coordinates from local img coordinates into the global coordinate.
        - Apply mask to stars
        - Normalize star values.
        - Modify the star format.

            Input: np.array with (starcat_id, ccd_n, path) for every file in one starcat_id.
            Output: Three lists [stars],[pos],[mask]
        """

        number_ccd = starcat_array.shape[0]

        star_list = []
        position_list = []
        mask_list = []
        ccd_list = []
        SNR_list = []
        RA_list = []
        DEC_list = []

        for it in range(number_ccd):
            starcat = fits.open(starcat_array[it, 2])
            ccd = starcat_array[it, 1].astype('int')

            positions = np.array([self.loc2glob.loc2glob_img_coord(ccd, x, y) for x, y \
                                  in zip(starcat[2].data[self.coord_x_descriptor],
                                         starcat[2].data[self.coord_y_descriptor])])

            stars = utils.rca_format(starcat[2].data['VIGNET'])
            masks = self.handle_mask(stars, thresh=self.mask_thresh,
                                     apply_to_stars=self.save_masks)

            star_list.append(stars)
            position_list.append(positions)
            mask_list.append(masks)
            ccd_list.append(ccd)
            try:
                SNR = starcat[2].data['SNR_WIN']
                SNR_list.append(SNR)
            except Exception:
                SNR_list = None
            try:
                RA_list.append(starcat[2].data['XWIN_WORLD'])
                DEC_list.append(starcat[2].data['YWIN_WORLD'])
            except Exception:
                RA_list = None
                DEC_list = None


        self.SNR_list = SNR_list
        self.star_list = star_list
        self.position_list = position_list
        self.mask_list = mask_list
        self.ccd_list = ccd_list
        self.RA_list = RA_list
        self.DEC_list = DEC_list

        return star_list, position_list, mask_list, ccd_list, SNR_list, RA_list, DEC_list

    def preprocess_data(self, folder_path, pattern):
        """ Parse folder and return the list of ids.
        """
        # Parser folder
        self.parse_folder(folder_path, pattern)

        return self.catalog_ids

    def proprocess_pipeline_data(self,input_list,element_position):
        """ Preprocess pipeline's input file list and return the list of ids.
        """
        # Parser pipeline input file list
        self.parse_pipeline_input_list(input_list,element_position)

        return self.catalog_ids

    def get_inputs(self, catalog_id):
        """ Input: catalog_ig (int) to be processed
            Output: Three lists [stars],[pos],[mask] fo the mccd algortithm input
        """
        print('Extracting catalog_id %s ..' % (catalog_id))

        # Look for the catalog_id in the list
        idx = np.array([catalog_id == starcat[0, 0] for starcat in self.starcat_list])
        idx = np.argwhere(idx)[0][0]

        # Return the inputs
        return self.prep_mccd_inputs(self.starcat_list[idx])

    def plot_ccd_positions(self):
        """ Plot the star positions on the global coordinates for all CCDs.
        """
        try:
            plt.figure()
            for it in range(len(self.position_list)):
                plt.scatter(self.position_list[it][:, 0], self.position_list[it][:, 1], s=2)
            plt.show()
        except:
            print('Try first to calculate the positions. Call get_inputs()..')

    def outlier_rejection(self, star_list, pos_list, mask_list, ccd_list,
        SNR_list=None, RA_list=None, DEC_list=None, shape_std_max=3., print_fun = None ):
        # for the moment based on the e1,e2,R2 value

        if print_fun is None:
            print_fun = lambda x : print(x)

        # Reject outliers
        all_stars = np.concatenate(star_list, axis=2)
        all_stars = utils.reg_format(np.copy(all_stars))
        all_masks = np.concatenate(mask_list, axis=2)
        all_masks = utils.reg_format(np.copy(all_masks))
        badpix_masks = np.rint(np.abs(all_masks - 1))  # hsm thinks 0 means good

        star_moms = [hsm.FindAdaptiveMom(Image(star), badpix=Image(bp), strict=False)
                     for star, bp in zip(all_stars, badpix_masks)]
        star_shapes = np.array([[moms.observed_shape.g1,
                                 moms.observed_shape.g2,
                                 2. * moms.moments_sigma ** 2,
                                 int(bool(moms.error_message))]
                                for moms in star_moms])

        # Outlier rejection based on e1, e2 and R2
        R2_thresh = shape_std_max * np.std(star_shapes[:, 2]) + np.mean(star_shapes[:, 2])
        R2_bad_stars = (abs(star_shapes[:, 2]) > R2_thresh)

        e2_thresh = shape_std_max * np.std(star_shapes[:, 1]) + np.mean(star_shapes[:, 1])
        e2_bad_stars = (abs(star_shapes[:, 1]) > e2_thresh)

        e1_thresh = shape_std_max * np.std(star_shapes[:, 0]) + np.mean(star_shapes[:, 0])
        e1_bad_stars = (abs(star_shapes[:, 0]) > e1_thresh)

        bad_stars = np.logical_or(e1_bad_stars, e2_bad_stars)
        bad_stars = np.logical_or(bad_stars, R2_bad_stars)

        bad_stars_idx = np.nonzero(bad_stars)[0]
        print_fun(bad_stars_idx)
        print_fun(bad_stars_idx.shape)

        # create masks
        erase_masks = [np.zeros(star_list[i].shape[2], dtype=bool) for i in range(len(star_list))]

        if bad_stars_idx.size > 0:

            # We have to erase the outliers
            # Create the reference ids (to match the global array to the list of arrays)
            idx_ref = np.zeros((len(all_stars), 3), dtype=int)  # glob_id, star_id, ccd_id
            glob_id, star_id, ccd_id = 0, 0, 0
            for stars in star_list:
                star_id = 0
                for star in utils.reg_format(stars):
                    idx_ref[glob_id, 0], idx_ref[glob_id, 1], idx_ref[glob_id, 2] = glob_id, star_id, ccd_id
                    glob_id += 1
                    star_id += 1
                ccd_id += 1

            # select outlier stars
            for bad_id in bad_stars_idx:
                print_fun('Outlier: Glob_id=%d , star_id=%d , ccd_id=%d' % (
                    idx_ref[bad_id, 0], idx_ref[bad_id, 1], idx_ref[bad_id, 2]))
                erase_masks[idx_ref[bad_id, 2]][idx_ref[bad_id, 1]] = True

            for it_star in range(len(star_list)):
                mask = ~erase_masks[it_star]
                star_list[it_star] = star_list[it_star][:, :, mask]  # erase star and overwrite
                mask_list[it_star] = mask_list[it_star][:, :, mask]  # erase mask and overwrite
                pos_list[it_star] = pos_list[it_star][mask, :]  # erase position and overwrite
                if SNR_list is not None:
                    SNR_list[it_star] = SNR_list[it_star][mask]
                if RA_list is not None:
                    RA_list[it_star] = RA_list[it_star][mask]
                    DEC_list[it_star] = DEC_list[it_star][mask]

        return star_list, pos_list, mask_list, ccd_list, SNR_list, RA_list, DEC_list, erase_masks


def random_indexes(n_tot,train_per=0.8,min_n_train=20):
    """ Generate random indexes to separate train and test sets.
    """
    # Define number of stars
    train_star_nb = np.max([np.floor(train_per*n_tot).astype(int),min_n_train])
    test_star_nb = n_tot - train_star_nb

    # Generate the random test positions in the star field
    # The positions will be maintained throughout the star sets
    rand_seq = np.random.randn(n_tot).argsort()
    train_idx = rand_seq[0:train_star_nb]
    test_idx = rand_seq[train_star_nb:]

    return train_idx, test_idx

def save_fits(dictionary, train_bool, cat_id, output_path,example_path = None):
    """ fits saving function
    """
    if example_path is None:
        example_path = '/Users/tliaudat/Documents/PhD/codes/venv_p3/JB-data/CFIS-data/all_w3_star_cat/star_selection-2079614-13.fits'
    # Save data into the FITS format extension
    train_pattern = 'train_star_selection'
    test_pattern = 'test_star_selection'
    try:
        number_scheme = "-%07d"%(cat_id)
    except:
        number_scheme = '-' + cat_id
    ext = '.fits'

    if train_bool == True:
        saving_path = output_path + train_pattern + number_scheme + ext
    elif train_bool == False:
        saving_path = output_path + test_pattern + number_scheme + ext

    fits_file = file_io.FITSCatalog(saving_path,\
        open_mode = file_io.BaseCatalog.OpenMode.ReadWrite, SEx_catalog=True)
    fits_file.save_as_fits(dictionary, \
        sex_cat_path = example_path)

def return_loc_neighbors(new_pos, obs_pos, vals, n_neighbors):
    """ Find the local ''n_neighbors'' nearest neighbors locally in one ccd."""
    distances = np.linalg.norm(obs_pos-new_pos, axis=1)
    nbs = vals[np.argsort(distances)[:n_neighbors]]
    pos = obs_pos[np.argsort(distances)[:n_neighbors]]
    return nbs, pos

def return_glob_neighbors(new_pos, obs_pos_list, val_list, n_neighbors):
    """ Find the global ''n_neighbors'' nearest neighbors in all the available ccds.
    """
    num_ccds = len(obs_pos_list)

    # Calculate all the distances
    dist_list = [np.array([np.linalg.norm(obs_pos-new_pos, axis=1),(np.ones(obs_pos.shape[0])*ccd_n),
                           np.arange(obs_pos.shape[0])])
                 for obs_pos,ccd_n in zip(obs_pos_list,np.arange(len(obs_pos_list)))]
    # Sort the distances
    distances = np.concatenate(dist_list,axis=1)
    sort_idxs = np.argsort(distances[0,:])[:n_neighbors]

    # Extract values
    ccd_idxs = distances[1,sort_idxs].astype(int)
    inter_ccd_idxs = distances[2,sort_idxs].astype(int)

    values = np.array([val_list[ccd_idxs[it]].T[inter_ccd_idxs[it],:] for it in range(n_neighbors)])
    positions = np.array([obs_pos_list[ccd_idxs[it]][inter_ccd_idxs[it],:] for it in range(n_neighbors)])

    return values, positions

class MomentInterpolator(object):
    """ Allows to interpolate moments from a bin image like the one
    from MeanShapes function.
    """

    def __init__(self, moment_map, n_neighbors = 1000, rbf_function = 'thin_plate'):
        # Save variables
        self.loc2glob = Loc2Glob()
        self.n_neighbors = n_neighbors
        self.rbf_function = rbf_function
        self.moment_map = np.zeros(moment_map.shape)
        self.x_pix = self.loc2glob.x_npix
        self.y_pix = self.loc2glob.y_npix

        # Define parameters
        self.n_ccd = moment_map.shape[0] # 40
        self.x_grid = moment_map.shape[1] # 20
        self.y_grid = moment_map.shape[2] # 40
        self.bin_x = self.x_pix/self.x_grid
        self.bin_y = self.y_pix/self.y_grid

        # Correct MegaCam origin conventions
        for ccd_it in range(self.n_ccd):
            for it_x in range(self.x_grid):
                for it_y in range(self.y_grid):

                    if ccd_it < 18 or ccd_it in [36,37]:
                        # swap x axis so origin is on top-right
                        x = it_x
                        y = it_y

                    else:
                        # swap y axis so origin is on bottom-left
                        x = self.x_grid - it_x - 1
                        y = self.y_grid - it_y - 1

                    self.moment_map[ccd_it,x,y] = moment_map[ccd_it,it_x,it_y]


        # Generate local generic grid
        x_lin = np.linspace(start = self.bin_x/2, stop = self.x_pix - self.bin_x/2, num=self.x_grid)
        y_lin = np.linspace(start = self.bin_y/2, stop = self.y_pix - self.bin_y/2, num=self.y_grid)
        xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
        self.xv = xv
        self.yv = yv

        # Generate global positions for the bins
        self.x_pos = np.zeros(moment_map.shape)
        self.y_pos = np.zeros(moment_map.shape)
        for ccd_it in range(self.n_ccd):
            x_glob , y_glob = self.loc2glob.loc2glob_img_coord(ccd_n = ccd_it,
                                    x_coor=np.copy(self.xv.flatten()), y_coor=np.copy(self.yv.flatten()))

            self.x_pos[ccd_it,:,:] = x_glob.reshape(self.x_grid,self.y_grid)
            self.y_pos[ccd_it,:,:] = y_glob.reshape(self.x_grid,self.y_grid)


    def interpolate_position(self,target_x, target_y):
        # Calculate distances
        res_x = self.x_pos.flatten() - target_x
        res_y = self.y_pos.flatten() - target_y
        dist = np.sqrt(res_x**2 + res_y**2)

        # Select bins to use. The n_neighbors closest positions
        sort_idxs = np.argsort(dist)[:self.n_neighbors]

        # Extract values
        x_pos_interp = self.x_pos.flatten()[sort_idxs]
        y_pos_interp = self.y_pos.flatten()[sort_idxs]
        val_interp = self.moment_map.flatten()[sort_idxs]

        # Generate the interpolation function
        rbf = Rbf(x_pos_interp, y_pos_interp, val_interp, function=self.rbf_function)
        output_val = rbf(target_x, target_y)

        return output_val

def interpolation_Pi(position_list, d_comp_glob):
    # Create a Pi matrix list that will be used for the interpolation of the global model

    n_comp_glob = (d_comp_glob + 1) * (d_comp_glob + 2) // 2

    interp_Pi = [utils.poly_pos(interp_pos, d_comp_glob, normalice=False, center=False)
                 for interp_pos in position_list]

    # Global position model
    # Normalization is not done on poly_pos() but globaly here
    sum_vals = np.zeros((n_comp_glob))
    for it in range(n_comp_glob):
        for it_ccd in range(len(position_list)):
            sum_vals[it]+= np.sum(interp_Pi[it_ccd][it,:]**2)
        sum_vals[it] = np.sqrt(sum_vals[it])
    interp_Pi = [interp_Pi[it]/sum_vals.reshape(-1,1) for it in range(len(interp_Pi))]
    # Normalize wrt to the constant polynome
    norm_val = interp_Pi[0][0,0]
    for it in range(len(interp_Pi)):
        interp_Pi[it] /= norm_val

    return interp_Pi
