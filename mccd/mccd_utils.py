# -*- coding: utf-8 -*-

r"""MCCD UTILS.

These functions include several functions needed by the MCCD.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>
            Samuel Farrens <samuel.farrens@cea.fr>

Notes
-----
I added two functions from Pysap astro plugin and from ModOpt.signal.wavelet
so that I could replicate the functionv ``get_mr_transform()`` from ModOpt
without the need of having sparse2d installed as an exectubale by using
Pysap python bindings of the aforementioned C++ package.
Thanks Samuel Farrens (main developer of ModOpt and Pysap) for the help.

"""

from __future__ import absolute_import, print_function
import os
import numpy as np
import glob
from astropy.io import fits
from scipy.interpolate import Rbf
import galsim.hsm as hsm
from galsim import Image
import mccd.utils as utils
from pysap import load_transform


def find_ccd_idx(ccd_id, ccd_list):
    r"""Find the index of an element in a list."""
    idx = np.array([ccd_id == ccd_i for ccd_i in ccd_list])
    return np.argwhere(idx)[0][0]


class Loc2Glob(object):
    r"""Change from local to global coordinates.

    Class to pass from local coordinates to global coordinates under
    CFIS (CFHT) MegaCam instrument. The geometrical informcation of the
    instrument is encoded in this function.

    Parameters
    ----------
    x_gap: int
        Gap between the CCDs on the horizontal direction.
        Default to ``70`` (CFIS value).
    y_gap: int
        Gap between the CCDs on the vertical direction.
        Default to ``425`` (CFIS value).
    x_npix: int
        Number of pixels on one CCD on the horizontal direction.
        Default to ``2048`` (CFIS value).
    y_npix: int
        Number of pixels on one CCD on the vertical direction.
        Default to ``4612`` (CFIS value).

    Notes
    -----
    MegaCams geometry. Watch out with the conventions ba,ab that means where
    is the local coordinate system origin for each CCD.
    For more info look MegaCam's instrument webpage.

    Examples
    --------
    >>> 'COMMENT (North on top, East to the left)',
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
        r"""Initialize with instrument geometry."""
        self.x_gap = x_gap
        self.y_gap = y_gap
        self.x_npix = x_npix
        self.y_npix = y_npix

    def loc2glob_img_coord(self, ccd_n, x_coor, y_coor):
        r"""Go from the local to the global img (pixel) coordinate system.

        Global system with (0,0) in the intersection of ccds [12,13,21,22].

        Parameters
        ----------
        ccd_n: int
            CCD number of the considered positions.
        x_coor: float
            Local coordinate system hotizontal value.
        y_coor: float
            Local coordinate system vertical value.

        Returns
        -------
        glob_x_coor: float
            Horizontal position in global coordinate system.
        glob_y_coor: float
            Vertical position in global coordinate system.

        """
        # Flip axes
        x_coor, y_coor = self.flip_coord(ccd_n, x_coor, y_coor)

        # Calculate the shift
        x_shift, y_shift = self.shift_coord(ccd_n)

        # Return new coordinates
        return x_coor + x_shift, y_coor + y_shift

    def flip_coord(self, ccd_n, x_coor, y_coor):
        r"""Change of coordinate convention.

        So that all of them are coherent on the global coordinate system.
        So that the origin is on the south-west corner.
        Positive: South to North ; West to East.
        """
        if ccd_n < 18 or ccd_n in [36, 37]:
            x_coor = self.x_npix - x_coor + 1
            y_coor = self.y_npix - y_coor + 1
        else:
            pass

        return x_coor, y_coor

    def shift_coord(self, ccd_n):
        r"""Provide the shifting.

        It is needed to go from the local coordinate
        system origin to the global coordinate system origin.
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
            x_shift = 5. * (self.x_gap + self.x_npix)
            y_shift = 0.
            return x_shift, y_shift

        elif ccd_n < 39:
            # ccd= 38 ears, third row
            x_shift = (-5.) * (self.x_gap + self.x_npix)
            y_shift = -1. * (self.y_gap + self.y_npix)
            return x_shift, y_shift

        elif ccd_n < 40:
            # ccd= 39 ears, third row
            x_shift = 5. * (self.x_gap + self.x_npix)
            y_shift = -1. * (self.y_gap + self.y_npix)
            return x_shift, y_shift


class MccdInputs(object):
    r"""Handle inputs for the MCCD algorithm.

    This method was set up to work with a catalog outputed by SExtractor but
    it should not be restricted to that.
    The input files are generally created one per exposure per CCD.
    Methods like PSFEx that build one model per exposure per CCD do not have
    any preprocessing to do.
    The MCCD method that uses all the stars in one exposure to build the
    model needs to have only one file that will contain all the information
    gathered in one exposure.
    This is the main reason of this class, to be able to handle the
    different CCD files from a particular exposure and integrate them in a
    single file following the MCCD conventions, ie changing the local
    coordinate system to the global one.
    The expected files are in fits format.
    An filename example would be: ``star_selection-1234567-04.fits``
        Where the exposure ID is: ``1234567``
        Where the CCD ID is: ``04``

    Parameters
    ----------
    separator: str
        String separating the file names. Between the file pattern,
        the exposure ID and the CCD ID.
        Default is ``'-'``.
    coord_x_descriptor: str
        Name of the fits column to be used as the horizontal
        position coordinate.
        Default is ``'XWIN_IMAGE'``.
    coord_y_descriptor: str
        Name of the fits column to be used as the vertical position coordinate.
        Default is ``'YWIN_IMAGE'``.
    mask_thresh: float
        Threshold to be used when constructing the masks.
        SExtractor has the particular way of masking pixels by setting
        pixels to the value -1e30.
        Values beneath the threshold will be considered masked.
        Default is ``-1e5``.
    """

    def __init__(self, separator='-', coord_x_descriptor='XWIN_IMAGE',
                 coord_y_descriptor='YWIN_IMAGE', mask_thresh=-1e5):
        r"""Initialize class attributes."""
        self.separator = separator
        self.coord_x_descriptor = coord_x_descriptor
        self.coord_y_descriptor = coord_y_descriptor
        self.mask_thresh = mask_thresh
        self.loc2glob = Loc2Glob()

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

    @staticmethod
    def handle_mask(stars, thresh, apply_to_stars=False):
        r"""Handle and generate masks.

        Reads SExtracted star stamps, generates MCCD-compatible masks
        (that is, binary weights), and replaces bad pixels with 0s -
        they will not be used by MCCD, but the ridiculous numerical
        values can otherwise still lead to problems because of convolutions.

        Parameters
        ----------
        stars: numpy.ndarray
            Stars to be masked.
        thresh: float
            Threshold that will define the mask.
            Values below the threhsold will be considered masked.
            Default is -1e5.
        apply_to_stars: bool
            Boolean to define if the stars should be thresholded
            with ``thresh`` on top of the mask construction.
        """
        mask = np.ones(stars.shape)
        mask[stars < thresh] = 0
        if apply_to_stars:
            stars[stars < thresh] = 0
        return mask

    def parse_path(self, path):
        r"""Parse one path and extract info.

        Parameters
        ----------
        path: str
            Path to the file to be parsed in order to extract the
            exposure ID and the CCD ID.

        """
        # Remove the .fits or the extension
        my_path = path.split('.')[-2]

        # split the path to get the info
        splitted_str = my_path.split(self.separator)

        # Extract info
        ccd_n = splitted_str[-1]
        starcat_id = splitted_str[-2]

        return starcat_id, ccd_n

    def parse_folder(self, folder_path, pattern):
        r"""Parse a folder that match a specfic ``pattern``.

        Separate into a list of np.array each one for each starcat_id
        present in the folder.
        Each np.array in the list contains (starcat_id, ccd_n, path).

        Parameters
        ----------
        folder_path: str
            Path to the folder to be preprocessed.
        pattern: str
            ``pattern`` to match on the files found in ``folder_path``.

        """
        file_paths = glob.glob(folder_path + pattern)
        file_paths.sort()

        complete_list = np.array(
            [[self.parse_path(path)[0], self.parse_path(path)[1], path]
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

    def parse_pipeline_input_list(self, input_list, element_position=0):
        r"""Parse a pipeline input file list.

        Separate into a list of np.array for each starcat_id
        present in the folder.

        Parameters
        ----------
        input_list: list of str
            List containing the path to all the files to be processed.
        element_position:
            The element to consider on the input list.
            If there are multiple elements that share the name pattern and
            the CCD ID, ie train/test, the element_position determines
            which one will be used.
            If there is only one element it should be 0.
            Default is 0.

        Notes
        -----
        The ``starcat_list`` that is saved contains in each
        element (starcat_id, ccd_n, path).

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
        r"""Prepare the inputs for mccd algorithm.

        Taks done:

        - Translate from local to global coordinate system.

        - Apply mask to stars.

        - Normalize star values.

        - Modify the star format.


        Parameters
        ----------
        starcat_array: numpy.ndarray
            Array with (starcat_id, ccd_n, path) for every file in one
            starcat_id (exposure ID).

        Returns
        -------
        star_list: list
            List containing the masked star stamps.
        position_list: list
            List containing the positions in the global coordinate system.
        mask_list: list
            List containing the masks corresponding to the star stamps.
        ccd_list: list
            List containing the CCD ids for the stars.
        SNR_list: list
            List containing the estimated SNR values for the stars.
            Will be None if there are no SNR values available.
        RA_list: list
            List containing the RA coordinate for the stars.
            Will be None if there are no RA coordinates available.
        DEC_list: list
            List containing the DEC coordinate for the stars.
            Will be None if there are no DEC coordinates available.

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

            positions = np.array(
                [self.loc2glob.loc2glob_img_coord(ccd, x, y) for x, y in
                 zip(starcat[2].data[self.coord_x_descriptor],
                     starcat[2].data[self.coord_y_descriptor])])

            stars = utils.rca_format(starcat[2].data['VIGNET'])
            masks = self.handle_mask(stars, thresh=self.mask_thresh,
                                     apply_to_stars=True)

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

        return star_list, position_list, mask_list, ccd_list, SNR_list,\
            RA_list, DEC_list

    def preprocess_data(self, folder_path, pattern):
        r"""Preprocess the data.

        Parameters
        ----------
        folder_path: str
            Path to the folder containing the data files.
        pattern: str
            Pattern to match with the files in ``folder_path``.

        Returns
        -------
        catalog_ids: list
            List of exposure IDs preprocessed.

        """
        # Parser folder
        self.parse_folder(folder_path, pattern)

        return self.catalog_ids

    def proprocess_pipeline_data(self, input_list, element_position):
        r"""Preprocess Shapepipe pipeline's input file list.

        Also return the list of ids.

        Parameters
        ----------
        input_list: list of str
            List containing the path to all the files to be processed.
        element_position:
            The element to consider on the input list.
            If there are multiple elements that share the name
            pattern and the CCD ID, ie train/test, the element_position
            determines which one will be used.
            If there is only one element it should be 0.
            Default is 0.

        Returns
        -------
        catalog_ids: list
            List of exposure IDs preprocessed.

        """
        # Parser pipeline input file list
        self.parse_pipeline_input_list(input_list, element_position)

        return self.catalog_ids

    def get_inputs(self, catalog_id):
        r"""Get the MCCD inputs from a specific exposure id.

        Returns several lists corresponding to the outputs of the
        function ``prep_mccd_inputs``.

        Parameters
        ----------
        catalog_id: int
            Catalog id (exposure id) to be processed.

        """
        print('Extracting catalog_id %s ..' % catalog_id)

        # Look for the catalog_id in the list
        idx = np.array(
            [catalog_id == starcat[0, 0] for starcat in self.starcat_list])
        idx = np.argwhere(idx)[0][0]

        # Return the inputs
        return self.prep_mccd_inputs(self.starcat_list[idx])

    def get_catalog_ids(self):
        r"""Get preprocessed catalog ids."""
        return self.catalog_ids

    @staticmethod
    def outlier_rejection(star_list, pos_list, mask_list, ccd_list,
                          SNR_list=None, RA_list=None, DEC_list=None,
                          shape_std_max=5., print_fun=None):
        r"""Outlier star rejection method.

        Notes
        -----
        It is based on the measurements from Galsim's HSM adaptive moments.
        The method calculates the 2nd order moments from the stars
        in the exposure. If there are stars that have an aberrant value in one
        of the stats, e1, e2 or R2 we discard the star.
        An aberrant value is defined as a value that is more
        than ``shape_std_max`` sigmas away from the mean.

        It inputs all the lists that will be used as a method and returns
        the same lists without the stars that where considered as outliers.

        ``print_fun`` is a function that prints details about the stars
        being removed.

        """
        # Define the printing function. Could be printing on a log file.
        if print_fun is None:
            def print_fun(msg):
                print(msg)

        # Reject outliers
        all_stars = np.concatenate(star_list, axis=2)
        all_stars = utils.reg_format(np.copy(all_stars))
        all_masks = np.concatenate(mask_list, axis=2)
        all_masks = utils.reg_format(np.copy(all_masks))
        # hsm thinks 0 means good
        badpix_masks = np.rint(np.abs(all_masks - 1))

        star_moms = [hsm.FindAdaptiveMom(Image(star),
                                         badpix=Image(bp),
                                         strict=False)
                     for star, bp in zip(all_stars, badpix_masks)]
        star_shapes = np.array([[moms.observed_shape.g1,
                                 moms.observed_shape.g2,
                                 2. * moms.moments_sigma ** 2,
                                 int(bool(moms.error_message))]
                                for moms in star_moms])

        # Outlier rejection based on e1, e2 and R2
        R2_thresh = shape_std_max * np.std(star_shapes[:, 2]) + \
            np.mean(star_shapes[:, 2])
        R2_bad_stars = (abs(star_shapes[:, 2]) > R2_thresh)

        e2_thresh = shape_std_max * np.std(star_shapes[:, 1]) + \
            np.mean(star_shapes[:, 1])
        e2_bad_stars = (abs(star_shapes[:, 1]) > e2_thresh)

        e1_thresh = shape_std_max * np.std(star_shapes[:, 0]) + \
            np.mean(star_shapes[:, 0])
        e1_bad_stars = (abs(star_shapes[:, 0]) > e1_thresh)

        bad_stars = np.logical_or(e1_bad_stars, e2_bad_stars)
        bad_stars = np.logical_or(bad_stars, R2_bad_stars)

        bad_stars_idx = np.nonzero(bad_stars)[0]
        print_fun(bad_stars_idx)
        print_fun(bad_stars_idx.shape)

        # Create masks
        erase_masks = [np.zeros(star_list[i].shape[2], dtype=bool)
                       for i in range(len(star_list))]

        if bad_stars_idx.size > 0:
            # We have to erase the outliers
            # Create the reference ids (to match the global
            # array to the list of arrays)
            idx_ref = np.zeros((len(all_stars), 3), dtype=int)
            glob_id, star_id, ccd_id = 0, 0, 0
            for stars in star_list:
                star_id = 0
                for _star in utils.reg_format(stars):
                    idx_ref[glob_id, 0] = glob_id
                    idx_ref[glob_id, 1] = star_id
                    idx_ref[glob_id, 2] = ccd_id
                    glob_id += 1
                    star_id += 1
                ccd_id += 1

            # select outlier stars
            for bad_id in bad_stars_idx:
                print_fun('Outlier: Glob_id=%d , star_id=%d , ccd_id=%d' % (
                    idx_ref[bad_id, 0],
                    idx_ref[bad_id, 1],
                    idx_ref[bad_id, 2]))
                erase_masks[idx_ref[bad_id, 2]][idx_ref[bad_id, 1]] = True

            for it_star in range(len(star_list)):
                mask = ~erase_masks[it_star]
                # erase elements and overwrite
                star_list[it_star] = star_list[it_star][:, :, mask]
                mask_list[it_star] = mask_list[it_star][:, :, mask]
                pos_list[it_star] = pos_list[it_star][mask, :]
                if SNR_list is not None:
                    SNR_list[it_star] = SNR_list[it_star][mask]
                if RA_list is not None:
                    RA_list[it_star] = RA_list[it_star][mask]
                    DEC_list[it_star] = DEC_list[it_star][mask]

        return star_list, pos_list, mask_list, ccd_list, SNR_list, RA_list,\
            DEC_list, erase_masks


def random_indexes(n_tot, train_per=0.8, min_n_train=20):
    r"""Generate random indexes to separate datasets.

    Separate datasets into train and test sets following some criteria.

    Parameters
    ----------
    n_tot: int
        Total number of elements.
    train_per: float
        Percentage of those elements to be used for train.
        Default is 0.8.
    min_n_train: float
        Minimum number of elements that should be used for training.
        Default is 20.

    Returns
    -------
    train_idx: numpy.ndarray
        Array of random indexes used for the training.
    test_idx: numpy.ndarray
        Array of random indexes used for the testing.

    """
    # Define number of stars
    train_star_nb = np.max([np.floor(train_per * n_tot).astype(int),
                            min_n_train])

    # Generate the random test positions in the star field
    # The positions will be maintained throughout the star sets
    rand_seq = np.random.randn(n_tot).argsort()
    train_idx = rand_seq[0:train_star_nb]
    test_idx = rand_seq[train_star_nb:]

    return train_idx, test_idx


def _get_fits_col_type(col_data_type):
    if col_data_type is None or len(col_data_type) == 0:
        col_type = 'D'
    elif col_data_type in [np.int16]:
        col_type = 'I'
    elif col_data_type in [np.int32]:
        col_type = 'J'
    elif col_data_type in [int, np.int64]:
        col_type = 'K'
    elif col_data_type in [float, np.float16, np.float32, np.float64]:
        col_type = 'D'
    elif col_data_type is bool:
        col_type = 'L'
    elif col_data_type in [str, np.str, np.str_, np.str0]:
        col_type = 'A'
    else:
        col_type = 'D'

    return col_type


def save_to_fits(dictionary, output_path):
    r"""Save dictionary of numpy.ndarray into a fits file.

    ``output_path`` should be the path + new_name to save the fits and should
    include ``.fits`` extension.

    Parameters
    ----------
    dictionary: dict
        Dictionary to be saved to the fits file.
    output_path: str
        Should be the concatenation of the path to the folder and the name
        of the new fits file and should include ``.fits`` extension.

    Notes
    -----
    It is important that all the numpy.ndarrays share the first dimension as
    we are saving a table.

    """
    # Define the header
    hdr = fits.Header()
    hdr['OWNER'] = 'MCCD package'
    hdr['COMMENT'] = 'Generated internally by the MCCD package.'
    empty_primary_hdu = fits.PrimaryHDU(header=hdr)

    # Construct the table
    dict_keys = list(dictionary.keys())
    n_elem = len(dict_keys)
    col_list = []

    for it in range(n_elem):
        data = dictionary.get(dict_keys[it])
        key = dict_keys[it]

        data_type = _get_fits_col_type(data.dtype)
        data_shape = data.shape[1:]
        fits_dim = str(tuple(data_shape))

        mem_size = 1
        if len(data_shape) != 0:
            for k in data_shape:
                mem_size *= k
            data_format = '{0}{1}'.format(mem_size, data_type)
            col_list.append(fits.Column(name=key, format=data_format,
                                        array=data, dim=fits_dim))
        else:
            data_format = '{0}{1}'.format(mem_size, data_type)
            col_list.append(fits.Column(name=key, format=data_format,
                                        array=data))

    table_hdu = fits.BinTableHDU.from_columns(col_list)
    hdul = fits.HDUList([empty_primary_hdu, table_hdu])
    # Check if file exists and in that case erase it before rewriting it
    if os.path.isfile(output_path):
        os.remove(output_path)
    # Write file
    hdul.writeto(output_path)


def save_fits(dictionary, train_bool, cat_id, output_path):
    r"""Save fits file.

    Save a dictionary into a fits file format.

    Parameters
    ----------
    dictionary: dict
        Dictionary containing the data to be saved.
    train_bool: bool
        Bool to determine if it will be a training catalog or a testing
        catalog.
        Changes the name pattern used for the file.
    cat_id: int or str
        Catalog id (exposure id) to be added in the file name.
    output_path: str
        Path to folder to save the new file.

    """
    # Save data into the FITS format extension
    train_pattern = 'train_star_selection'
    test_pattern = 'test_star_selection'
    try:
        number_scheme = "-%07d" % cat_id
    except Exception:
        number_scheme = '-' + cat_id
    ext = '.fits'

    if train_bool:
        saving_path = output_path + train_pattern + number_scheme + ext
    else:
        saving_path = output_path + test_pattern + number_scheme + ext

    save_to_fits(dictionary, saving_path)


def return_loc_neighbors(new_pos, obs_pos, vals, n_neighbors):
    r"""Find nearest neighbors locally in one CCD.

    Parameters
    ----------
    new_pos: numpy.ndarray
        Array containing the new target position (x,y).
    obs_pos: numpy.ndarray
        Array containing the positions of the training stars.
    vals: numpy.ndarray
        Values that will be used to interpolate and need to be extracted.
    n_neighbors: int
        Number of closest neighbors to return.

    Returns
    -------
    nbs: numpy.ndarray
        Values from ``vals`` of the closest ``n_neighbors`` from ``new_pos``.
    pos: numpy.ndarray
        Positions of the closest ``n_neighbors`` from ``new_pos``.

    """
    distances = np.linalg.norm(obs_pos - new_pos, axis=1)
    nbs = vals[np.argsort(distances)[:n_neighbors]]
    pos = obs_pos[np.argsort(distances)[:n_neighbors]]
    return nbs, pos


def return_glob_neighbors(new_pos, obs_pos_list, val_list, n_neighbors):
    r"""Find nearest neighbors locally in all the available CCDs.

    Parameters
    ----------
    new_pos: numpy.ndarray
        Array containing the new target position (x,y).
    obs_pos_list: list
        List containing the positions of the training stars from all the CCDs.
    val_list: list
        List containing the values from all the CCDs that will be used to
        interpolate and need to be extracted.
    n_neighbors: int
        Number of closest neighbors to return.

    Returns
    -------
    values: numpy.ndarray
        Values from ``vals`` of the closest ``n_neighbors`` from ``new_pos``.
    positions: numpy.ndarray
        Positions of the closest ``n_neighbors`` from ``new_pos``.

    """
    # Calculate all the distances
    dist_list = [np.array([np.linalg.norm(obs_pos - new_pos, axis=1),
                           (np.ones(obs_pos.shape[0]) * ccd_n),
                           np.arange(obs_pos.shape[0])])
                 for obs_pos, ccd_n in
                 zip(obs_pos_list, np.arange(len(obs_pos_list)))]
    # Sort the distances
    distances = np.concatenate(dist_list, axis=1)
    sort_idxs = np.argsort(distances[0, :])[:n_neighbors]

    # Extract values
    ccd_idxs = distances[1, sort_idxs].astype(int)
    inter_ccd_idxs = distances[2, sort_idxs].astype(int)

    values = np.array([val_list[ccd_idxs[it]].T[inter_ccd_idxs[it], :]
                       for it in range(n_neighbors)])
    positions = np.array([obs_pos_list[ccd_idxs[it]][inter_ccd_idxs[it], :]
                          for it in range(n_neighbors)])

    return values, positions


def interpolation_Pi(position_list, d_comp_glob):
    r"""Create polynomial interpolation Pi matrix.

    Create a Pi matrix list that will be used for the interpolation
    of the global model.

    Parameters
    ----------
    position_list: list
        List containing the all the positions of the training stars.
    d_comp_glob: int
        Maximum polynomial degree to be used for the Pi matrix construction.

    Returns
    -------
    interp_Pi: list
        List containing all the Pi polynomial matrices, one for each
        CCD (object in the position list).

    """
    n_comp_glob = (d_comp_glob + 1) * (d_comp_glob + 2) // 2

    # Calculate max and min values of global coordinate system
    # This configuration is specific for CFIS MegaCam configuration
    loc2glob = Loc2Glob()
    max_x = loc2glob.x_npix * 6 + loc2glob.x_gap * 5
    min_x = loc2glob.x_npix * (-5) + loc2glob.x_gap * (-5)
    max_y = loc2glob.y_npix * 2 + loc2glob.y_gap * 1
    min_y = loc2glob.y_npix * (-2) + loc2glob.y_gap * (-2)

    interp_Pi = [utils.poly_pos(pos=interp_pos, max_degree=d_comp_glob,
                                center_normalice=True,
                                x_lims=[min_x, max_x], y_lims=[min_y, max_y],
                                normalice_Pi=False)
                 for interp_pos in position_list]

    # Global position model normalisation
    # Start with the list Pi
    conc_Pi = np.concatenate((interp_Pi), axis=1)
    Pi_norms = np.sqrt(np.sum(conc_Pi**2, axis=1)).reshape(-1, 1)
    interp_Pi = [interp_Pi[k] / Pi_norms for k in range(len(interp_Pi))]

    return interp_Pi


def trim_filter(filter_array):
    r"""Trim the filters to the minimal size.

    This method will get rid of the extra zero coefficients in the filter.
    Parameters
    ----------
    filter_array: numpy.ndarray
        The filter to be trimmed
    Returns
    -------
    numpy.ndarray
        Trimmed filter

    Notes
    -----
    Function copied from ModOpt.signal.wavelet as we need it to
    replicate ModOpt's ``get_mr_filters()`` but
    using the wavelet transforms from Pysap.

    """
    non_zero_indices = np.array(np.where(filter_array != 0))
    min_idx = np.min(non_zero_indices, axis=-1)
    max_idx = np.max(non_zero_indices, axis=-1)
    return filter_array[min_idx[0]:max_idx[0] + 1, min_idx[1]:max_idx[1] + 1]


def get_mr_filters(data_shape, opt, n_scales=3, coarse=False, trim=False):
    r"""Get transform filters.

    This method obtains wavelet filters.

    Parameters
    ----------
    data_shape : tuple
        2D data shape
    opt : str
        Name of wavelet transform (in Pysap convention, see Notes)
    n_scales : int, optional
        Number of transform scales.
        Default is ``4``.
    coarse : bool, optional
        Option to keep coarse scale.
        Default is ``False``.
    trim: bool, optional
        Option to trim the filters down to their minimal size
        Default is ``False``.

    Returns
    -------
    numpy.ndarray 3D array of wavelet filters.

    Notes
    -----
    Function copied from Pysap package's astro plugin.
    Added the trim_filter() functionality from the ModOpt package.
    The name of the wavelet transform must be in Pysap convention that
    differs from the sparse2d input arguments.
    To see the available transforms in Pysap, you need to import the
    python module (``import pysap``) and then call
    ``pysap.AVAILABLE_TRANSFORMS``.

    """
    # Adjust the shape of the input data.
    data_shape = np.array(data_shape)
    data_shape += data_shape % 2 - 1

    # Create fake data.
    fake_data = np.zeros(data_shape)
    fake_data[tuple(zip(data_shape // 2))] = 1

    # Transform fake data
    wavelet_transform = (load_transform(opt)(nb_scale=n_scales, verbose=True))
    wavelet_transform.data = fake_data
    wavelet_transform.analysis()
    filters = np.array(wavelet_transform.analysis_data)

    # Added function to replicate ModOpt's get_mr_transform()
    if trim:
        filters = np.array([trim_filter(f) for f in filters])

    # Return filters
    if coarse:
        return filters
    else:
        return filters[:-1]


class MomentInterpolator(object):
    r"""Allow to interpolate moments from a bin image.

    Bin image like the one from the MeanShapes function.

    Notes
    -----
    Not used for the moment.
    """

    def __init__(self, moment_map, n_neighbors=1000,
                 rbf_function='thin_plate'):
        r"""Initialize class attributes."""
        # Save variables
        self.loc2glob = Loc2Glob()
        self.n_neighbors = n_neighbors
        self.rbf_function = rbf_function
        self.moment_map = np.zeros(moment_map.shape)
        self.x_pix = self.loc2glob.x_npix
        self.y_pix = self.loc2glob.y_npix

        # Define parameters
        self.n_ccd = moment_map.shape[0]  # 40
        self.x_grid = moment_map.shape[1]  # 20
        self.y_grid = moment_map.shape[2]  # 40
        self.bin_x = self.x_pix / self.x_grid
        self.bin_y = self.y_pix / self.y_grid

        # Correct MegaCam origin conventions
        for ccd_it in range(self.n_ccd):
            for it_x in range(self.x_grid):
                for it_y in range(self.y_grid):

                    if ccd_it < 18 or ccd_it in [36, 37]:
                        # swap x axis so origin is on top-right
                        x = it_x
                        y = it_y

                    else:
                        # swap y axis so origin is on bottom-left
                        x = self.x_grid - it_x - 1
                        y = self.y_grid - it_y - 1

                    self.moment_map[ccd_it, x, y] = moment_map[ccd_it, it_x,
                                                               it_y]

        # Generate local generic grid
        x_lin = np.linspace(start=self.bin_x / 2,
                            stop=self.x_pix - self.bin_x / 2, num=self.x_grid)
        y_lin = np.linspace(start=self.bin_y / 2,
                            stop=self.y_pix - self.bin_y / 2, num=self.y_grid)
        xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
        self.xv = xv
        self.yv = yv

        # Generate global positions for the bins
        self.x_pos = np.zeros(moment_map.shape)
        self.y_pos = np.zeros(moment_map.shape)
        for ccd_it in range(self.n_ccd):
            x_glob, y_glob = self.loc2glob.loc2glob_img_coord(
                ccd_n=ccd_it,
                x_coor=np.copy(self.xv.flatten()),
                y_coor=np.copy(self.yv.flatten()))

            self.x_pos[ccd_it, :, :] = x_glob.reshape(self.x_grid,
                                                      self.y_grid)
            self.y_pos[ccd_it, :, :] = y_glob.reshape(self.x_grid,
                                                      self.y_grid)

    def interpolate_position(self, target_x, target_y):
        r"""Interpolate positions."""
        # Calculate distances
        res_x = self.x_pos.flatten() - target_x
        res_y = self.y_pos.flatten() - target_y
        dist = np.sqrt(res_x ** 2 + res_y ** 2)

        # Select bins to use. The n_neighbors closest positions
        sort_idxs = np.argsort(dist)[:self.n_neighbors]

        # Extract values
        x_pos_interp = self.x_pos.flatten()[sort_idxs]
        y_pos_interp = self.y_pos.flatten()[sort_idxs]
        val_interp = self.moment_map.flatten()[sort_idxs]

        # Generate the interpolation function
        rbf = Rbf(x_pos_interp, y_pos_interp,
                  val_interp, function=self.rbf_function)
        output_val = rbf(target_x, target_y)

        return output_val
