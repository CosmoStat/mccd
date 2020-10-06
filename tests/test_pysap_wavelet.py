from unittest import TestCase
import numpy.testing as npt
import numpy as np
from mccd.mccd_utils import get_mr_filters


class PysapWaveletTestCase(TestCase):
    def setUp(self):
        self.data_shape = (5, 5)
        self.opt = 'BsplineWaveletTransformATrousAlgorithm'
        self.n_scales = 3
        self.new_filters = None

        self.expected_filter = np.array([
            [[-0.00390625, -0.015625, -0.0234375, -0.015625, -0.00390625],
             [-0.015625, -0.0625, -0.09375, -0.0625, -0.015625],
             [-0.0234375, -0.09375, 0.859375, -0.09375, -0.0234375],
             [-0.015625, -0.0625, -0.09375, -0.0625, -0.015625],
             [-0.00390625, -0.015625, -0.0234375, -0.015625, -0.00390625]
             ],

            [[-0.01586914, -0.00964355, -0.00183105, -0.00964355, -0.01586914],
             [-0.00964355, 0.0302124, 0.0614624, 0.0302124, -0.00964355],
             [-0.00183105, 0.0614624, 0.1083374, 0.0614624, -0.00183105],
             [-0.00964355, 0.0302124, 0.0614624, 0.0302124, -0.00964355],
             [-0.01586914, -0.00964355, -0.00183105, -0.00964355, -0.01586914]
             ],

            [[0.01977539, 0.02526855, 0.02526855, 0.02526855, 0.01977539],
             [0.02526855, 0.0322876, 0.0322876, 0.0322876, 0.02526855],
             [0.02526855, 0.0322876, 0.0322876, 0.0322876, 0.02526855],
             [0.02526855, 0.0322876, 0.0322876, 0.0322876, 0.02526855],
             [0.01977539, 0.02526855, 0.02526855, 0.02526855, 0.01977539]
             ]],
             dtype=np.float32)

    def tearDown(self):
        self.data_shape = None
        self.opt = None
        self.n_scales = None
        self.new_filters = None
        self.expected_filter = None

    def get_filters(self):
        self.new_filters = get_mr_filters(self.data_shape, self.opt,
                                          n_scales=self.n_scales,
                                          coarse=True, trim=False)

    def test_pysap_wavelet(self):
        self.get_filters()
        npt.assert_almost_equal(self.new_filters, self.expected_filter,
                                decimal=6,
                                err_msg='''The filter output from Pysap is not
                                the expected one.''')


if __name__ == '__main__':
    test_instance = PysapWaveletTestCase()
    test_instance.setUp()
    test_instance.test_pysap_wavelet()
