# -*- coding: utf-8 -*-

"""UNIT TESTS FOR EXAMPLE

This module contains unit tests for the example module.

"""

from unittest import TestCase
import numpy.testing as npt
from ..example import *


class ExampleTestCase(TestCase):

    def setUp(self):

        self.x = 1
        self.y = 2

    def tearDown(self):

        self.x = None
        self.y = None

    def test_add_int(self):

        npt.assert_equal(math.add_int(self.x, self.y), 3,
                         err_msg='Incorrect addition result.')

        npt.assert_raises(TypeError, math.add_int, self.x,
                          float(self.y))
