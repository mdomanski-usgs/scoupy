import unittest

import numpy as np

from scoupy.sedimentsizedistribution import SedimentSizeDistribution


class TestSedimentSizeDistribution(unittest.TestCase):

    def test_get_median_diameter(self):

        cdf_diameters = np.linspace(0, 50)
        pdf_diameters = (cdf_diameters[1:] + cdf_diameters[:-1])/2

        number_d50 = np.median(pdf_diameters)

        volume_in_bins = 4/3*(pdf_diameters/2)**3

        cumulative_volume = np.cumsum(volume_in_bins)
        volume_cdf = np.insert(cumulative_volume, 0, 0) / np.sum(volume_in_bins)

        volume_d50 = np.interp(0.5, volume_cdf, cdf_diameters)

        psd = SedimentSizeDistribution(cdf_diameters, volume_cdf)

        calc_volume_d50 = psd.median('volume')

        self.assertTrue(np.isclose(volume_d50, calc_volume_d50))

        calc_number_d50 = psd.median('number')

        self.assertTrue(np.isclose(number_d50, calc_number_d50))
