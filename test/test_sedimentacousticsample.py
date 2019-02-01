import unittest

import numpy as np

from scoupy.sedimentsample import SedimentSample
from scoupy.sedimentsizedistribution import SedimentSizeDistribution
from scoupy.acousticsample import form_function, AcousticSample


class TestFormFactorCalculation(unittest.TestCase):

    def test_form_factor_calculation(self):

        d50 = 10/1e6

        a50 = d50/2
        std = 1.

        mean = np.log(a50)
        radii = np.random.lognormal(mean, std, int(1e6))

        frequency = 1200  # kHz

        f_e = form_function(2 * radii, frequency)

        mean_radius = radii.mean()
        second_integral = np.mean(radii ** 2 * f_e ** 2)
        third_integral = np.mean(radii ** 3)

        expected_form_function = np.sqrt(mean_radius * second_integral / third_integral)

        bin_edges = np.logspace(np.log10(0.375124), np.log10(2000), num=93)/2/1e6
        hist, bin_edges = np.histogram(radii, bins=bin_edges)

        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_particle_volumes = 4 / 3 * np.pi * bin_centers ** 3
        bin_volumes = hist * bin_particle_volumes

        cumulative_volume = np.cumsum(bin_volumes)
        cumulative_volume = np.insert(cumulative_volume, 0, 0)
        volume_cdf = cumulative_volume / np.sum(bin_volumes)

        cdf_diameters = 2 * bin_edges

        size_distribution = SedimentSizeDistribution(cdf_diameters, volume_cdf)
        sample = SedimentSample(1, size_distribution=size_distribution)

        calculated_form_function = AcousticSample(sample).form_function(frequency)

        self.assertTrue(np.isclose(expected_form_function, calculated_form_function, rtol=0.01))
