import unittest

import numpy as np
from scoupy.sedimentsample import SedimentSample
from scoupy.sedimentsizedistribution import LogScaleSedimentSizeDistribution


class TestSedimentSample(unittest.TestCase):

    def test_add(self):

        first_sample = SedimentSample(
            1, size_distribution=[[0.01, 0.1], [0., 1.]])
        second_sample = SedimentSample(
            1, size_distribution=[[0.01, 0.1], [0., 1.]])

        third_sample = first_sample.add(second_sample)

        self.assertEqual(first_sample.size_distribution(),
                         third_sample.size_distribution())
        self.assertEqual(third_sample.concentration(), 2)

    def test_add_raises(self):

        first_distribution = LogScaleSedimentSizeDistribution(0.5, 1)
        first_sample = SedimentSample(1, size_distribution=first_distribution)

        second_distribution = LogScaleSedimentSizeDistribution(0.25, 1)
        second_sample = SedimentSample(
            2, size_distribution=second_distribution)

        self.assertRaisesRegex(ValueError, "Size distributions must have " +
                               "equivalent diameter arrays",
                               first_sample.add, second_sample)

    def test_add_simple(self):

        first_sample = SedimentSample(1)
        second_sample = SedimentSample(2)

        third_sample = first_sample.add(second_sample)

        self.assertEqual(third_sample.concentration(), 3)

    def test_init(self):

        distribution = LogScaleSedimentSizeDistribution(0.5, 1).copy()
        sample = SedimentSample(1, size_distribution=distribution)

        self.assertEqual(distribution, sample._size_distribution)
        self.assertIsNot(distribution, sample._size_distribution)

    def test_size_distribution(self):

        sample = SedimentSample(1, size_distribution=[[0.01, 0.1], [0., 1.]])

        size_distribution = sample.size_distribution()

        self.assertEqual(sample._size_distribution, size_distribution)
        self.assertIsNot(sample._size_distribution, size_distribution)

    def test_split(self):

        concentration = 100

        size_distribution = LogScaleSedimentSizeDistribution(5e-5, 0.25)
        sample = SedimentSample(
            concentration, size_distribution=size_distribution)
        fines_sample, sand_sample = sample.split()
        summed_sample = sand_sample.add(fines_sample)
        _, cdf = sample.size_distribution().cdf()
        _, summed_cdf = summed_sample.size_distribution().cdf()

        self.assertTrue(np.allclose(cdf, summed_cdf, atol=1e-5, rtol=0))
