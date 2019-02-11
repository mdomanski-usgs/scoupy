"""This module contains class definitions for sediment size distribution data types.

The data types calculate distribution parameters (mean, median, standard deviation) and allow conversion between
distribution by volume and number of particles.

`SedimentSizeDistribution` is the base, and most generic, type.

Notes
-----
Distribution type (volume, number) conversions are calculated according to equations (9) and (10) of [1]_.

References
----------
.. [1] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using multi-frequency acoustic attenuation to
   monitor grain size and concentration of suspended sediment in rivers: Journal of the Acoustical Society of America,
   v. 133, no. 4, p. 1959−1970, accessed March 11, 2016, http://dx.doi.org/10.1121/1.4792645.

"""

import copy

import numpy as np
from scipy.stats import lognorm


class SedimentSizeDistribution:
    """Sediment size distribution.

    `particle_diameters` and `cumulative_distribution` must be one-dimensional and of the same length.

    Parameters
    ----------
    particle_diameters : array_like
        Particle diameters, in meters.

    cumulative_distribution : array_like
        Cumulative distribution of the of particles. The cumulative distribution may be by either volume or number of
        particles, as specified by `distribution`.

    distribution : {'volume', 'number'}, optional
        The type of distribution of `cumulative_volume_distribution` (the default is 'volume').


    """

    def __init__(self, particle_diameters, cumulative_distribution, distribution='volume'):
        """Initialize self. See help(type(self)) for accurate signature."""

        particle_diameters = np.array(particle_diameters)
        cumulative_distribution = np.array(cumulative_distribution)

        if np.any(np.diff(particle_diameters) < 0) or np.any(np.diff(cumulative_distribution) < 0):
            raise ValueError("Values must be ascending order")

        if np.ndim(particle_diameters) != 1 or np.ndim(cumulative_distribution) != 1:
            raise ValueError("Array must be one-dimensional")

        if len(particle_diameters) != len(cumulative_distribution):
            raise ValueError("Array shapes must be equal")

        self._cdf_diameters = copy.deepcopy(particle_diameters)
        if distribution == 'volume':
            self._volume_cdf = copy.deepcopy(cumulative_distribution)
            self._number_cdf = self._calc_number_cdf(
                self._cdf_diameters, self._volume_cdf)
        elif distribution == 'number':
            self._number_cdf = copy.deepcopy(cumulative_distribution)
            self._volume_cdf = self._calc_volume_cdf(
                self._cdf_diameters, self._number_cdf)
        else:
            raise ValueError("Unknown distribution")

        self._pdf_diameters, self._number_pdf = self._calc_number_pdf(
            self._cdf_diameters, self._volume_cdf)
        _, self._volume_pdf = self._calc_volume_pdf(
            self._cdf_diameters, self._volume_cdf)

    def __eq__(self, other):
        """

        Parameters
        ----------
        other : SedimentSizeDistribution

        Returns
        -------

        """

        if self.__class__ != other.__class__:
            return False
        else:
            return np.array_equal(self._cdf_diameters, other._cdf_diameters) \
                and np.array_equal(self._volume_cdf, other._volume_cdf)

    @staticmethod
    def _calc_number_cdf(cum_particle_diameters, volume_cdf):

        diameter_diff = np.diff(cum_particle_diameters)
        diameter_mid_points = cum_particle_diameters[:-1] + diameter_diff / 2

        particle_volumes = 4 / 3 * np.pi * (diameter_mid_points / 2) ** 3

        volume_fractions = np.diff(volume_cdf)
        number_fractions = volume_fractions / particle_volumes / \
            np.sum(volume_fractions / particle_volumes)

        cumulative_number_distribution = np.repeat(np.nan, volume_cdf.shape)

        cumulative_number_distribution[0] = 1 - np.sum(number_fractions)
        cumulative_number_distribution[1:] = np.cumsum(
            number_fractions) + cumulative_number_distribution[0]

        return cumulative_number_distribution

    @staticmethod
    def _calc_number_pdf(diameters, volume_cdf):

        volume_fractions = np.diff(volume_cdf)

        diameters_diff = np.diff(diameters)
        distribution_diameters = diameters[:-1] + diameters_diff/2
        particle_volumes = 4/3*np.pi*(distribution_diameters/2)**3

        number_in_bins = volume_fractions/particle_volumes

        number_fractions = number_in_bins/np.sum(number_in_bins)

        number_distribution = number_fractions/diameters_diff

        return distribution_diameters, number_distribution

    @staticmethod
    def _calc_volume_cdf(diameters, cumulative_number_distribution):

        # calculate the volume of particles at the mid-point of each bin
        diameter_diff = np.diff(diameters)
        diameter_mid_points = diameters[:-1] + diameter_diff / 2
        particle_volumes = 4 / 3 * np.pi * (diameter_mid_points / 2) ** 3

        # get the fraction by volume in each bin
        number_fractions = np.diff(cumulative_number_distribution)
        volume_in_bins = particle_volumes * number_fractions
        volume_fractions = volume_in_bins / sum(volume_in_bins)

        cumulative_volume_distribution = np.repeat(
            np.nan, cumulative_number_distribution.shape)

        # make sure the sum is 1 by filling the first bin with the difference
        cumulative_volume_distribution[0] = 1 - np.sum(volume_fractions)
        cumulative_volume_distribution[1:] = np.cumsum(
            volume_fractions) + cumulative_volume_distribution[0]

        return cumulative_volume_distribution

    @staticmethod
    def _calc_volume_pdf(diameters, cumulative_volume_distribution):

        volume_fractions = np.diff(cumulative_volume_distribution)

        diameters_diff = np.diff(diameters)
        distribution_diameters = diameters[:-1] + diameters_diff/2

        volume_pdf = volume_fractions/diameters_diff

        return distribution_diameters, volume_pdf

    def fraction(self, distribution='volume'):
        """The fraction of the distribution in each size class.

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').

        Returns
        -------
        diameters, fraction : numpy.ndarray
            Representative diameters and fraction of distribution

        Notes
        -----
        The values in `diameters` are mid-points of the bins defined by the `particle_diameters` parameter of the
        initialization method.

        `fraction` is calculated as the difference of the CDF value in each bin.

        """

        _, cdf = self.cdf(distribution)

        return self._pdf_diameters, np.diff(cdf)

    def cdf(self, distribution='volume', scale='normal'):
        """Cumulative distribution function.

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').
        scale : {'normal', 'log'}, optional
            Scale of distribution (the default is 'normal').

        Returns
        -------
        diameters, cdf : numpy.ndarray
            If `scale` is 'normal', `diameters` is in meters.

        """

        if distribution == 'volume':
            cdf = self._volume_cdf.copy()
        elif distribution == 'number':
            cdf = self._number_cdf.copy()
        else:
            raise ValueError("Invalid distribution")

        if scale == 'normal':
            x = self._cdf_diameters.copy()
        elif scale == 'log':
            x = np.log(self._cdf_diameters)
        else:
            raise ValueError("Invalid scale")

        return x, cdf

    def copy(self):
        """Returns a copy of this instance

        Returns
        -------
        size_distribution : SedimentSizeDistribution

        """

        return SedimentSizeDistribution(self._cdf_diameters, self._volume_cdf)

    def mean(self, distribution='volume', scale='normal'):
        """The mean particle diameter of the distribution.

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').
        scale : {'normal', 'log'}, optional.
            Scale of distribution (the default is 'normal').

        Returns
        -------
        mean : float
            Mean particle diameter (in meters if `scale` is normal).

        """

        x, pdf = self.pdf(distribution, scale)

        mean = np.trapz(x * pdf, x)

        return mean

    def median(self, distribution='volume', scale='normal'):
        """The median particle diameter of the distribution

        Parameters
        ----------
        distribution
        scale

        Returns
        -------
        median : float
            Median particle diameter (in meters if `scale` is normal).

        """

        x, cdf = self.cdf(distribution, scale)

        median = np.interp(0.5, cdf, x)

        return median

    def pdf(self, distribution='volume', scale='normal'):
        """Probability density function (PDF).

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').
        scale : {'normal', 'log'}, optional
            Scale of distribution (the default is 'normal').

        Returns
        -------
        diameters, pdf : numpy.ndarray
            The diameters and values of a PDF. If `scale` is 'normal', `diameters` is in meters and `pdf` is in
            meters**-1.

        """

        if distribution == 'volume':
            pdf = self._volume_pdf.copy()
        elif distribution == 'number':
            pdf = self._number_pdf.copy()
        else:
            raise ValueError("Invalid distribution")

        if scale == 'normal':
            x = self._pdf_diameters.copy()
        elif scale == 'log':
            x = np.log(self._pdf_diameters)
            pdf = pdf / np.trapz(pdf, x)
        else:
            raise ValueError("Invalid scale")

        return x, pdf

    def std(self, distribution='volume', scale='normal'):
        """Standard deviation.

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').
        scale : {'normal', 'log'}, optional
            Scale of distribution (the default is 'normal').

        Returns
        -------
        std : float
            Standard deviation of the distribution. If `scale` is 'normal', `std` is in meters.

        """

        mean_x = self.mean(distribution=distribution, scale=scale)

        x, pdf = self.pdf(distribution, scale)

        variance = np.trapz((x - mean_x)**2 * pdf, x)

        standard_deviation = np.sqrt(variance)

        return standard_deviation


class LogScaleSedimentSizeDistribution(SedimentSizeDistribution):
    """Lognormal sediment size distribution.

    Parameters
    ----------
    median_diameter : float
        Median diameter, in meters.

    std_log : float
        Geometric standard deviation (log-scale)

    """

    def __init__(self, median_diameter, std_log):

        self._dist = lognorm(s=std_log, loc=0, scale=median_diameter)

        # get a CDF for the distribution
        alpha = 0.000001
        d_low_quantile = self._dist.ppf(alpha)
        d_high_quantile = self._dist.ppf(1-alpha)
        d_dist = np.logspace(np.log(d_low_quantile), np.log(
            d_high_quantile), 1000, base=np.e)
        cdf = self._dist.cdf(d_dist)

        super().__init__(d_dist, cdf)

    def mean(self, distribution='volume', scale='normal'):
        """The mean particle diameter of the distribution.

        Parameters
        ----------
        distribution : {'volume', 'number'}, optional
            Distribution type (the default is 'volume').
        scale : {'normal', 'log'}, optional.
            Scale of distribution (the default is 'normal').

        Returns
        -------
        mean : float
            Mean particle diameter (in meters if `scale` is normal).

        """

        if distribution == 'volume' and scale == 'normal':

            mean_diameter = self._dist.mean()

        else:

            mean_diameter = super().mean(distribution, scale)

        return mean_diameter

    def median(self, distribution='volume', scale='normal'):
        """The median particle diameter of the distribution

        Parameters
        ----------
        distribution
        scale

        Returns
        -------
        median : float
            Median particle diameter (in meters if `scale` is normal).

        """

        if distribution == 'volume' and scale == 'normal':

            median_diameter = self._dist.median()

        else:

            median_diameter = super().median(distribution)

        return median_diameter


class PhiScaleSedimentSizeDistribution(LogScaleSedimentSizeDistribution):
    """Phi-scale size distribution.

    Parameters
    ----------
    median_diameter : float
        Median diameter (D50), in meters

    sigma_phi : float
        Geometric standard deviation (phi-scale)

    """

    def __init__(self, median_diameter, sigma_phi):

        # scale the phi transform standard deviation so it can be used in a lognormal distribution
        std_log = np.log(2) * sigma_phi

        super().__init__(median_diameter, std_log)
