import copy

import numpy as np
from scipy.stats import lognorm


class SedimentSizeDistribution:
    """Sediment size distribution

    Parameters
    ----------
    particle_diameters : numpy array
        Particle diameters in meters

    cumulative_distribution : numpy array
        Cumulative distribution of the of particles. The cumulative distribution may be by either volume or number of
        particles, as specified by the distribution parameter.

    distribution : str
        The type of distribution cumulative_volume_distribution is in. 'volume' or 'number'. Default 'number'.

    """

    def __init__(self, particle_diameters, cumulative_distribution, distribution='volume'):

        if np.any(np.diff(particle_diameters) < 0) or np.any(np.diff(cumulative_distribution) < 0):
            raise ValueError("Values must be ascending order")

        self._cdf_diameters = copy.deepcopy(particle_diameters)
        if distribution == 'volume':
            self._volume_cdf = copy.deepcopy(cumulative_distribution)
            self._number_cdf = self._calc_number_cdf(self._cdf_diameters, self._volume_cdf)
        elif distribution == 'number':
            self._number_cdf = copy.deepcopy(cumulative_distribution)
            self._volume_cdf = self._calc_volume_cdf(self._cdf_diameters, self._number_cdf)
        else:
            raise ValueError("Unknown distribution")

        self._pdf_diameters, self._number_pdf = self._calc_number_pdf(self._cdf_diameters, self._volume_cdf)
        _, self._volume_pdf = self._calc_volume_pdf(self._cdf_diameters, self._volume_cdf)

    @staticmethod
    def _calc_number_cdf(cum_particle_diameters, volume_cdf):
        """

        :param cum_particle_diameters: Diameters of the cumulative density function
        :param volume_cdf:
        :return:
        """

        diameter_diff = np.diff(cum_particle_diameters)
        diameter_mid_points = cum_particle_diameters[:-1] + diameter_diff / 2

        particle_volumes = 4 / 3 * np.pi * (diameter_mid_points / 2) ** 3

        volume_fractions = np.diff(volume_cdf)
        number_fractions = volume_fractions / particle_volumes / np.sum(volume_fractions / particle_volumes)

        cumulative_number_distribution = np.repeat(np.nan, volume_cdf.shape)

        cumulative_number_distribution[0] = 1 - np.sum(number_fractions)
        cumulative_number_distribution[1:] = np.cumsum(number_fractions) + cumulative_number_distribution[0]

        return cumulative_number_distribution

    @staticmethod
    def _calc_number_pdf(diameters, volume_cdf):
        """Calculates the probability density function of the size distribution by number of particles.

        Moore et al. 2013

        :param diameters:
        :param volume_cdf:
        :return:
        """

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
        diameter_diff = np.diff(particle_diameters)
        diameter_mid_points = particle_diameters[:-1] + diameter_diff / 2
        particle_volumes = 4 / 3 * np.pi * (diameter_mid_points / 2) ** 3

        # get the fraction by volume in each bin
        number_fractions = np.diff(cumulative_number_distribution)
        volume_in_bins = particle_volumes * number_fractions
        volume_fractions = volume_in_bins / sum(volume_in_bins)

        cumulative_volume_distribution = np.repeat(np.nan, cumulative_number_distribution.shape)

        # make sure the sum is 1 by filling the first bin with the difference
        cumulative_volume_distribution[0] = 1 - np.sum(volume_fractions)
        cumulative_volume_distribution[1:] = np.cumsum(volume_fractions) + cumulative_volume_distribution[0]

        return cumulative_volume_distribution

    @staticmethod
    def _calc_volume_pdf(diameters, cumulative_volume_distribution):
        """Calculates the probability density function of the size distribution by volume of particles.

        :param diameters:
        :param cumulative_volume_distribution:
        :return:
        """

        volume_fractions = np.diff(cumulative_volume_distribution)

        diameters_diff = np.diff(diameters)
        distribution_diameters = diameters[:-1] + diameters_diff/2

        volume_pdf = volume_fractions/diameters_diff

        return distribution_diameters, volume_pdf

    def fraction(self, distribution='volume'):
        """

        :param distribution:
        :return:
        """

        diameters, cdf = self.cdf(distribution)

        return self._pdf_diameters, np.diff(cdf)

    def cdf(self, distribution='volume', scale='normal'):
        """

        :param distribution:
        :param scale:
        :return:
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

    def mean(self, distribution='volume', scale='normal'):
        """Returns the mean particle diameter of the distribution.

        :param distribution: 'volume' or 'number'
        :param scale: 'normal' or 'log'
        :return: Mean diameter in meters
        """

        x, pdf = self.pdf(distribution, scale)

        mean = np.trapz(x * pdf, x)

        return mean

    def median(self, distribution='volume', scale='normal'):
        """Returns the median particle diameter.

        :param distribution: 'volume' or 'number'
        :param scale: 'normal' or 'log'
        :return: Median diameter in meters
        """

        x, cdf = self.cdf(distribution, scale)

        median = np.interp(0.5, cdf, x)

        return median

    def pdf(self, distribution='volume', scale='normal'):
        """

        :param distribution: 'volume' or 'number'
        :param scale: 'normal' or 'log'
        :return:
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
        """

        :param distribution:
        :param scale:
        :return:
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
    median_diameter : Median diameter (D50), in meters

    std_log : Geometric standard deviation (log-scale)

    """

    def __init__(self, median_diameter, std_log):

        self._dist = lognorm(s=std_log, loc=0, scale=median_diameter)

        # get a CDF for the distribution
        alpha = 0.000001
        d_low_quantile = self._dist.ppf(alpha)
        d_high_quantile = self._dist.ppf(1-alpha)
        d_dist = np.logspace(np.log(d_low_quantile), np.log(d_high_quantile), 1000, base=np.e)
        cdf = self._dist.cdf(d_dist)

        super().__init__(d_dist, cdf)

    def mean(self, distribution='volume', scale='normal'):
        """

        :param distribution:
        :param scale:
        :return:
        """

        if distribution == 'volume' and scale == 'normal':

            mean_diameter = self._dist.mean()

        else:

            mean_diameter = super().mean(distribution, scale)

        return mean_diameter

    def median(self, distribution='volume', scale='normal'):
        """

        :param distribution:
        :param scale:
        :return:
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
    median_diameter : Median diameter (D50), in meters

    sigma_phi : Geometric standard deviation (phi-scale)

    """

    def __init__(self, median_diameter, sigma_phi):

        # scale the phi transform standard deviation so it can be used in a lognormal distribution
        std_log = np.log(2) * sigma_phi

        super().__init__(median_diameter, std_log)
