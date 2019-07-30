"""This module contains the class definition for SedimentSample.

"""

import numpy as np

from scoupy.sedimentsizedistribution import SedimentSizeDistribution


class SedimentSample:
    """Sediment sample.

    Data type containing sediment sample data from lab analysis.

    Parameters
    ----------
    concentration : float
        Sediment concentration in kg/m**3
    density : float, optional
        Density of sediment in kg/m**3 (the default is 2650).
    size_distribution : {tuple, SedimentSizeDistribution}, optional
        Size distribution of the sediment sample (the default is None).

    Notes
    -----
    If `size_distribution` is a tuple, `size_distribution[0]` is a
    numpy.ndarray containing sediment diameters in meters and
    `size_distribution[1]` is a numpy.ndarray containing a CDF by volume for
    the size distribution. `size_distribution[0].shape` must equal
    `size_distribution[1].shape`.

    If size_distribution is not a tuple, it must be an instance of
    SedimentSizeDistribution or None.

    """

    def __init__(self, concentration, density=2650., size_distribution=None):

        self._concentration = concentration
        self._density = density

        if isinstance(size_distribution, SedimentSizeDistribution):
            self._size_distribution = size_distribution.copy()
        elif size_distribution is None:
            self._size_distribution = None
        else:
            self._size_distribution = SedimentSizeDistribution(
                *size_distribution)

    def _combine_size_distributions(self, other):
        """Combine sediment size distributions.

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        SedimentSizeDistribution

        """

        cdf_diameters, cumulative_distribution = \
            self._size_distribution.cdf('volume')

        other_cdf_diameters, other_cumulative_distribution = \
            other._size_distribution.cdf('volume')

        # CDF diameters must be equivalent
        if not np.array_equal(cdf_diameters, other_cdf_diameters):
            raise ValueError(
                "Size distributions must have equivalent diameter arrays")

        # calculate the volume in each bin for both distributions
        bin_volume_fraction = np.diff(cumulative_distribution)
        bin_volume_concentration = self._concentration / \
            self._density * bin_volume_fraction

        other_bin_volume_fraction = np.diff(other_cumulative_distribution)
        other_bin_volume_concentration = other._concentration / \
            other._density * other_bin_volume_fraction

        # add the volume from each distribution in each bin and get
        # calculate a cumulative volume
        new_bin_volume_concentration = bin_volume_concentration + \
            other_bin_volume_concentration
        new_cumulative_volume = np.insert(
            np.cumsum(new_bin_volume_concentration), 0, 0)

        # calculate a cumulative volume fraction (same as the CDF)
        new_cumulative_distribution = new_cumulative_volume / \
            new_cumulative_volume[-1]

        # initialize and return a new sediment size distribution
        new_sediment_size_distribution = SedimentSizeDistribution(
            cdf_diameters, new_cumulative_distribution)

        return new_sediment_size_distribution

    def _mean_scalar_density(self, other):
        """Calculates weighted harmonic mean from this and other densities.

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        float

        """

        new_density = (self._concentration + other._concentration) / \
            (self._concentration / self._density +
             other._concentration / other._density)

        return new_density

    def add(self, other):
        """Add a sediment sample to this sample.

        If self or other don't have size distributions, the returned
        SedimentSample will not have a size distribution.
        If both self and other have size distributions, the length of the
        diameter array of the size distributions must
        be equal.

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        SedimentSample
            Combined sediment sample.

        Notes
        -----
        Samples are added on the assumption they are taken from the same volume
        of water.

        """

        concentration = self._concentration + other._concentration

        if self._density == other._density:
            new_density = self._density
        else:
            new_density = self._mean_scalar_density(other)

        if self._size_distribution is None or other._size_distribution is None:
            new_sediment_size_distribution = None
        else:
            new_sediment_size_distribution = self._combine_size_distributions(
                other)

        return self.__class__(concentration, new_density,
                              new_sediment_size_distribution)

    def concentration(self):
        """The concentration of this sample

        Returns
        -------
        float
            Concentration in kg/m**3

        """

        return self._concentration

    def copy(self):
        """A deep copy of this SedimentSample

        Returns
        -------
        SedimentSample

        """

        cls = self.__class__
        new_sample = cls.__new__(cls)

        new_sample._density = self._density
        new_sample._concentration = self._concentration
        new_sample._size_distribution = self._size_distribution.copy()

        return new_sample

    def density(self):
        """The density of this sample

        Returns
        -------
        float
            Density in kg/m**3

        """

        return self._density

    def size_distribution(self):
        """The size distribution of this sample

        Returns
        -------
        SedimentSizeDistribution

        """

        return self._size_distribution.copy()

    def split(self, diameter=62.5e-6):
        """Splits this sample into two samples given a diameter

        Parameters
        ----------
        diameter : float, optional
            The diameter to split this sample on, in meters, (the 
            default is 62.5e-6).

        Returns
        -------
        SedimentSample, SedimentSample

        Notes
        -----
        The particle size distribution diameters in the returned
        samples are the same diameters that are contained in this
        sample. This means that `diameter` will not be a characteristic
        particle diameter in the distribution unless `diameter` is
        defined as a bin edge in the distribution of this sample.

        """

        diameters, cdf = self._size_distribution.cdf()

        split_fraction = np.interp(diameter, diameters, cdf)

        c_low = self._concentration * split_fraction
        c_high = self._concentration * (1 - split_fraction)

        cdf_low = np.clip(cdf, 0, split_fraction) / split_fraction
        cdf_adj_high = cdf - split_fraction
        cdf_high = np.clip(cdf_adj_high, 0, 1) / cdf_adj_high.max()

        sample_low = SedimentSample(c_low, self._density, (diameters, cdf_low))
        sample_high = SedimentSample(
            c_high, self._density, (diameters, cdf_high))

        return sample_low, sample_high
