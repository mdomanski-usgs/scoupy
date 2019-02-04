"""This module contains the class definition for SedimentSample.

"""

import numpy as np

from scoupy.sedimentsizedistribution import SedimentSizeDistribution


class SedimentSample:
    """Sediment sample.

    Data type containing sediment sample data from lab analysis.

    Parameters
    ------
    concentration : float
        Sediment concentration in kg/m**3

    density : float, optional
        Density of sediment in kg/m**3 (the default is 2650).

    size_distribution : {tuple, SedimentSizeDistribution}, optional
        Size distribution of the sediment sample (the default is None).

        If size_distribution is a tuple,
            size_distribution[0] is a numpy.ndarray containing sediment diameters in meters and
            size_distribution[1] is a numpy.ndarray containing a CDF by volume for the size distribution.
        size_distribution[0].shape must equal size_distribution[1].shape.

        If size_distribution is not a tuple, it must be an instance of SedimentSizeDistribution or None.

    """

    def __init__(self, concentration, density=2650., size_distribution=None):

        self._concentration = concentration
        self._density = density

        if isinstance(size_distribution, SedimentSizeDistribution):
            self._size_distribution = size_distribution.copy()
        elif size_distribution is None:
            self._size_distribution = None
        else:
            self._size_distribution = SedimentSizeDistribution(*size_distribution)

    def _combine_size_distributions(self, other):
        """Combine sediment size distributions.

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        size_distribution : SedimentSizeDistribution

        """

        cdf_diameters, cumulative_distribution = self._size_distribution.cdf('volume')

        other_cdf_diameters, other_cumulative_distribution = other._size_distribution.cdf('volume')

        # if the diameter arrays are equivalent, set the new diameters to self diameters
        if not np.array_equal(cdf_diameters, other_cdf_diameters):
            raise ValueError("Size distributions must have equivalent diameter arrays")

        bin_volume_fraction = np.diff(cumulative_distribution)
        bin_volume_concentration = self._concentration / self._density * bin_volume_fraction

        other_bin_volume_fraction = np.diff(other_cumulative_distribution)
        other_bin_volume_concentration = other._concentration / other._density * other_bin_volume_fraction

        new_bin_volume_concentration = bin_volume_concentration + other_bin_volume_concentration
        new_cumulative_volume = np.insert(np.cumsum(new_bin_volume_concentration), 0, 0)
        new_cumulative_distribution = new_cumulative_volume/new_cumulative_volume[-1]

        new_sediment_size_distribution = SedimentSizeDistribution(cdf_diameters, new_cumulative_distribution)

        return new_sediment_size_distribution

    def _weighted_average_scalar_density(self, other):
        """

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        density : float

        """

        # get a volume weighted average of the sample densities for the new sample density
        sediment_volume = self._concentration/self._density
        other_sediment_volume = other._concentration/other._density
        total_sediment_volume = sediment_volume + other_sediment_volume

        density_weight = sediment_volume/total_sediment_volume
        other_density_weight = other_sediment_volume/total_sediment_volume

        new_density = density_weight*self._density + other_density_weight*other._density

        return new_density

    def add(self, other):
        """Add a sediment sample to this sample.

        If self or other don't have size distributions, the returned SedimentSample will not have a size distribution.
        If both self and other have size distributions, the length of the diameter array of the size distributions must
        be equal.

        Parameters
        ----------
        other : SedimentSample

        Returns
        -------
        combined_sample : SedimentSample
            Combined sediment sample.

        Notes
        -----
        Samples are added on the assumption they are taken from the same volume of water.

        """

        concentration = self._concentration + other._concentration

        if self._density == other._density:
            new_density = self._density
        else:
            new_density = self._weighted_average_scalar_density(other)

        if self._size_distribution is None or other._size_distribution is None:
            new_sediment_size_distribution = None
        else:
            new_sediment_size_distribution = self._combine_size_distributions(other)

        return self.__class__(concentration, new_density, new_sediment_size_distribution)

    def concentration(self):
        """The concentration of this sample

        Returns
        -------
        concentration : float
            Concentration in kg/m**3

        """

        return self._concentration

    def density(self):
        """The density of this sample

        Returns
        -------
        density : float
            Density in kg/m**3

        """

        return self._density

    def size_distribution(self):
        """The size distribution of this sample

        Returns
        -------
        size_distribution : SedimentSizeDistribution

        """

        return self._size_distribution.copy()
