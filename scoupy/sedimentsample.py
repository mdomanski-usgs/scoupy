import numpy as np

from scoupy.sedimentsizedistribution import SedimentSizeDistribution


class SedimentSample:
    """Sediment sample

    Parameters
    ------
    concentration : float
        Sediment concentration in kg/m**3

    density : float
        Density of sediment in kg/m**3. Default 2650.

    size_distribution : iterable or scoupy.sedimentsizedistribution.SedimentSizeDistribution
        Size distribution of the sediment sample.

        If size_distribution is iterable,
            size_distribution[0] is a numpy.ndarray containing sediment diameters in meters and
            size_distribution[1] is a numpy.ndarray containing a CDF by volume for the size distribution.
        size_distribution[0].shape must equal size_distribution[1].shape.

        If size_distribution is not iterable, it must be an instance of SedimentSizeDistribution or None.

        Default None.

    """

    def __init__(self, concentration, density=2650., size_distribution=None):

        self._concentration = concentration
        self._density = density

        try:
            self._size_distribution = SedimentSizeDistribution(*size_distribution)
        except TypeError:
            self._size_distribution = size_distribution

    def _combine_size_distributions(self, other):
        """

        :param other:
        :type other: SedimentSample
        :return:
        :rtype: SedimentSizeDistribution

        """

        cdf_diameters, cumulative_distribution = self._size_distribution.cdf('volume')

        other_cdf_diameters, other_cumulative_distribution = other._size_distribution.cdf('volume')

        # if the diameter arrays are equivalent, set the new diameters to self diameters
        if np.array_equiv(cdf_diameters, other_cdf_diameters):
            new_diameters = cdf_diameters

        # otherwise create a new array
        else:
            lower_diameter_bound = np.min([cdf_diameters.min(), other_cdf_diameters.min()])
            upper_diameter_bound = np.max([cdf_diameters.max(), other_cdf_diameters.max()])

            number_of_diameters = np.max([cdf_diameters.shape[0], other_cdf_diameters.shape[0]])

            new_diameters = np.logspace(np.log(lower_diameter_bound), np.log(upper_diameter_bound),
                                        number_of_diameters, base=np.e)

        interp_cumulative_volume = np.interp(new_diameters, cdf_diameters, cumulative_distribution, left=0)
        bin_volume_fraction = np.diff(interp_cumulative_volume)
        bin_volume_concentration = self._concentration / self._density * bin_volume_fraction

        interp_other_cumulative_volume = np.interp(new_diameters, other_cdf_diameters,
                                                   other_cumulative_distribution, left=0)
        other_bin_volume_fraction = np.diff(interp_other_cumulative_volume)
        other_bin_volume_concentration = other._concentration / other._density * other_bin_volume_fraction

        new_bin_volume_concentration = bin_volume_concentration + other_bin_volume_concentration
        new_cumulative_volume = np.insert(np.cumsum(new_bin_volume_concentration), 0, 0)
        new_cumulative_distribution = new_cumulative_volume/new_cumulative_volume[-1]

        new_sediment_size_distribution = SedimentSizeDistribution(new_diameters, new_cumulative_distribution)

        return new_sediment_size_distribution

    def _weighted_average_scalar_density(self, other):
        """

        :param other:
        :type other: SedimentSample
        :return:
        :rtype: float

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
        """Add a sediment sample to this sample. Returns a new instance of SedimentSample.

        Samples are added based on the assumption they are from the same volume of water.

        :param other: Other sediment sample
        :type other: SedimentSample
        :return: Combined sediment sample
        :rtype: SedimentSample

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
        """Returns the concentration of this sample in kilograms per meter squared

        :return: Concentration
        :rtype: float

        """

        return self._concentration

    def density(self):
        """Returns the density of this sample in kilograms per meter squared

        :return: Density
        :rtype: float

        """

        return self._density

    def size_distribution(self):
        """Returns the size distribution of this sample.

        :return: Size distribution
        :rtype: scoupy.sedimentsizedistribution.SedimentSizeDistribution

        """

        return self._size_distribution
