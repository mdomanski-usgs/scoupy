"""This module contains the class definition and supporting functions for AcousticSample.

The primary function of the AcousticSample class is to calculate attenuation and form function for a SedimentSample.

Notes
-----
    Form function and attenuation are calculated according to [1]_ and [2]_.

References
----------
.. [1] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using multi-frequency acoustic attenuation to
   monitor grain size and concentration of suspended sediment in rivers: Journal of the Acoustical Society of America,
   v. 133, no. 4, p. 1959−1970, accessed March 11, 2016, http://dx.doi.org/10.1121/1.4792645.

.. [2] Thorne, P.D., and Meral, R., 2008, Formulations for the scattering properties of suspended sandy sediments for
   use in the application of acoustics to sediment transport processes: Continental Shelf Research, v. 28, no. 2, p.
   309–317, doi:10.1016/j.csr.2007.08.002.

"""

import numpy as np

from scoupy.water import WaterProperties


def calc_x(particle_diameter, frequency):
    """Calculate x = ka, where k is the wave number and a is particle radius

    Parameters
    ----------
    particle_diameter : float
        Particle diameter, in m

    frequency : float
        Acoustic frequency in kHz

    Returns
    -------
    x : float, array_like
        ka

    """

    # particle radius, in meters
    a = particle_diameter / 2

    wave_number = WaterProperties.wavenumber(frequency)

    return wave_number * a


def form_function(particle_diameter, frequency):
    """Backscatter form function for a single particle diameter.

    Parameters
    ----------
    particle_diameter : float
        Particle diameter in m

    frequency : float
        Acoustic frequency in kHz

    Returns
    -------
    f_e : float
        Form function

    Notes
    -----
    f_e is calculated according to equation (7) of [1]_.

    References
    ----------
    .. [1] Thorne, P.D., and Meral, R., 2008, Formulations for the scattering properties of suspended sandy sediments
       for use in the application of acoustics to sediment transport processes: Continental Shelf Research, v. 28, no.
       2, p. 309–317, doi:10.1016/j.csr.2007.08.002.

    """

    x = calc_x(particle_diameter, frequency)

    # first bracketed term
    a = 1 - 0.35 * np.exp(-((x - 1.5) / 0.7) ** 2)

    # second bracketed term
    b = 1 + 0.5 * np.exp(-((x - 1.8) / 2.2) ** 2)

    dividend = x ** 2 * a * b
    divisor = 1 + 0.9 * x ** 2

    f_e = dividend / divisor

    return f_e


def scattering_attenuation_coefficient(particle_diameter, frequency, sediment_density):
    """Scattering attenuation coefficient for mono-sized distribution

    Parameters
    ----------
    particle_diameter : float
        Particle diameter in m

    frequency : float
        Acoustic frequency in kHz

    sediment_density : float
        Density of sediment in kg/m**3

    Returns
    -------
    zeta_s : float
        Scattering attenuation coefficient in m**2/kg

    Notes
    -----
    Scattering attenuation coefficient is calculated using equation (5) of [1]_.

    References
    ----------
    .. [1] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using multi-frequency acoustic attenuation to
       monitor grain size and concentration of suspended sediment in rivers: Journal of the Acoustical Society of
       America, v. 133, no. 4, p. 1959−1970, accessed March 11, 2016, http://dx.doi.org/10.1121/1.4792645.

    """

    particle_radius = particle_diameter / 2  # in meters
    spherical_mass = 4/3*sediment_density*particle_radius**3
    scattering_xs = scattering_cross_section(particle_diameter, frequency)

    dividend = particle_radius**2 * scattering_xs

    return dividend/spherical_mass


def scattering_cross_section(particle_diameter, frequency):
    """Total scattering cross section for a mono-sized distribution

    Parameters
    ----------
    particle_diameter : float
        Particle diameter in m

    frequency : float
        Acoustic frequency in kHz

    Returns
    -------
    chi_e : Scattering cross section
        float

    Notes
    -----
    The scattering cross section is calculated with equation (9) of [1]_.

    References
    ----------
    .. [1] Thorne, P.D., and Meral, R., 2008, Formulations for the scattering properties of suspended sandy sediments
       for use in the application of acoustics to sediment transport processes: Continental Shelf Research, v. 28, no.
       2, p. 309–317, doi:10.1016/j.csr.2007.08.002.

    """

    x = calc_x(particle_diameter, frequency)

    dividend = 0.29*x**4
    divisor = 0.95 + 1.28*x**2 + 0.25*x**4

    chi_e = dividend / divisor

    return chi_e


def viscous_attenuation_coefficient(particle_diameter, frequency, sediment_density):
    """Viscous attenuation coefficient for a mono-sized distribution

    Parameters
    ----------
    particle_diameter : float
        Particle diameter in m

    frequency : float
        Acoustic frequency in kHz

    sediment_density : float
        Density of sediment in kg/m**3

    Returns
    -------
    zeta_v : float
        Viscous attenuation coefficient in m**2/kg

    Notes
    -----
    The viscous attenuation coefficient is calculated using equation (7) of [1]_.

    References
    ----------
    .. [1] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using multi-frequency acoustic attenuation to
       monitor grain size and concentration of suspended sediment in rivers: Journal of the Acoustical Society of
       America, v. 133, no. 4, p. 1959−1970, accessed March 11, 2016, http://dx.doi.org/10.1121/1.4792645.

    """

    frequency_Hz = frequency * 1000  # convert frequency to Hz

    angular_frequency = 2 * np.pi * frequency_Hz

    kinematic_viscosity = 1.3e-6
    b = np.sqrt(angular_frequency / (2*kinematic_viscosity))

    particle_radius = particle_diameter / 2  # particle radius in meters

    delta = 0.5 * (1 + 9/(2*b*particle_radius))
    s = 9/(4*b*particle_radius) * (1 + 1/(b*particle_radius))

    wave_number = WaterProperties.wavenumber(frequency)

    specific_gravity_of_sediment = sediment_density / 1000
    vac = wave_number*(specific_gravity_of_sediment - 1)**2 / \
        (2*sediment_density) * \
        (s / (s**2 + (specific_gravity_of_sediment + delta)**2))

    return vac


class AcousticSample:
    """

    Parameters
    ----------
    sediment_sample : scoupy.sedimentsample.SedimentSample

    """

    def __init__(self, sediment_sample):

        self._sample = sediment_sample

    @staticmethod
    def _mean_scattering_attenuation_coefficient(a, number_pdf, scattering_xs, rho_s):
        """Ensemble averaged scattering attenuation coefficient

        Parameters
        ----------
        a : array_like
            Particle radii in m

        number_pdf : array_like
            Number PDF

        scattering_xs : array_like
            Scattering cross section

        rho_s : float
            Density of sediment in kg/m**3

        Returns
        -------
        mean_zeta_s : float
            Ensemble averaged scattering attenuation coefficient in m**2/kg

        """

        dividend = 3*np.trapz(a ** 2 * scattering_xs * number_pdf, a)
        divisor = 4*rho_s*np.trapz(a ** 3 * number_pdf, a)

        return dividend / divisor

    @staticmethod
    def _mean_form_function(a, number_pdf, f_e):
        """Ensemble averaged backscatter form function

        Parameters
        ----------
        a : array_like
            Particle radii in m

        number_pdf : array_like
            Number PDf

        f_e : array_like
            Form function

        Returns
        -------
        mean_f_e : float
            Ensemble averaged form function

        """

        first_integral = np.trapz(a * number_pdf, a)
        second_integral = np.trapz(a ** 2 * f_e ** 2 * number_pdf, a)
        third_integral = np.trapz(a ** 3 * number_pdf, a)

        mean_f_e = np.sqrt(first_integral * second_integral / third_integral)

        return mean_f_e

    @staticmethod
    def _mean_viscous_attenuation_coefficient(a, number_pdf, viscous_coeff):
        """Ensemble averaged viscous attenuation coefficient

        Parameters
        ----------
        a : array_like
            Particle radii in m

        number_pdf : array_like
            Number PDF

        viscous_coeff : float
            Viscous attenuation coefficient in m**2/kg

        Returns
        -------
        mean_zeta_v : float
            Ensemble averaged viscous attenuation coefficient in m**2/kg

        """

        dividend = np.trapz(viscous_coeff * a ** 3 * number_pdf, a)
        divisor = np.trapz(a ** 3 * number_pdf, a)

        return dividend / divisor

    def attenuation_coefficient(self, frequency):
        """Attenuation coefficient of this sample

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        mean_zeta : float
            Ensemble averaged attenuation coefficient for this sample in m**2/kg.

        Notes
        -----
        The ensemble averaged attenuation coefficient consists of the sum of the scattering and viscous attenuation
        coefficients ([1]_). The ensemble averaged scattering and viscous coefficients are calculated according to
        equations (5) and (7) of [1]_.

        References
        ----------
        .. [1] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using multi-frequency acoustic attenuation
           to monitor grain size and concentration of suspended sediment in rivers: Journal of the Acoustical Society of
           America, v. 133, no. 4, p. 1959−1970, accessed March 11, 2016, http://dx.doi.org/10.1121/1.4792645.

        """

        # diameters, number PDF in meters
        diameters, number_pdf = self._sample.size_distribution().pdf(distribution='number')

        # radius, radius number PDF in meters
        particle_radius = diameters / 2
        radius_number_pdf = 2*number_pdf

        viscous_coeff = viscous_attenuation_coefficient(
            diameters, frequency, self._sample.density())
        mean_viscous = self._mean_viscous_attenuation_coefficient(
            particle_radius, radius_number_pdf, viscous_coeff)

        scattering_xs = scattering_cross_section(diameters, frequency)
        mean_scattering = self._mean_scattering_attenuation_coefficient(particle_radius, radius_number_pdf,
                                                                        scattering_xs, self._sample.density())

        return mean_viscous + mean_scattering

    def bin_attenuation(self, frequency):
        """Sediment attenuation contribution from each bin of the sediment sample of this instance

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        diameters, attenuation : array_like
            `diameters` is an array of diameters in m. `attenuation` is an array of attenuation values in dB/m.

        Notes
        -----
        `diameters` is an array of mid-points of the distribution of this instance. `attenuation` is the attenuation
        calculated for each diameter.

        """

        bin_diameters, volume_fraction = self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction

        bin_sac = scattering_attenuation_coefficient(
            bin_diameters, frequency, self._sample.density())
        bin_vac = viscous_attenuation_coefficient(
            bin_diameters, frequency, self._sample.density())

        bin_attenuation = bin_concentration*(bin_vac + bin_sac)

        return bin_diameters, bin_attenuation

    def bin_scattering_strength(self, frequency):
        """Scattering strength contribution from each bin of the sediment sample of this instance

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        diameter, scattering_strength : array_like
            `diameters` is an array of diameters in m. `scattering_strength` is an array of scattering strength values
            in dB.
        """

        bin_diameters, volume_fraction = self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction

        bin_form_function = form_function(bin_diameters, frequency)

        bin_scattering_strength = \
            bin_form_function ** 2 * \
            (3 / 8 * bin_concentration /
             (np.pi * bin_diameters * self._sample.density()))

        return bin_diameters, bin_scattering_strength

    def form_function(self, frequency):
        """Ensemble averaged form function of this sample

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        mean_f_e : float
            Ensemble averaged form function

        Notes
        -----
        The ensemble averaged form function is calculated using equation (3) of [1]_.

        References
        ----------
        .. [1] Thorne, P.D., and Meral, R., 2008, Formulations for the scattering properties of suspended sandy
           sediments for use in the application of acoustics to sediment transport processes: Continental Shelf
           Research, v. 28, no. 2, p. 309–317, doi:10.1016/j.csr.2007.08.002.

        """

        particle_diameters, number_distribution = self._sample.size_distribution().pdf(
            distribution='number')

        f_e = form_function(particle_diameters, frequency)

        mean_form_function = self._mean_form_function(
            particle_diameters, number_distribution, f_e)

        return mean_form_function

    def sediment_sample(self):
        """Sediment sample of this instance

        Returns
        -------
        sample : SedimentSample

        """

        return self._sample.copy()
