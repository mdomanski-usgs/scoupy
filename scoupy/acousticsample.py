"""This module contains the class definition and supporting functions for
AcousticSample.

The primary function of the AcousticSample class is to calculate attenuation
and form function for a SedimentSample.

Notes
-----
Form function and attenuation are calculated according to [3]_ and [4]_.

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
    float, array_like

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
    float
        Form function

    Notes
    -----
    f_e is calculated according to equation (7) of [4]_.

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


def scattering_attenuation_coefficient(particle_diameter, frequency,
                                       sediment_density):
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
    float
        Scattering attenuation coefficient in m**2/kg

    Notes
    -----
    Scattering attenuation coefficient is calculated using equation (5)
    of [3]_.

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
    float
        Scattering cross section

    Notes
    -----
    The scattering cross section is calculated with equation (9) of [4]_.

    """

    x = calc_x(particle_diameter, frequency)

    dividend = 0.29*x**4
    divisor = 0.95 + 1.28*x**2 + 0.25*x**4

    chi_e = dividend / divisor

    return chi_e


def viscous_attenuation_coefficient(particle_diameter, frequency,
                                    sediment_density):
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
    float
        Viscous attenuation coefficient in m**2/kg

    Notes
    -----
    The viscous attenuation coefficient is calculated using equation (7) of
    [3]_.

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
    sediment_sample : SedimentSample

    """

    def __init__(self, sediment_sample):

        self._sample = sediment_sample.copy()

    @staticmethod
    def _mean_scat_attenuation_coefficient(a, number_pdf, scattering_xs,
                                           rho_s):
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
        float
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
        float
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
        float
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
        float
            Ensemble averaged attenuation coefficient for this sample in
            dB * m**2/kg.

        Notes
        -----
        The ensemble averaged attenuation coefficient consists of the sum of
        the scattering and viscous attenuation coefficients (eqation (4) of
        [3]_). The ensemble averaged scattering and viscous coefficients are
        calculated according to equations (5) and (7) of [3]_.

        """

        # diameters, number PDF in meters
        diameters, number_pdf = \
            self._sample.size_distribution().pdf(distribution='number')

        # radius, radius number PDF in meters
        particle_radius = diameters / 2
        radius_number_pdf = 2*number_pdf

        viscous_coeff = viscous_attenuation_coefficient(
            diameters, frequency, self._sample.density())
        mean_viscous = self._mean_viscous_attenuation_coefficient(
            particle_radius, radius_number_pdf, viscous_coeff)

        scattering_xs = scattering_cross_section(diameters, frequency)
        mean_scattering = \
            self._mean_scat_attenuation_coefficient(particle_radius,
                                                    radius_number_pdf,
                                                    scattering_xs,
                                                    self._sample.density())

        return mean_viscous + mean_scattering

    def bin_attenuation(self, frequency):
        """Sediment attenuation contribution from each bin of the sediment
        sample of this instance

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        diameters, attenuation : array_like
            `diameters` is an array of diameters in m. `attenuation` is an
            array of attenuation values in dB/m.

        Notes
        -----
        `diameters` is an array of mid-points of the distribution of this
        instance. `attenuation` is the attenuation calculated for each
        diameter.

        """

        bin_diameters, volume_fraction = \
            self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction

        bin_sac = scattering_attenuation_coefficient(
            bin_diameters, frequency, self._sample.density())
        bin_vac = viscous_attenuation_coefficient(
            bin_diameters, frequency, self._sample.density())

        bin_attenuation = bin_concentration*(bin_vac + bin_sac)

        return bin_diameters, bin_attenuation

    def bin_scattering_strength(self, frequency):
        """Scattering strength contribution from each bin of the sediment
        sample of this instance

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        diameter, scattering_strength : array_like
            `diameters` is an array of diameters in m. `scattering_strength` is
            an array of scattering strength values in dB.
        """

        bin_diameters, volume_fraction = \
            self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction
        density = self._sample.density()

        bin_form_function = form_function(bin_diameters, frequency)

        bin_scattering_strength = \
            bin_form_function ** 2 * \
            (3 / 8 * bin_concentration /
             (np.pi * bin_diameters * density))

        return bin_diameters, bin_scattering_strength

    def form_function(self, frequency):
        """Ensemble averaged form function of this sample

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        float
            Ensemble averaged form function

        Notes
        -----
        The ensemble averaged form function is calculated using equation (3)
        of [4]_.

        """

        particle_diameters, number_distribution = \
            self._sample.size_distribution().pdf(distribution='number')

        f_e = form_function(particle_diameters, frequency)

        mean_form_function = self._mean_form_function(
            particle_diameters, number_distribution, f_e)

        return mean_form_function

    def scattering_strength(self, frequency):
        """Scattering strength for this this sample.

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz

        Returns
        -------
        float
            Scattering strength in dB

        Notes
        -----
        The form function in the scattering strength from this method is
        from `form_function`.

        """

        form_function = self.form_function(frequency)
        concentration = self._sample.concentration()
        mean_diameter = self._sample.size_distribution().mean('number')
        density = self._sample.density()

        scattering_strength = form_function ** 2 * \
            (3 / 8 * concentration /
             (np.pi * mean_diameter * density))

        return scattering_strength

    def sediment_sample(self):
        """Sediment sample of this instance

        Returns
        -------
        SedimentSample

        """

        return self._sample.copy()
