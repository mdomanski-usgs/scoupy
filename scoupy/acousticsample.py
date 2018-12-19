import numpy as np

from scoupy.water import WaterProperties


def calc_x(particle_diameter, frequency):
    """Calculate x = ka, where k is the wave number and a is particle radius

    :param particle_diameter: Particle diameter in m
    :param frequency: Acoustic frequency in kHz
    :return:
    """

    # particle radius, in meters
    a = particle_diameter / 2

    wave_number = WaterProperties.wave_number(frequency)

    return wave_number * a


def form_function(particle_diameter, frequency):
    """Returns a backscatter form function for a single particle size as calculated with equation 7 of
    Thorne and Meral (2008)

    :param particle_diameter: Particle diameter in mm
    :param frequency: Acoustic frequency in kHz

    :return: f_e
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
    """Calculate the scattering attenuation coefficient for a mono-sized distribution as described in
    Moore and others (2013) (equation 5)

    :param particle_diameter: Particle diameter in mm
    :param frequency: Acoustic frequency in kHz
    :param sediment_density: Sediment density in kg/m**3
    :return:
    """

    particle_radius = particle_diameter / 2  # in meters
    spherical_mass = 4/3*sediment_density*particle_radius**3
    scattering_xs = scattering_cross_section(particle_diameter, frequency)

    dividend = particle_radius**2 * scattering_xs

    return dividend/spherical_mass


def scattering_cross_section(particle_diameter, frequency):
    """Returns a total scattering cross section for a mono-sized distribution as calculated with equation 9 of
    Thorne and Meral (2008)

    :param particle_diameter: Particle diameter in mm
    :param frequency: Acoustic frequency in kHz
    :return:
    """

    # particle radius in meters

    x = calc_x(particle_diameter, frequency)

    dividend = 0.29*x**4
    divisor = 0.95 + 1.28*x**2 + 0.25*x**4

    chi_e = dividend / divisor

    return chi_e


def viscous_attenuation_coefficient(particle_diameter, frequency, sediment_density):
    """Calculate the viscous attenuation coefficient for a mono-sized distribution as described in
    Moore and others (2013) (equation 7)

    :param particle_diameter: Particle diameter in m
    :param frequency: Acoustic frequency in kHz
    :param sediment_density: Density of sediment in kg/m**3
    :return:
    """

    frequency_Hz = frequency * 1000  # convert frequency to Hz

    angular_frequency = 2 * np.pi * frequency_Hz

    kinematic_viscosity = 1.3e-6
    b = np.sqrt(angular_frequency / (2*kinematic_viscosity))

    particle_radius = particle_diameter / 2  # particle radius in meters

    delta = 0.5 * (1 + 9/(2*b*particle_radius))
    s = 9/(4*b*particle_radius) * (1 + 1/(b*particle_radius))

    wave_number = WaterProperties.wave_number(frequency)

    specific_gravity_of_sediment = sediment_density / 1000
    vac = wave_number*(specific_gravity_of_sediment - 1)**2 / \
        (2*sediment_density) * (s / (s**2 + (specific_gravity_of_sediment + delta)**2))

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
        """Returns the ensemble averaged scattering attenuation coefficient as calculated with equation 5 of
        Moore and others (2008)

        :param a:
        :param number_pdf:
        :param scattering_xs:
        :param rho_s:
        :return:
        """

        dividend = 3*np.trapz(a ** 2 * scattering_xs * number_pdf, a)
        divisor = 4*rho_s*np.trapz(a ** 3 * number_pdf, a)

        return dividend / divisor

    @staticmethod
    def _mean_form_function(a, number_pdf, f_e):
        """Returns form factor for a distribution of particles as calculated with equation 3 of Thorne and Meral (2008)

        :param a: Particle radii in meters
        :param number_pdf: Number probability density function for particle radius array a
        :param f_e: Form function
        :return: mean_f_e
        """

        first_integral = np.trapz(a * number_pdf, a)
        second_integral = np.trapz(a ** 2 * f_e ** 2 * number_pdf, a)
        third_integral = np.trapz(a ** 3 * number_pdf, a)

        mean_f_e = np.sqrt(first_integral * second_integral / third_integral)

        return mean_f_e

    @staticmethod
    def _mean_viscous_attenuation_coefficient(a, number_pdf, viscous_coeff):
        """Returns the ensemble averaged viscous attenuation coefficient as calculated with equation 8 of
        Moore and others (2008)

        :param a:
        :param number_pdf:
        :param viscous_coeff:
        :return:
        """

        dividend = np.trapz(viscous_coeff * a ** 3 * number_pdf, a)
        divisor = np.trapz(a ** 3 * number_pdf, a)

        return dividend / divisor

    def attenuation_coefficient(self, frequency):
        """

        :param frequency: Frequency of acoustic signal in kHz
        :return:
        """

        # diameters, number PDF in meters
        diameters, number_pdf = self._sample.size_distribution().pdf(distribution='number')

        # radius, radius number PDF in meters
        particle_radius = diameters / 2
        radius_number_pdf = 2*number_pdf

        viscous_coeff = viscous_attenuation_coefficient(diameters, frequency, self._sample.density())
        mean_viscous = self._mean_viscous_attenuation_coefficient(particle_radius, radius_number_pdf, viscous_coeff)

        scattering_xs = scattering_cross_section(diameters, frequency)
        mean_scattering = self._mean_scattering_attenuation_coefficient(particle_radius, radius_number_pdf,
                                                                        scattering_xs, self._sample.density())

        return mean_viscous + mean_scattering

    def bin_attenuation(self, frequency):
        """Returns the attenuation for each bin

        :param frequency: Frequency of the acoustic signal in kHz
        :return: Bin centers, bin attenuation
        :rtype: tuple
        """

        bin_diameters, volume_fraction = self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction

        bin_sac = scattering_attenuation_coefficient(bin_diameters, frequency, self._sample.density())
        bin_vac = viscous_attenuation_coefficient(bin_diameters, frequency, self._sample.density())

        bin_attenuation = bin_concentration*(bin_vac + bin_sac)

        return bin_diameters, bin_attenuation

    def bin_scattering_strength(self, frequency):
        """Returns the scattering strength for each bin

        :param frequency: Frequency of acoustic signal in kHz
        :return: Bin centers, bin scattering strength
        :rtype: tuple
        """

        bin_diameters, volume_fraction = self._sample.size_distribution().fraction('volume')
        bin_concentration = self._sample.concentration()*volume_fraction

        bin_form_function = form_function(bin_diameters, frequency)

        bin_scattering_strength = \
            bin_form_function ** 2 * (3 / 8 * bin_concentration / (np.pi * bin_diameters * self._sample.density()))

        return bin_diameters, bin_scattering_strength

    def form_function(self, frequency):
        """

        :param frequency: Frequency of acoustic signal in kHz
        :return:
        """

        particle_diameters, number_distribution = self._sample.size_distribution().pdf(distribution='number')

        f_e = form_function(particle_diameters, frequency)

        mean_form_function = self._mean_form_function(particle_diameters, number_distribution, f_e)

        return mean_form_function

    def sediment_sample(self):
        """

        :return:
        """

        return self._sample
