import numpy as np


class WaterProperties:

    SPEED_OF_SOUND_IN_WATER = 1442.5  # m/s
    KINEMATIC_VISCOSITY = 1.3e-6

    @classmethod
    def speed_of_sound(cls, temperature=None):
        """Calculate the speed of sound in water (in meters per second) based on Marczak, 1997.

        If temperature is None, returns 1442.5 m/s

        :param temperature: Temperature, in degrees C
        :return: speed_of_sound: Speed of sound in m/s
        """

        if temperature is None:
            speed_of_sound = cls.SPEED_OF_SOUND_IN_WATER
        else:
            speed_of_sound = 1.402385 * 10 ** 3 + 5.038813 * temperature - \
                             (5.799136 * 10 ** -2) * temperature ** 2 + \
                             (3.287156 * 10 ** -4) * temperature ** 3 - \
                             (1.398845 * 10 ** -6) * temperature ** 4 + \
                             (2.787860 * 10 ** -9) * temperature ** 5

        return speed_of_sound

    @classmethod
    def kinematic_viscosity(cls, temperature=None):
        """Calculates the kinematic viscosity of water

        :param temperature: Temperature, in degrees C
        """

        if temperature is None:
            return cls.KINEMATIC_VISCOSITY
        else:
            raise NotImplementedError("Method is not implemented")

    @classmethod
    def water_absorption(cls, frequency, temperature=None):
        """Calculate alpha_w - the water-absorption coefficient (WAC) in dB/m.

        :param frequency:
        :param temperature:
        :return: alpha_w
        """

        if temperature is None:
            temperature = cls.SPEED_OF_SOUND_IN_WATER

        f_T = 21.9 * 10 ** (6 - 1520 / (temperature + 273))  # temperature-dependent relaxation frequency
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T  # water attenuation coefficient

        return alpha_w

    @classmethod
    def wave_number(cls, frequency, temperature=None):

        wavelength = cls.wavelength(frequency, temperature)
        wave_number = 2*np.pi / wavelength
        return wave_number

    @classmethod
    def wavelength(cls, frequency, temperature=None):
        """Calculate the acoustic wavelength

        :param frequency: Acoustic frequency in kHz
        :param temperature: Temperature in deg C
        :return:
        """
        speed_of_sound = cls.speed_of_sound(temperature)
        frequency_Hz = frequency*1000
        wavelength = speed_of_sound / frequency_Hz
        return wavelength
