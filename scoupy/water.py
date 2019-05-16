"""This module contains the WaterProperties class definition.

WaterProperties contains methods for calculating properties of water for
acoustic processing.

"""

import numpy as np


class WaterProperties:
    """Calculates properties of water

    WaterProperties contains methods for calculating properties of water. All
    methods of WaterProperties are class so it is unnecessary to create an
    instance of the class. See Examples.

    Class Attributes
    ----------------
    SPEED_OF_SOUND_IN_WATER : float
        1442.5 m/s. The speed of sound in water used as a default when
        necessary.
    TEMPERATURE : float
        8.81187 deg C. Water temperature at which the speed of sound in water
        is ~1442.5 m/s (as calculated with `speed_of_sound()`).

    Methods
    -------
    speed_of_sound(temperature=None)
        Calculates the speed of sound in water
    water_absorption(frequency, temperature=None)
        Calculates the water absorption coefficient
    wavenumber(frequency, temperature=None)
        Calculates the wavenumber of an acoustic wave
    wavelength(frequency, temperature=None)
        Calculates the wavelength of an acoustic wave

    Examples
    --------
    >>> WaterProperties.speed_of_sound()
    1442.5000230670473

    """

    SPEED_OF_SOUND_IN_WATER = 1442.5  # m/s

    TEMPERATURE = 8.81187

    @classmethod
    def speed_of_sound(cls, temperature=TEMPERATURE):
        """Speed of sound in water

        Calculate the speed of sound in water in m/s.

        Parameters
        ----------
        temperature : float, optional
            Water temperature in deg C (the default is `TEMPERATURE`).

        Returns
        -------
        float
            Speed of sound in water in m/s.

        Notes
        -----
        The speed of sound in water is calculated using the fifth order
        described in [2]_.

        """

        speed_of_sound = 1.402385 * 10 ** 3 + 5.038813 * temperature - \
            (5.799136 * 10 ** -2) * temperature ** 2 + \
            (3.287156 * 10 ** -4) * temperature ** 3 - \
            (1.398845 * 10 ** -6) * temperature ** 4 + \
            (2.787860 * 10 ** -9) * temperature ** 5

        return speed_of_sound

    @classmethod
    def water_absorption(cls, frequency, temperature=TEMPERATURE):
        """Water absorption of an acoustic signal

        Calculates the water absorption of an acoustic signal in dB/m.

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz.
        temperature : float, optional
            Water temperature in deg C (the default is `TEMPERATURE`).

        Returns
        -------
        float
            Water absorption in dB/m.

        Notes
        -----
        Water absorption is calculated according to equation (8) of [1]_
        (under the assumption of freshwater rivers at shallow depths).

        References
        ----------
        """

        # temperature-dependent relaxation frequency
        f_T = 21.9 * 10 ** (6 - 1520 / (temperature + 273))
        # water attenuation coefficient
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T

        return alpha_w

    @classmethod
    def wavenumber(cls, frequency, temperature=TEMPERATURE):
        """Acoustic wavenumber

        Calculates the acoustic wavenumber in 1/m.

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz.
        temperature : float, optional
            Water temperature in deg C (the default is `TEMPERATURE`).

        Returns
        -------
        float
            Wave number in 1/m.

        """

        wavelength = cls.wavelength(frequency, temperature)
        wavenumber = 2*np.pi / wavelength
        return wavenumber

    @classmethod
    def wavelength(cls, frequency, temperature=None):
        """Acoustic wavelength

        Calculates the acoustic wavelength in m.

        Parameters
        ----------
        frequency : float
            Acoustic frequency in kHz.
        temperature : float
            Water temperature in deg C (the default is `TEMPERATURE`).

        Returns
        -------
        float
            Wavelength in m.

        """

        speed_of_sound = cls.speed_of_sound(temperature)
        frequency_Hz = frequency*1000
        wavelength = speed_of_sound / frequency_Hz
        return wavelength
