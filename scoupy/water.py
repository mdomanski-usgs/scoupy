"""This module contains the WaterProperties class definition.

WaterProperties contains methods for calculating properties of water for acoustic processing.

"""

import numpy as np


class WaterProperties:
    """Calculates properties of water

    WaterProperties contains methods for calculating properties of water. All methods of WaterProperties are class
    so it is unnecessary to create an instance of the class. See Examples.

    Class Attributes
    ----------------
    SPEED_OF_SOUND_IN_WATER : float
        1442.5 m/s. The speed of sound in water used as a default when necessary.
    TEMPERATURE : float
        8.81187 deg C. Water temperature at which the speed of sound in water is ~1442.5 m/s (as calculated with
        `speed_of_sound()`).

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
    TEMPERATURE = 8.81187  # deg C. water temperature at which the speed of sound is equal to SPEED_OF_SOUND_IN_WATER

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
        speed_of_sound : float
            Speed of sound in water in m/s.

        Notes
        -----
        The speed of sound in water is calculated using the fifth order described in [1]_.

        References
        ----------
        .. [1] Marczak, Wojciech, 1997, Water as a standard in the measurements of speed of sound in liquids: Journal of
        the Acoustical Society of America, v. 102, no. 5, p. 2776−2779, accessed March 11, 2016, at
        http://scitation.aip.org/content/asa/jour-nal/jasa/102/5/10.1121/1.420332.

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
        alpha_w : float
            Water absorption in dB/m.

        Notes
        -----
        Water absorption is calculated according to equation (8) of [1]_ with the assumption of freshwater rivers at
        shallow depths.

        References
        ----------
        .. [1] Landers, M.N., Straub, T.D., Wood, M.S., and Domanski, M.M., 2016, Sediment acoustic index method for
        computing continuous suspended-sediment concentrations: U.S. Geological Survey Techniques and Methods, book 3,
        chap. C5, 63 p., http://dx.doi.org/10.3133/tm3C5.

        """

        f_T = 21.9 * 10 ** (6 - 1520 / (temperature + 273))  # temperature-dependent relaxation frequency
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T  # water attenuation coefficient

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
        wavenumber : float
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
        wavelength : float
            Wavelength in m.

        """

        speed_of_sound = cls.speed_of_sound(temperature)
        frequency_Hz = frequency*1000
        wavelength = speed_of_sound / frequency_Hz
        return wavelength
