"""

References
----------
.. [1] Landers, M.N., Straub, T.D., Wood, M.S., and Domanski, M.M.,
   2016, Sediment acoustic index method for computing continuous
   suspended-sediment concentrations: U.S. Geological Survey Techniques
   and Methods, book 3, chap. C5, 63 p.,
   http://dx.doi.org/10.3133/tm3C5.

.. [2] Marczak, Wojciech, 1997, Water as a standard in the measurements of
   speed of sound in liquids: Journal of the Acoustical Society of America, v.
   102, no. 5, p. 2776−2779, accessed March 11, 2016, at
   http://scitation.aip.org/content/asa/journal/jasa/102/5/10.1121/1.420332.

.. [3] Moore, S.A., Le Coz, J., Hurther, D., and Paquier, A., 2013, Using
   multi-frequency acoustic attenuation to monitor grain size and concentration
   of suspended sediment in rivers: Journal of the Acoustical Society of
   America, v. 133, no. 4, p. 1959−1970, accessed March 11, 2016,
   http://dx.doi.org/10.1121/1.4792645.

.. [4] Thorne, P.D., and Meral, R., 2008, Formulations for the scattering
   properties of suspended sandy sediments for use in the application of
   acoustics to sediment transport processes: Continental Shelf Research, v.
   28, no. 2, p. 309–317, doi:10.1016/j.csr.2007.08.002.

"""

from scoupy.acousticsample import AcousticSample
from scoupy.sedimentsample import SedimentSample
from scoupy.sedimentsizedistribution import SedimentSizeDistribution
from scoupy.water import WaterProperties
