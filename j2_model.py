# jpert model

import numpy as np
import matplotlib.pyplot as plt

def j2_acceleration(
          r,
          muE,
          Re,
          J2,
):

    rmag = np.linalg.norm(r)
    x = r[0]
    y = r[1]
    z = r[2]

    r2 = rmag**2
    z2 = z**2
    kJ2 = ( 1.5 * J2 * muE * (Re**2) ) / (rmag**5)

    aJ2x = -kJ2 * x * (1 - 5*z2/r2)
    aJ2y = -kJ2 * y * (1 - 5*z2/r2)
    aJ2z = -kJ2 * z * (3 - 5*z2/r2)

    a_J2 = [aJ2x, aJ2y, aJ2z]

    return a_J2


