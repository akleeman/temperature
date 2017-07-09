import numpy as np


def pot_of_water(radius_m, height_m):

    area = np.pi * radius_m ** 2
    volume = height_m * area * 1000  # volume in litres

    return {'c_v': 4185.5,  # J/(kg K)
            # litres
            'volume': volume,
            # low speed flow of air over surface then less heat loss
            # over surface of the pot
            'h': 100 * area + 2 * (2 * np.pi * radius_m + area),
            # It takes a minute and a half for the stove to heat up.
            'k': 0.17,
            # The pot starts at 30 C
            'u_0' : 303.,
            # the ambient temperature is 25 C
            'u_env': 298.,
            'wattage': 800,
            'init_time_off': 80.
     }
