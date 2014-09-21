import numpy as np

num_resonators = 20

id_to_coordinate = {19:(0,0.5),
                    14:(0,1.5),
                    9 :(0,2.5),
                    4 :(0,3.5),
                    8 :(1,0.0),
                    3 :(1,1.0),
                    18:(1,2.0),
                    13:(1,3.0),
                    17:(2,0.5),
                    12:(2,1.5),
                    7 :(2,2.5),
                    2 :(2,3.5),
                    6 :(3,0.0),
                    1 :(3,1.0),
                    16:(3,2.0),
                    11:(3,3.0),
                    15:(4,0.5),
                    10:(4,1.5),
                    5 :(4,2.5),
                    0 :(4,3.5),}

# The shape of the following is 20,2
coordinate_array = np.array([id_to_coordinate[x] for x in range(num_resonators)])

nominal_frequencies = np.array([100,105,110,115,120,130,135,140,145,150,165,165.25,165.5,165.75,166,175,180,185,190,195])