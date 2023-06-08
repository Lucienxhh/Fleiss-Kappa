import numpy as np
from fleiss_kappa import fleiss_kappa, transform, tranform2

def test():
    # Example data provided by wikipedia https://en.wikipedia.org/wiki/Fleiss_kappa
    data = np.array([
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7]
    ])

    fleiss_kappa(data)

    # need transform
    rater1 = [1, 2, 2, 1, 2, 2, 1, 1, 3, 1, 2, 2]
    rater2 = [1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 1]
    rater3 = [1, 2, 2, 1, 3, 3, 3, 2, 1, 2, 3, 1]

    data = transform(rater1, rater2, rater3)
    fleiss_kappa(data)

    # The first row indicates that both rater 1 and 2 rated as category 0, this case occurs 8 times.
    # need transform2
    weighted_data = [
        [0, 0, 8],
        [0, 1, 2],
        [0, 2, 0],
        [1, 0, 0],
        [1, 1, 17],
        [1, 2, 3],
        [2, 0, 0],
        [2, 1, 5],
        [2, 2, 15]
    ]
    data = tranform2(weighted_data)
    fleiss_kappa(data)

test()
