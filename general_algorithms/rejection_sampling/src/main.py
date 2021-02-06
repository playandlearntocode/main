'''
Rejection Sampling example in Python
https://playandlearntocode.com
'''
import math
from random import Random

# sample non-uniform discrete probability distribution (target distribution):
probability_distribution = [
    {"value": 1, "probability": 0.1},
    {"value": 5, "probability": 0.7},
    {"value": 8, "probability": 0.2}
]

def draw_from_non_uniform_distribution(probability_distribution):
    '''
    Draws one observation from the provided probability distribution using Rejection Sampling

    :param probability_distribution:
    :return: one of the X values of the provided probability distribution
    '''
    member_count = len(probability_distribution)
    step_size = 1.00 / (member_count * 1.00)

    accept = False
    r_value = 0
    r_probability = 0

    while accept == False:
        # generate r_temp and use it to determine R:
        random_generator = Random()
        r_temp = random_generator.random()

        bin = math.ceil(r_temp / step_size)
        binned_object = probability_distribution[bin - 1]
        r_value = binned_object['value']
        r_probability = binned_object['probability']

        # now, after we have R (r_value), draw S:
        s = random_generator.random()

        # accept or reject this observation:
        if (s <= r_probability):
            accept = True
        else:
            accept = False

    return r_value

print('****REJECTION SAMPLING PROGRAM STARTED****')

# histogram of all sampled numbers:
bins = {}

# total number of observations to draw:
total_draws = 1000

# draw now:
for i in range(total_draws):

    r_value = draw_from_non_uniform_distribution(probability_distribution)

    if (bins.get(r_value) != None):
        bins[r_value] += 1
    else:
        bins[r_value] = 1

# output the resulting histogram:
print(bins)

print('****REJECTION SAMPLING PROGRAM COMPLETED****')