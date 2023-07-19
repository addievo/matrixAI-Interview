## Decorators
# 1. Write a decorator that prints a function with arguments passed to it.
from functools import wraps
def prepost(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        # The decorator was used without arguments
        function = args[0]

        @wraps(function)
        def wrapper(*f_args, **f_kwargs):
            print("start")
            result = function(*f_args, **f_kwargs)
            print("end")
            return result
        return wrapper
    else:
        # The decorator was used with arguments
        prefix = kwargs.get('prefix')

        def real_decorator(function):
            @wraps(function)
            def wrapper(*f_args, **f_kwargs):
                print(f"{prefix} start")
                result = function(*f_args, **f_kwargs)
                print(f"{prefix} end")
                return result
            return wrapper
        return real_decorator

# Testing

print("Testing Task 1")
@prepost(prefix='outer')
@prepost(prefix='inner')
@prepost
def multiplication(x, y):
    print('middle')
    return x * y

multiplication(3, 4)

## Removing Boundaries

import numpy as np


def boundary_cropping(a, m):
    # Get the indices where m is True
    indices = np.where(m)

    # Get the min and max indices along each axis
    min_indices = np.min(indices, axis=1)
    max_indices = np.max(indices, axis=1) + 1  # add 1 because slicing in numpy is exclusive at the end

    # Use slicing to get the cropped array
    slices = [slice(min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices)]
    return a[tuple(slices)].tolist()


# Testing the boundary_cropping function

a1 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]])
a2 = np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

print()
print()
print("Testing Task 2")

print(boundary_cropping(a1, a1 != 0))
print(boundary_cropping(a2, a2 != 0))


## Stream Shuffling

import random

def fisher_yates_shuffle(input_stream):
    output_list = list(input_stream)

    for i in range(len(output_list) - 1, 0, -1):
        j = random.randint(0, i)
        output_list[i], output_list[j] = output_list[j], output_list[i]

    return output_list


# Testing the fisher_yates_shuffle function

input_stream = range(0, 100)
output_list = fisher_yates_shuffle(input_stream)

# Verifying output list contains same elements as input list
print()
print()
print("Testing Task 3")

print(sorted(output_list) == list(input_stream))

## Round Robin

from collections import deque

class RoundRobin():
    def __init__(self, l):
        self.deque = deque(l)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.deque:
            raise StopIteration
        item = self.deque.popleft()
        self.deque.append(item)
        return item

    def __getitem__(self, index):
        return self.deque[index]

    def __setitem__(self, index, value):
        self.deque[index] = value

    def __delitem__(self, index):
        del self.deque[index]

rr = RoundRobin(["a", "b", "c", "d"])

print()
print()
print("Testing Task 4")
for (_, v) in zip(range(4), rr):
    print(v)

print('---')

rr[0] = 'A'

for (_, v) in zip(range(3), rr):
    print(v)

print('---')

del rr[2]

for (_, v) in zip(range(10), rr):
    print(v)

## Functional Arrays
import numpy as np

def create_array_from_function(f, d, dtype=None):
    #creating array of given shape, filled with indices
    indices = np.indices(d)

    #apply func to indices
    result = f(*indices)

    #convert to given dtype

    if dtype is not None:
        result = result.astype(dtype)

    return result.tolist()

# Testing the create_array_from_function function

print()
print()
print("Testing Task 5")

print(create_array_from_function(lambda i,j: (i - j)**2, [4, 4]))
# [[0. 1. 4. 9.]
#  [1. 0. 1. 4.]
#  [4. 1. 0. 1.]
#  [9. 4. 1. 0.]]

## Acquiring Coordinates

import itertools
import numpy as np

def coordinates_from_steps(a, s, dtype=int):
     # use np.mgrid to get the indices of the array
     indices = np.mgrid[tuple(slice(0, dim, step) for dim, step in zip(a.shape, s))]

     # reshape the indices to get the coordinates

     coordinates = indices.transpose().reshape(-1, indices.shape[0])

     coordinates = coordinates.astype(dtype)

     return coordinates

print()
print()
print("Testing Task 6")

print(coordinates_from_steps(np.array([[1,2],[3,4]]), (1,1)))
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]

print(coordinates_from_steps(np.array([[1,2],[3,4]]), (1,2)))
# [[0 0]
#  [1 0]]

## Population Variance from Subpopulation Variance

import numpy as np

def pop_var_from_subpop_var(groups):
    avg = np.mean(np.concatenate(groups))

    #var
    g_var = [np.var(group, ddof=0) for group in groups]

    #size
    size = np.sum([len(group) for group in groups])

    #total var
    t_var = sum(group.size* (g_var[i]+ (group.mean()-avg)**2)for i, group in enumerate(groups)) / size

    return t_var

groups = [np.array([1,2,3,4]), np.array([5,6])]

print()
print()
print("Testing Task 7")
print(pop_var_from_subpop_var(groups))
# 2.9166666666666665

