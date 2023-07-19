# Python Takehome

Hi prospective Matrix AI Cloud Software Engineer! Here is your assignment.

Points are given for succinct but clear code, and when there are ambiguities, comments should be provided.

Using functional programming style is allowed. For Python, using the pep8 standard is encouraged. 

The challenges below will give you a few correct inputs and outputs, however we will be testing your functions against unseen inputs. 

So make sure you understand exactly the purpose of the function.

All code is to be submitted that works against Python 3.

Submit the code as separate `takehome.py` file.

## Decorators

Using only the `functools` module. Create a decorator called `prepost`.

This decorator can optionally take a `prefix` parameter.

The decorator must print `"start"` before calling the decorated function, and must print `"end"` after calling the decorated function.

The `prefix` must be prepended to the printed message `"start"` and `"end"` with a space inbetween.

The decorator is used like this:

```py
@prepost(prefix='outer')
@prepost(prefix='inner')
@prepost
def multiplication(x, y):
    print('middle')
    return x * y
    
multiplication(3, 4)
```

When the above code is executed, it must print this output:

```
outer start
inner start
 start
middle
 end
inner end
outer end
```

Please implement `prepost`:

```py
def prepost(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        # The decorator was used without arguments
        function = args[0]

        # wraps(function)
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

# Now let's test the decorator

@prepost(prefix='outer')
@prepost(prefix='inner')
@prepost
def multiplication(x, y):
    print('middle')
    return x * y

multiplication(3, 4)
```

## Removing Boundaries

Using only `numpy`, create a function that takes an array and a binary mask and produces a cropped array based on the binary mask.

```py
import numpy as np

def boundary_cropping(a, m):
    pass

a1 = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,1,0,1,1], [0,0,0,0,0]])
a2 = np.array([[ [0,0,0], [0,1,0], [0,1,0] ], [ [0,0,0], [0,1,0], [0,0,0] ], [ [0,0,0], [0,1,0], [0,0,0] ]])

print(boundary_cropping(a1, a1 != 0))
# [[1 0 1 1]]
print(boundary_cropping(a2, a2 != 0))
# [[[1] [1]] [[1] [0]] [[1] [0]]]
```

## Stream Shuffling

Using only `random`, implement the Fisher Yates shuffling algorithm that can work on an input stream.

```py
import random

input_stream = range(0, 100)
output_list = []

for (i, v) in enumerate(input_stream):
    pass
```

## Round Robin

Using only `collections`, implement a data structure that implements these methods:

* `__iter__`
* `__next__`
* `__getitem__`
* `__setitem__`
* `__delitem__`

This data structure is constructed from a list of values.

When enumerated, it must cycle through the values forever from the first value to the last value.

The list of values is also mutable, so the round robin structure continues to work even as you replace values or delete values.

```py
class RoundRobin():
    def __init__(self, l):
        pass
        
rr = RoundRobin(["a", "b", "c", "d"])

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
```

When the above code is executed, the printed output should be:

```
a
b
c
d
---
A
b
c
---
d
A
b
d
A
b
d
A
b
d
```

## Functional Arrays

Create a function that takes a lambda, a dimensions shape and the Numpy dtype, and produces an array.

```py
import numpy as np

def create_array_from_function(f, d, dtype=None):
    pass

print(create_array_from_function(lambda i,j: (i - j)**2, [4, 4]))
# [[0. 1. 4. 9.]
#  [1. 0. 1. 4.]
#  [4. 1. 0. 1.]
#  [9. 4. 1. 0.]]
```

## Acquiring Coordinates

Given an array and a step shape, return a list of coordinates based on each step.

```py
import itertools
import numpy as np

def coordinates_from_steps(a, s, dtype=int):
    pass

print(coordinates_from_steps(np.array([[1,2],[3,4]]), (1,1)))
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]

print(coordinates_from_steps(np.array([[1,2],[3,4]]), (1,2)))
# [[0 0]
#  [1 0]]
```

## Population Variance from Subpopulation Variance

Given a list of numpy arrays, where each array is a subpopulation and the entire list is the population, calculate the variance of the entire population from the variance of the subpopulations.

```py
import numpy as np

def pop_var_from_subpop_var(groups):
    pass

groups = [np.array([1,2,3,4]), np.array([5,6])]
print(pop_var_from_subpop_var(groups))
# 2.9166666666666665
```
