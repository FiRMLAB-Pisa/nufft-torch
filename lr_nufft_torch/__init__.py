from numba import errors

try:
    from numba import cuda # otherwise numba throws an error
except:
    pass