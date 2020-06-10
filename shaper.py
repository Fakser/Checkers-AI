import numpy as np

def shape_of_you(array):
    size = 0
    for var in array:
        var = np.array(var)
        if len(var.shape) == 1:
            size += shape_of_you(var)
        else:
            size += 1      
    return size

def use_this_gospel(array):
    mutations = []
    for var in array:
        var = np.array(var)
        if len(var.shape) >= 1:
            mutations.append(use_this_gospel(var))
        else:
            mutations.append(np.random.uniform(0,1))       
    return np.array(mutations)

def y4ndhi(array, size = None):
    if size == None:
        size = shape_of_you(array)
    
    #T1 = 1/np.math.sqrt(2*size)
    #T2 = 1/np.math.sqrt(2*np.math.sqrt(size))
    
    for i,var in enumerate(array):
        var = np.array(var)
        if len(var.shape) == 1:
            array[i] = y4ndhi(var,size)
        else:
            if np.random.uniform(0,1) <= 0.0001:
                array[i] = array[i] + np.random.uniform(-0.1, 0.1)    
    return np.array(array)

