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

def get_biases(array):
    biases = []
    for var in array:
        var = np.array(var)
        if len(var.shape) >= 1:
            biases.append(get_biases(var))
        else:
            biases.append(np.random.uniform(0,1))       
    return np.array(biases)

def evolve(weights, biases):
    
    for i,var in enumerate(weights):
        var = np.array(var)
        if len(var.shape) >= 1:
            weights[i], biases[i] = evolve(var,biases[i])
        else:
            biases[i] = biases[i] * np.math.exp(0.001*np.random.uniform(0,1))
            weights[i] = weights[i] + biases[i] * np.random.uniform(0,1)    
    return np.array(weights), np.array(biases)

def to_list(array):
    array = np.array(array)
    for i in range(len(array)):
        var = np.array(array[i])
        if len(var.shape) >= 1:
            array[i] = to_list(var)
        else:
            array[i] = array[i] 
    return array.tolist()

