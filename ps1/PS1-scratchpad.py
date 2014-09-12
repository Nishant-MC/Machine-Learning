### PROBLEM 3: The perceptron algorithm! (Uh oh...)

# Perhaps pickle the data generated after Problems 1-2 and then simply read it from file...
# ...Generating the feature vectors + other information from scratch takes several seconds (too long)

# Some utility functions for the classifier (dot & scalar products)
def vector_dot_product(v1, v2):
    if len(v1) != len(v2):
        return False
    else:
        dot_product = 0
        for x in range(len(v1)):
            dot_product += (v1[x] * v2[x])
        return dot_product

def vector_scalar_product(v1, s):
    for x in range(len(v1)):
        v1[x] *= s
    return v1


# Vector addition calculator
def vector_addition(v1, v2):
    if len(v1) != len(v2):
        return False
    else:
        for x in range(len(v1)):
            v1[x] += v2[x]
    return v1


print( vector_dot_product ( [1,2,3] , [1,2,3] ) )
print( vector_scalar_product ( [1,2,3] , 5 ) )
print( vector_addition ( [1,2,3] , [4,5,6] ) )
