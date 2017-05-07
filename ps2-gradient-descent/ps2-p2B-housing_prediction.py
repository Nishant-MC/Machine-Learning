"""
import pylab
"""

L_RATE = 0.5
PASS_LIMIT = 200

# Partial deriv w.r.t. w0
def dw0(w,data):
    return sum( [ w[0] + (w[1]*data[i][0]) + (w[2]*data[i][1]) - data[i][2] 
                for i in range(len(data)) ] ) / len(data)

# Partial deriv w.r.t. w1
def dw1(w,data):
    return sum( [ (w[0]*data[i][0]) + (w[1]*(data[i][0]**2)) + (w[2]*data[i][1]*data[i][0]) - (data[i][2]*data[i][0])
                for i in range(len(data)) ] ) / len(data)

# Partial deriv w.r.t. w2
def dw2(w,data):
    return sum( [ (w[0]*data[i][1]) + (w[1]*data[i][0]*data[i][1]) + (w[2]*(data[i][1]**2)) - (data[i][2]*data[i][1]) 
                for i in range(len(data)) ] ) / len(data)

# Loss function
def J(w,data):
    return sum( [ (w[0] + (w[1]*data[i][0]) + (w[2]*data[i][1]) - data[i][2])**2
                for i in range(len(data)) ] ) / (2*len(data))

# Gradient descent
def gd_one_pass(w,data):
    return [ w[0] - L_RATE*dw0(w,data), w[1] - L_RATE*dw1(w,data), w[2] - L_RATE*dw2(w,data) ]


if __name__ == "__main__":
    # Read the normalized.txt data and format into a list of lists of ints
    raw_data = open("normalized.txt","r").read().split("\n")[:-1]
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split(",")
        for j in range(len(raw_data[i])):
            raw_data[i][j] = float(raw_data[i][j])
    
    # Using constants to normalize our test input (1650 sq ft, 3 bedroom)
    # Values for mean and stdev calculated from P2A
    input_sqft_norm = (1650.0 - 2000.6808510638298) / 786.2026187430467
    input_beds_norm = (3.0 - 3.1702127659574466) / 0.7528428090618782
    
    # Weight vector will start zeroed out... y = w0 + w1*x1 + w2*x2
    weights = [0,0,0] # w0 = price constant, w1 = sq footage, w2 = beds

    # Doing gradient descent
    passes = 0
    while passes < PASS_LIMIT:
        weights = gd_one_pass(weights, raw_data)
        print(passes, weights)
        passes += 1

    print("LEARNING RATE:", L_RATE)
    print("W VECTOR AFTER", passes, "PASSES: ", weights)
    print("PRICE PREDICTION FOR 1650 SQFT, 3 BEDS: ", weights[0]+ weights[1]*input_sqft_norm + weights[2]*input_beds_norm)
    print()

    """
    # Doing a sanity check for 2-var linear regression by tossing bedroom data...
    # ...and using only sqft data to do a univariate un-normalized regression
    # The results should be close to the 2-var case
    test_data = open("housing.txt","r").read().split("\n")[:-1]
    for i in range(len(test_data)):
        test_data[i] = test_data[i].split(",")
        for j in range(len(test_data[i])):
            test_data[i][j] = int(test_data[i][j])
    sq_footage = [ test_data[i][0] for i in range(len(test_data)) ]
    houseprice = [ test_data[i][2] for i in range(len(test_data)) ]
    # Quick regression test with just sq footage vs. price (Gives ~$293,237 for 1650 sqft house)
    pylab.scatter(sq_footage,houseprice)
    # Polyfit returns a (slope,intercept) pair which best fits the data (least squares)
    (m,b) = pylab.polyfit(sq_footage,houseprice,1)
    # Polyval returns [ y-values ] which solve for y = mx + b for some list of x-coords
    yc = pylab.polyval([m,b],sq_footage)
    # Plotting the regression line
    pylab.plot(sq_footage,yc)
    pylab.show()
    """
