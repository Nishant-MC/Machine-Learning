import math

L_RATE = 0.4
PASS_LIMIT = 100

# Returns the average of a list L of numbers
def mean(L):
    return sum(L)/len(L)

# Returns the (population*) standard deviation of a list L of numbers
def stdev(L):
    avg = mean(L)
    return math.sqrt( mean( [ (x - avg)**2 for x in L ] ) )

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

# MSE (not sure if this is 100% correct formulation considering we decomposed into 2 problems)
def MSE(w1,w2,d1,d2):
    x = sum( [ (w1[0] + w1[1]*d1[i][0] + w1[2]*d1[i][1] - d1[i][2])**2 for i in range(len(d1)) ] )
    y = sum( [ (w2[0] + w2[1]*d2[i][0] + w2[2]*d2[i][1] - d2[i][2])**2 for i in range(len(d2)) ] )
    return (x+y)/(len(d1))

if __name__ == "__main__":
    # Read the YY.txt data and format into a list of lists of floats
    raw_data = open("YY.txt","r").read().split("\n")[1:]
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split(",")
        for j in range(len(raw_data[i])):
            raw_data[i][j] = float(raw_data[i][j])
    
    # Separate out each feature (column) for mean & stdev calculation
    east = [ raw_data[i][0] for i in range(len(raw_data)) ]
    south = [ raw_data[i][1] for i in range(len(raw_data)) ]
    west = [ raw_data[i][2] for i in range(len(raw_data)) ]
    north = [ raw_data[i][3] for i in range(len(raw_data)) ]
    latitude = [ raw_data[i][4] for i in range(len(raw_data)) ]
    longitude = [ raw_data[i][5] for i in range(len(raw_data)) ]

    # Calculating mean & stdev for each of the 4 features
    e_mean, e_stdev = mean(east), stdev(east)
    s_mean, s_stdev = mean(south), stdev(south)
    w_mean, w_stdev = mean(west), stdev(west)
    n_mean, n_stdev = mean(north), stdev(north)

    # Create normalized data lists (dummy values for now)
    normalized_data_lat = [ ]
    normalized_data_long = [ ]

    # Calculating the normalized dataset
    for i in range(len(raw_data)):

        normalized_data_lat.append( [ (raw_data[i][1] - s_mean)/s_stdev,
                                      (raw_data[i][3] - n_mean)/n_stdev,
                                       raw_data[i][4]
                                     ] )
        
        normalized_data_long.append( [ (raw_data[i][0] - e_mean)/e_stdev,
                                       (raw_data[i][2] - w_mean)/w_stdev,
                                        raw_data[i][5]
                                     ] )
        
    # Weight vectors will start zeroed out... y = w0 + w1*x1 + w2*x2
    lat_weights = [0,0,0]  # w0 = constant, w1 = south const, w2 = north const
    long_weights = [0,0,0] # w0 = constant, w1 = east const, w2 = west const

    # Doing gradient descent for both latitude and longitude (simultaneously but separately)
    passes = 0
    while passes < PASS_LIMIT:
        lat_weights = gd_one_pass(lat_weights, normalized_data_lat)
        long_weights = gd_one_pass(long_weights, normalized_data_long)
        passes += 1

    print("LEARNING RATE:", L_RATE,"\n")
    print("W VECTOR (LATITUDE) AFTER", passes, "PASSES: ", lat_weights)
    print("LOSS FUNCTION J(W):", J(lat_weights, normalized_data_lat),"\n")
    print("W VECTOR (LONGITUDE) AFTER", passes, "PASSES: ", long_weights)
    print("LOSS FUNCTION J(W):", J(long_weights, normalized_data_long),"\n")
    print("OVERALL MSE:", MSE(lat_weights,long_weights,normalized_data_lat,normalized_data_long))
