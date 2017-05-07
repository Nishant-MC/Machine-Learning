import math

L_RATE = 0.1

raw_data = [ 
             (1600, 3, 330),
             (2400, 3, 369),
             (1416, 2, 232),
             (3000, 4, 540)
           ]

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


if __name__ == "__main__":
    w = [0,0,0]
    w_hist = [0] + w
    print(w)
    
    # Separate out first two features (sqft, # beds) for mean & stdev calculation
    sq_footage = [ raw_data[i][0] for i in range(len(raw_data)) ]
    num_bedrms = [ raw_data[i][1] for i in range(len(raw_data)) ]
    
    # Calculating mean & stdev for (sqft, # beds)
    sqft_mean, sqft_stdev = mean(sq_footage), stdev(sq_footage)
    beds_mean, beds_stdev = mean(num_bedrms), stdev(num_bedrms)

    # Create a normalized data list (dummy values for now)
    normalized_data = [ [] for i in range(len(raw_data)) ]

    # Calculating the normalized dataset
    for i in range(len(raw_data)):
        normalized_data[i].append( (raw_data[i][0] ))#- sqft_mean)/sqft_stdev )
        normalized_data[i].append( (raw_data[i][1] ))#- beds_mean)/beds_stdev )
        normalized_data[i].append( raw_data[i][2] ) # Don't normalize what you're predicting for

    # Gradient descent for 1 pass 
    passes = 0
    while passes < 1:
        w = gd_one_pass(w,normalized_data)
        passes += 1
        
    print( "Normalized: ", w, J(w, normalized_data) )
