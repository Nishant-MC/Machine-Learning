import math

# Returns the average of a list L of numbers
def mean(L):
    return sum(L)/len(L)

# Returns the (population*) standard deviation of a list L of numbers
def stdev(L):
    avg = mean(L)
    return math.sqrt( mean( [ (x - avg)**2 for x in L ] ) )

if __name__ == "__main__":

    print("Reading housing.txt data...")
    
    # Read the housing.txt data and format into a list of lists of ints
    raw_data = open("housing.txt","r").read().split("\n")[:-1]
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split(",")
        for j in range(len(raw_data[i])):
            raw_data[i][j] = int(raw_data[i][j])

    # Separate out 2 features (sqft, # beds) for mean & stdev calculation
    sq_footage = [ raw_data[i][0] for i in range(len(raw_data)) ]
    num_bedrms = [ raw_data[i][1] for i in range(len(raw_data)) ]
    
    # Calculating mean & stdev for each of the 3 features
    sqft_mean, sqft_stdev = mean(sq_footage), stdev(sq_footage)
    beds_mean, beds_stdev = mean(num_bedrms), stdev(num_bedrms)

    # Create a normalized data list (dummy values for now)
    normalized_data = [ [] for i in range(len(raw_data)) ]

    # Calculating the normalized dataset
    for i in range(len(raw_data)):
        normalized_data[i].append( (raw_data[i][0] - sqft_mean)/sqft_stdev )
        normalized_data[i].append( (raw_data[i][1] - beds_mean)/beds_stdev )
        normalized_data[i].append( raw_data[i][2] ) # Don't normalize what you predict for

    # Sanity check for the stats calculated
    print("SQFT stats (mean, stdev):", sqft_mean, sqft_stdev)
    print("BEDS stats (mean, stdev):", beds_mean, beds_stdev)

    # Writing normalized data to another file
    outfile = open("normalized.txt","w")
    for n in normalized_data:
        outfile.write( ",".join( [ str(d) for d in n ] ) + "\n")
    outfile.close()

    print("Data normalization complete. Check normalized.txt")
