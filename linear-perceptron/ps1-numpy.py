# Problem Set 1 Sample
# Nishant Mohanchandra
import pickle
import numpy
import sys
import os

# Global constants for unit testing
EMAIL_THRESH = 20
TRAIN_SIZE = 4000

# More constants for pass limit and how many "top" spam/non-spam words
PASS_LIMIT = 100
TOP_WORDS = 12


### Q3 + Q8: The perceptron algorithm!
# Perhaps pickle the data generated after Problems 1-2 and then simply read it from file...
# ...Generating the feature vectors + other information from scratch takes several seconds (too long)
"""
Each element of train_set is a 3-element list of of the form [ -1 or 1 , "Email body" , [Feature vector] ]
train_set[0][0] = Spam value of -1 or 1 (whether the e-mail at index 0 is not spam or spam)
train_set[0][1] = Body of e-mail at index 0
train_set[0][2] = Computed feature vector of e-mail at index 0
train_set[x][2][i] = Referencing the value of the i-th feature of the x-th e-mail
"""

# Perceptron training algorithm
def perceptron_train(train_set, pass_limit):

    # Initial setup
    weight_vector = numpy.zeros( len(train_set[0][2]) )
    number_of_mistakes = 0
    number_of_mistakes_historical = -1
    number_of_data_passes = 0

    # One element in the training set (e-mail 1108) is pathologically bad data.
    # This e-mail has no refined vocab words, so its feature vector is the zero vector.
    # Thus its mistake resulted in no update of the weight vector.
    # This is why I make a modification to the prediction rule; to take care of corner cases like this
    
    while number_of_mistakes != number_of_mistakes_historical and number_of_data_passes <= pass_limit:

        print("Pass",number_of_data_passes,"-",number_of_mistakes,"Mistakes")
        number_of_data_passes += 1
        number_of_mistakes_historical = number_of_mistakes
        prediction = 0

        # Iterating over all e-mails in the training set        
        for x in range(len(train_set)):

            # Calculating a prediction (spam / not spam)
            if numpy.dot( weight_vector, train_set[x][2] ) >= 0:
                prediction = 1                  # SPAM!
            else:
                prediction = -1                 # Not spam!

            # Checking whether our prediction matches the given label
            if train_set[x][0] == prediction:   # No mistake made!
                pass
            else:                               # Mistake!
                number_of_mistakes += 1
                weight_vector += ( train_set[x][2] * train_set[x][0]  ) # w = w + ( x(i) * y(i) )
            
            # Resetting prediction value for next e-mail
            prediction = 0
                
    return [ weight_vector , number_of_mistakes, number_of_data_passes ]


# Perceptron testing function
"""
Each element of validate_set is a 3-element list of of the form [ -1 or 1 , "Email body" , [Feature vector] ]
validate_set[0][0] = Spam value of -1 or 1 (whether the e-mail at index 0 is not spam or spam)
validate_set[0][1] = Body of e-mail at index 0
validate_set[0][2] = Computed feature vector of e-mail at index 0
validate_set[x][2][i] = Referencing the value of the i-th feature of the x-th e-mail
"""
def perceptron_test(weight_vector, validate_set):

    number_of_mistakes = 0
    prediction = 0

    # Iterating over all e-mails in the validation set
    for x in range(len(validate_set)):
        
        # Calculating a prediction (spam / not spam)
        if numpy.dot( weight_vector, validate_set[x][2] ) >= 0:
            prediction = 1                  # SPAM!
        else:
            prediction = -1                 # Not spam!

        if validate_set[x][0] == prediction:   # No mistake!
            pass
        else:                                  # Mistake!
            number_of_mistakes += 1
        
        # Resetting prediction value for next e-mail
        prediction = 0
        
    print(number_of_mistakes, "Mistakes for a", len(validate_set), "member dataset.")

    # Avoiding division by zero for an empty validation set when N=5000
    if len(validate_set) > 0:
        return ( number_of_mistakes / len(validate_set) )
    else:
        return 0.0


# Main code
def main():

    global TRAIN_SIZE
    global EMAIL_THRESH
    global PASS_LIMIT
    global TOP_WORDS

    # Support for command line arguments to overwrite default values given at the top of the code
    # Look for sys.argv on Python 3's reference for more information
    if ( len(sys.argv) == 5 ):
        TRAIN_SIZE = int(sys.argv[1].split("=")[1])
        EMAIL_THRESH = int(sys.argv[2].split("=")[1])
        PASS_LIMIT = int(sys.argv[3].split("=")[1])
        TOP_WORDS = int(sys.argv[4].split("=")[1])
        print("Arguments accepted: N =", TRAIN_SIZE, "X =", EMAIL_THRESH,
              "PL =", PASS_LIMIT, "TW =", TOP_WORDS)
    else:
        print("Command line arguments not provided in expected format.")
        print("N =", TRAIN_SIZE, "X =", EMAIL_THRESH, "PL =", PASS_LIMIT, "TW =", TOP_WORDS)


    ### Q1: Making the training and validation sets
    
    # Getting the raw training data and test data
    data = open("spam_train.txt","r").readlines()
    train_set = data[:TRAIN_SIZE]
    validate_set = data[TRAIN_SIZE:]
    test_set = open("spam_test.txt","r").readlines()

    # Some preprocessing for the train, validate and test sets (separating label, slicing off newline char)
    for x in range(len(train_set)):
        train_set[x] = [ int(train_set[x][0]), train_set[x][2:-1] ]
    for y in range(len(validate_set)):
        validate_set[y] = [ int(validate_set[y][0]), validate_set[y][2:-1] ]
    for z in range(len(test_set)):
        test_set[z] = [ int(test_set[z][0]), test_set[z][2:-1] ]

    print("Generating full vocabulary...")

    ### Q2: Build a vocabulary list.

    # Populating the initial dictionary; going word by word, e-mail by e-mail in the training set
    vocab_dictionary = {}

    # If we don't already have a pre-existing dictionary to load...
    if not os.path.exists(str(TRAIN_SIZE)+"-vocab.pk"):

        for x in range(len(train_set)):
            
            temp_list = train_set[x][1].split(" ")

            # Each word fits into one of three cases:
            # (1) Fresh addition (Make a fresh dictionary entry for it)
            # (2) Already there, 1st appearance in this e-mail (increment "count", append x to "appears")
            # (3) Already there and already appeared in the e-mail under consideration (do nothing)
            
            for word in temp_list:
                if word not in vocab_dictionary.keys():
                    vocab_dictionary[word] = { "count": 1,
                                               "appears" : [x] }
                elif word in vocab_dictionary.keys() and x not in vocab_dictionary[word]["appears"]:
                    vocab_dictionary[word]["count"] += 1
                    vocab_dictionary[word]["appears"].append(x)
                else:
                    pass

        # Dumping our work to a file so we don't have to make a dict for this N again
        outfile = open( str(TRAIN_SIZE)+"-vocab.pk", "wb" )
        pickle.dump(vocab_dictionary, outfile)
        outfile.close()

    # If we've run the program for this given N before...
    else:
        
        infile = open( str(TRAIN_SIZE)+"-vocab.pk", "rb" )
        vocab_dictionary = pickle.load(infile)
        infile.close()

    print("Generating refined vocabulary...")
        
    # Stripping words with count < thresh from the refined dictionary
    refined_vocab = {}
    vector_number = 0        

    # If we don't already have a pre-existing dictionary to load...
    if not os.path.exists(str(TRAIN_SIZE)+"-"+str(EMAIL_THRESH)+"-Rvocab.pk"):

        for word in vocab_dictionary.keys():
            if  vocab_dictionary[word]["count"] >= EMAIL_THRESH:
                refined_vocab[word] = vocab_dictionary[word]
                refined_vocab[word]["id"] = vector_number
                vector_number += 1

        # Dumping our work to a file so we don't have to make a dict for this N and X again
        outfile = open( str(TRAIN_SIZE)+"-"+str(EMAIL_THRESH)+"-Rvocab.pk", "wb" )
        pickle.dump(refined_vocab, outfile)
        outfile.close()
    
    # If we've run the program for this given N before...
    else:

        infile = open( str(TRAIN_SIZE)+"-"+str(EMAIL_THRESH)+"-Rvocab.pk", "rb" )
        refined_vocab = pickle.load(infile)
        infile.close()

            
    print( "Original vocab: ", len(vocab_dictionary), "\nRefined vocab: ", len(refined_vocab), "\n" )
    print( "Generating feature vectors..." )


    ### Generate feature vectors for every e-mail
###
###
    # Adding an entry for the n-dimensional feature vector for every training e-mail (initialized to all 0s)
    for x in range(len(train_set)):

        # Creating a vector for expressing e-mail features
        train_set[x].append( numpy.zeros( len(refined_vocab) ) )

        # 0 = not spam. I'm changing it to -1 = not spam (for feature vector analysis later)
        if train_set[x][0] == 0:
            train_set[x][0] = -1

        # Splitting email body by space and then updating the feature vector as necessary
        temp_list = train_set[x][1].split(" ")
        for word in temp_list:
            if word in refined_vocab.keys():
                train_set[x][2][ refined_vocab[word]["id"] ] = 1
###
###
    print( "Feature vectors for training set computed.\n" )


    # Time to train the classifier (create a W vector)

    """
    train_set[0] = email label (-1 or 1)
    train_set[1] = email body (string)
    train_set[2] = email feature vector (numpy.array)
    training_info[0] = weight vector (W)
    training_info[1] = number of mistakes (k)
    training_info[2] = number of data passes (iter)
    """
    training_info = perceptron_train(train_set,PASS_LIMIT)

###
###
    # Adding an entry for the n-dimensional feature vector for every validation e-mail (initialized to all 0s)
    for x in range(len(validate_set)):

        validate_set[x].append( numpy.zeros( len(refined_vocab) ) )

        # 0 = not spam. I'm changing it to -1 = not spam
        if validate_set[x][0] == 0:
            validate_set[x][0] = -1

        # Splitting email body by space and then updating the feature vector as necessary
        temp_list = validate_set[x][1].split(" ")
        for word in temp_list:
            if word in refined_vocab.keys():
                validate_set[x][2][ refined_vocab[word]["id"] ] = 1
###
###
    print( "\nFeature vectors for validation set computed." )

###
###
    # Adding an entry for the n-dimensional feature vector for every spam_test.txt e-mail (initialized to all 0s)
    for x in range(len(test_set)):

        test_set[x].append( numpy.zeros( len(refined_vocab) ) )

        # 0 = not spam. I'm changing it to -1 = not spam
        if test_set[x][0] == 0:
            test_set[x][0] = -1

         # Splitting email body by space and then updating the feature vector as necessary
        temp_list = test_set[x][1].split(" ")
        for word in temp_list:
            if word in refined_vocab.keys():
                test_set[x][2][ refined_vocab[word]["id"] ] = 1
###
###
    print( "Feature vectors for test set computed.\n" )


    ### Q4 + Q9: Finding out the validation and test error for our computed weight vector

    # Running perceptron_test() on both the training data (should return 0 errors)...
    # as well the validation data (should return some errors... empirically, around 2%)
    # AND the test data (should return an error % similar to that of the validation set)

    t_validate_info = perceptron_test(training_info[0], train_set)
    validate_info = perceptron_test(training_info[0] , validate_set)
    test_info = perceptron_test(training_info[0] , test_set)
    #print( "Classification error fraction for training data:", t_validate_info * 100, "% for", len(train_set), "items" )
    #print( "Classification error fraction for validation data:", validate_info * 100, "% for", len(validate_set), "items" )
    #print( "Classification error fraction for test set data:", test_info * 100, "% for", len(test_set), "items" )

    ### Q5: Finding the predictive words (#TW top spam-correlated and #TW top non-spam correlated words)

    weight_dictionary = {}

    for x in range(len(training_info[0])):
        if training_info[0][x] not in weight_dictionary.keys():
            weight_dictionary[ training_info[0][x] ] = [x]
        else:
            weight_dictionary[ training_info[0][x] ].append(x)

    weight_list = sorted( list( weight_dictionary.keys() ) )
    spam_correlated_words = []
    spam_inverse_correlated_words = []

    counter = 0
    while counter < TOP_WORDS:
        for y in range( int(weight_list[-1]), 0, -1 ):
            if y in weight_dictionary.keys() and len( weight_dictionary[y] ) > 0:
                while len( weight_dictionary[y] ) > 0:
                    spam_correlated_words.append( [weight_dictionary[y].pop(),y] )
                    counter += 1
                    if counter == TOP_WORDS:
                        break
                if counter == TOP_WORDS:
                    break

    counter = 0
    while counter < TOP_WORDS:
        for y in range( int(weight_list[0]), 0, 1 ):
            if y in weight_dictionary.keys() and len( weight_dictionary[y] ) > 0:
                while len( weight_dictionary[y] ) > 0:
                    spam_inverse_correlated_words.append( [weight_dictionary[y].pop(),y] )
                    counter += 1
                    if counter == TOP_WORDS:
                        break
                if counter == TOP_WORDS:
                    break

    print( "\nTOP SPAM CORRELATED WORDS:" )
    for number in spam_correlated_words:
        for word in refined_vocab:
            if refined_vocab[word]["id"] == number[0]:
                print(word, "(", number[1], ")")
                continue

    print("\n")

    print( "\nWORDS INVERSELY CORRELATED WITH SPAM:" )
    for number in spam_inverse_correlated_words:
        for word in refined_vocab:
            if refined_vocab[word]["id"] == number[0]:
                print(word, "(", number[1], ")")
                continue

    print()
    print( "Classification error fraction for training data:", t_validate_info)
    print( "Classification error fraction for validation data:", validate_info)
    print( "Classification error fraction for test set data:", test_info)
    print( "Perceptron algorithm total pass count for training:", training_info[2]-1)

    #return training error %, validation error %, test error %, training pass count         
    return t_validate_info, validate_info, test_info, training_info[2]

if __name__ == "__main__":
    main()
