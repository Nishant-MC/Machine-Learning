# Problem Set 1 Script
# Nishant Mohanchandra


### PROBLEM 1: Making the training and validation sets

# Getting the raw data
data = open("spam_train.txt","r").readlines()
train_set = data[:4000]
validate_set = data[4000:]

# Some preprocessing for the train and validate sets
for x in range(len(train_set)):
    train_set[x] = [ int(train_set[x][0]), train_set[x][2:-1] ]
for y in range(len(validate_set)):
    validate_set[y] = [ int(validate_set[y][0]), validate_set[y][2:-1] ]

###########################################
# Checking correctness...
"""
print(data[0])
print(train_set[0])
print(data[4000])
print(validate_set[0])
"""
###########################################

print("Generating refined vocabulary...")



### PROBLEM 2a: Build a vocabulary list.
# Rule: If a word appears in less than 30 distinct e-mails, ignore it.

# Populating the initial dictionary; going word by word, e-mail by e-mail in the training set
vocab_dictionary = {}

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

# Stripping words with count < 30 from the refined dictionary
refined_vocab = {}
vector_number = 0

for word in vocab_dictionary.keys():
    if  vocab_dictionary[word]["count"] > 30:
        refined_vocab[word] = vocab_dictionary[word]
        refined_vocab[word]["id"] = vector_number
        vector_number += 1
        
#############################################
# Checking correctness...
"""
for key in refined_vocab.keys():
    print( key, refined_vocab[key]["count"] )
"""
#############################################

print( "Original vocab: ", len(vocab_dictionary), "\nRefined vocab: ", len(refined_vocab), "\n" )
print( "Generating feature vectors..." )




### PROBLEM 2b: Generate feature vectors for every e-mail (2331-D acc. to the refined vocab)

# Adding an entry for the n-dimensional feature vector for every training e-mail (initialized to all 0s)
for x in range(len(train_set)):

    train_set[x].append( [0] * len(refined_vocab) )

    # 0 = not spam. I'm changing it to -1 = not spam (for feature vector analysis later)
    if train_set[x][0] == 0:
        train_set[x][0] = -1

    temp_list = train_set[x][1].split(" ")
    
    for word in refined_vocab.keys():
        if word in temp_list:
            train_set[x][2][ refined_vocab[word]["id"] ] = 1
            #print(word)

#print(train_set[0])
print( "Feature vectors for training set computed.\n" )




### PROBLEM 3: The perceptron algorithm! (Uh oh...)

# Perhaps pickle the data generated after Problems 1-2 and then simply read it from file...
# ...Generating the feature vectors + other information from scratch takes several seconds (too long)

# Some utility functions for the classifier (dot & scalar products, vector addition)

# Vector dot product 
def vector_dot_product(v1, v2):
    if len(v1) != len(v2):
        return False
    else:
        dot_product = 0
        for x in range(len(v1)):
            dot_product += (v1[x] * v2[x])
        return dot_product

# Vector scalar product (doesn't change v1 values)
def vector_scalar_product(v1, s):
    result = []
    for x in range(len(v1)):
        result.append( v1[x] * s )
    return result

# Vector addition 
def vector_addition(v1, v2):
    if len(v1) != len(v2):
        return False
    else:
        for x in range(len(v1)):
            v1[x] += v2[x]
    return v1

"""
Each element of train_set is a 3-element list of of the form [ -1 or 1 , "Email body" , [Feature vector] ]

train_set[0][0] = Boolean spam value (whether the 1st e-mail is spam or not)
train_set[0][1] = E-mail body of 1st e-mail
train_set[0][2] = Computer feature vector of 1st e-mail

train_set[x][2][i] = Referencing the value of the i-th feature of the x-th mail 
"""

# Perceptron training algorithm
def perceptron_train(train_set):
    weight_vector = [0] * len(train_set[0][2]) # len(train_set[0][2]) = 2331, the length of the refined vocab

    number_of_mistakes = 0
    number_of_mistakes_historical = -1
    number_of_data_passes = 0

    # One element in the training set (e-mail 1108) is pathologically bad data.
    # This e-mail has no refined vocab words appearing in it, so its feature vector is the zero vector.
    # Thus its mistake resulted in no update of the weight vector.
    # This shows that the >= 0 condition for the dot product is REALLY IMPORTANT. ( >0 results in an infinite loop)
    # You know, besides like, the math reasons, and stuff...
    
    while number_of_mistakes != number_of_mistakes_historical:

        print("Pass",number_of_data_passes,"-",number_of_mistakes,"Mistakes")
        number_of_data_passes += 1
        number_of_mistakes_historical = number_of_mistakes
        
        for x in range(len(train_set)):
            prediction = 0

            if vector_dot_product( weight_vector, train_set[x][2] ) >= 0:
                prediction = 1                  # 'Tis SPAM!
            else:
                prediction = -1                 # 'Tis not spam!

            if train_set[x][0] == prediction:   # No mistake!
                prediction = 0  
            else:                               # Mistake!
                prediction = 0  
                number_of_mistakes += 1
                weight_vector = vector_addition ( weight_vector, vector_scalar_product( train_set[x][2], train_set[x][0] ) )
                
                # Debug effort: seeing what caused the (erstwhile) infinite loop
                # if number_of_data_passes > 15:
                #     print("Data point",x,"caused the mistake")

    return [ weight_vector , number_of_mistakes, number_of_data_passes ]


# Perceptron testing function
def perceptron_test(weight_vector, validate_set):
    number_of_mistakes = 0

    for x in range(len(validate_set)):
        prediction = 0
        if vector_dot_product( weight_vector, validate_set[x][2] ) >= 0:
            prediction = 1                  # 'Tis SPAM!
        else:
            prediction = -1                 # 'Tis not spam!

        if validate_set[x][0] == prediction:   # No mistake!
            prediction = 0  
        else:                                  # Mistake!
            prediction = 0  
            number_of_mistakes += 1

    print(number_of_mistakes, "Mistakes for a", len(validate_set), "member dataset.")
    return ( number_of_mistakes / len(validate_set) )





# Time to train the classifier (create a W vector)

"""
training_info[0] = weight vector (W)
training_info[1] = number of mistakes (k)
training_info[2] = number of data passes (iter)
"""
training_info = perceptron_train(train_set)


# Adding an entry for the n-dimensional feature vector for every validation e-mail (initialized to all 0s)
for x in range(len(validate_set)):

    validate_set[x].append( [0] * len(refined_vocab) )

    # 0 = not spam. I'm changing it to -1 = not spam
    if validate_set[x][0] == 0:
        validate_set[x][0] = -1

    temp_list = validate_set[x][1].split(" ")
    
    for word in refined_vocab.keys():
        if word in temp_list:
            validate_set[x][2][ refined_vocab[word]["id"] ] = 1

print( "\nFeature vectors for validation set computed.\n" )




### PROBLEM 4: Finding out the validation error for our computed weight vector

# Running perceptron_test() on both the training data (should return 0 errors) + the validation data (should return some errors)
t_validate_info = perceptron_test(training_info[0], train_set)
validate_info = perceptron_test(training_info[0] , validate_set)

print( "Classification error fraction for training data:", t_validate_info * 100, "% for", len(train_set), "items" )
print( "Classification error fraction for validation data:", validate_info * 100, "% for", len(validate_set), "items" )



### PROBLEM 5: Finding the predictive words (15 top spam-correlated and 15 top non-spam correlated words)

weight_dictionary = {}

for x in range(len(training_info[0])):
    if training_info[0][x] not in weight_dictionary.keys():
        weight_dictionary[ training_info[0][x] ] = [x]
    else:
        weight_dictionary[ training_info[0][x] ].append(x)

weight_list = list( weight_dictionary.keys() )
weight_list.sort()

print( "Weight Dictionary Keys:",weight_list,"\n" )
#print( "Validate:", weight_list[-1], weight_list[0])

spam_correlated_words = []
spam_inverse_correlated_words = []


counter = 0
while counter < 15:
    for y in range( weight_list[-1], 0, -1 ):
        if y in weight_dictionary.keys() and len( weight_dictionary[y] ) > 0:
            while len( weight_dictionary[y] ) > 0:
                spam_correlated_words.append( [weight_dictionary[y].pop(),y] )
                counter += 1
                if counter == 15:
                    break
            if counter == 15:
                break

counter = 0
while counter < 15:
    for y in range( weight_list[0], 0, 1 ):
        if y in weight_dictionary.keys() and len( weight_dictionary[y] ) > 0:
            while len( weight_dictionary[y] ) > 0:
                spam_inverse_correlated_words.append( [weight_dictionary[y].pop(),y] )
                counter += 1
                if counter == 15:
                    break
            if counter == 15:
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
