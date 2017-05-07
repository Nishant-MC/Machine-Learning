import subprocess
import threading
import pylab

# Fill in unit tests here
UNIT_TESTS = [ ("python", "ps1.py", "N=100", "X=20", "PL=250", "TW=12"),
               ("python", "ps1.py", "N=200", "X=20", "PL=250", "TW=12"),
               ("python", "ps1.py", "N=400", "X=20", "PL=250", "TW=12"),
               ("python", "ps1.py", "N=800", "X=20", "PL=250", "TW=12"),
               ("python", "ps1.py", "N=2000", "X=20", "PL=250", "TW=12"),
               ("python", "ps1.py", "N=4000", "X=20", "PL=250", "TW=12")
             ] 

# The results will be filled up / calculated by the individual subprocesses
THREADS = []
RESULTS = {}
SEMAPHORE = threading.BoundedSemaphore(len(UNIT_TESTS))
UNIT_COUNT = 0

# Unit test target thread
def PS1_UNIT_TEST(env,target,n,x,pl,tw):
    global RESULTS, SEMAPHORE, UNIT_COUNT
    UT = tuple([env,target,n,x,pl,tw])
    output_string = subprocess.getoutput( " ".join(UT) ).split("\n")
    RESULTS[ UT ] = [ float(output_string[-4].split(": ")[1]), # Training error
                      float(output_string[-3].split(": ")[1]), # Validation error
                      float(output_string[-2].split(": ")[1]), # Test error
                      float(output_string[-1].split(": ")[1])  # Perceptron pass count
                    ]
    print("Configuration", " ".join(UT), "completed!")
    SEMAPHORE.release()
    UNIT_COUNT -= 1

# Used as a key for sorting plot points
def N_count(tup):
    return float(tup[2].split("=")[1])

# Main thread
def mainloop():
    
    # Wait / spin until all the unit test threads are complete and global RESULTS is populated
    global RESULTS, UNIT_COUNT
    while UNIT_COUNT != 0:
        pass
        
    # Displaying validation error as a function of N
    pylab.plot( [ float(d[2].split("=")[1]) for d in sorted(RESULTS,key=N_count) ],
                   [ RESULTS[d][1]*100 for d in sorted(RESULTS,key=N_count) ], color='g', label="V-error % vs. N" )

    # Displaying test error as a function of N
    pylab.plot( [ float(d[2].split("=")[1]) for d in sorted(RESULTS,key=N_count) ],
                   [ RESULTS[d][2]*100 for d in sorted(RESULTS,key=N_count) ], color='b', label="T-error % vs. N" )

    # Displaying pass count as a function of N
    pylab.plot( [ float(d[2].split("=")[1]) for d in sorted(RESULTS,key=N_count) ],
                   [ RESULTS[d][3] for d in sorted(RESULTS,key=N_count) ], color='r', label="Pass count vs. N" )

    # Setting axis limits, activating the legend, and displaying the window
    pylab.xlim( (0,4000) )
    pylab.ylim( (0,25) )
    pylab.legend()
    pylab.show()
    
               
if __name__ == "__main__":

    # Create a partitioned thread for every unit test so that they can be run simultaneously
    for i in range(len(UNIT_TESTS)):
        THREADS.append( threading.Thread( target=PS1_UNIT_TEST, args=UNIT_TESTS[i] ) )
        SEMAPHORE.acquire()
        UNIT_COUNT += 1
        THREADS[-1].start()

    # mainloop() contains all the graph plotting code
    mainloop()
