from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import math
import random

# After how many items should we stop?
THRESHOLD = -1 # To be set via command line


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=THRESHOLD:
        return
    remaining_items = THRESHOLD-streamLength[0]
    # Extract the distinct items from the batch
    batch_list = batch.map(lambda x: int(x)).collect()

    batch_list = batch_list[:remaining_items]

    # Update the streaming state
    exact_algorithm(batch_list)
    reservoir_sample(batch_list, streamLength[0])
    sticky_sampling(batch_list)

    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    streamLength[0] += batch_size
    # stopping condition is met
    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()

def exact_algorithm(batch_list):
    global histogram
    for item in batch_list:
        if item not in histogram:
            histogram[item] = 1
        else:
            histogram[item] += 1

def reservoir_sample(batch_list, t):
    global S, phi
    m = math.ceil(1/phi)
    for i, item in enumerate(batch_list):
        if len(S) <= m:
            S.append(item)
        elif random.random() <= (m / (t+i+1)):
            index = random.randint(0,m-1)
            S[index] = item

def sticky_sampling(batch_list):
    global SSmap, phi, epsilon, delta
    r = math.log(1/(delta*phi))/epsilon
    for item in batch_list:
        if item in SSmap:
            SSmap[item]+=1
        elif random.random() < r/THRESHOLD:
            SSmap[item] = 1



if __name__ == '__main__':
    assert len(sys.argv) == 6, "USAGE: n, phi, epsilon, delta, portExp"

    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("G017HW3")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    # conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")


    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()


    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



    THRESHOLD = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])

    print("INPUT PROPERTIES")
    print(f"n = {THRESHOLD} phi = {phi} epsilon = {epsilon} delta = {delta} port = {portExp}")

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    streamLength = [0] # Stream length (an array to be passed by reference)
    histogram = {} # Hash Table for the exact_algorithm
    S = []
    SSmap = {}

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    # MANAGING STREAMING SPARK CONTEXT
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, True)


    # TODO: Sort the items
    # EXACT ALGORITHM
    print("EXACT ALGORITHM")
    print(f"Number of items in the data structure = {len(histogram)}")
    exact_frequent_items = []
    for key, value in histogram.items():
        if value >= phi*THRESHOLD:
            exact_frequent_items.append(key)
    print(f"Number of true frequent items = {len(exact_frequent_items)}")
    print("True frequent items:")
    for fi in exact_frequent_items:
        print(fi)

    # RESERVOIR SAMPLING
    estimated_freq_items = list(set(S))
    print("RESERVOIR SAMPLING")
    print(f"Size m of the sample = {math.ceil(1 / phi)}")
    print(f"Number of estimated frequent items = {len(estimated_freq_items)}")
    print("Estimated frequent items:")
    for item in estimated_freq_items:
        output = str(item) + ' +' if item in exact_frequent_items else str(item) + ' -'
        print(output)

    # STICKY SAMPLING

    sticky_sample = [k for k,v in SSmap.items() if v >= (phi-epsilon)*THRESHOLD]
    print("STICKY SAMPLING")
    print(f"Number of items in the Hash Table = {len(SSmap)}")
    print(f"Number of estimated frequent items = {len(sticky_sample)}")
    print("Estimated frequent items:")
    for item in sticky_sample:
        output = str(item) + ' +' if item in exact_frequent_items else str(item) + ' -'
        print(output)