from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import math
import random


def process_batch(batch):
    global streamLength
    batch_size = batch.count()
    # If we already have enough points (> n), skip this batch.
    if streamLength[0]>=n:
        return
    remaining_items = n - streamLength[0]

    batch_list = batch.map(lambda x: int(x)).collect()
    batch_list = batch_list[:remaining_items]

    exact_algorithm(batch_list)
    reservoir_sampling(batch_list, streamLength[0])
    sticky_sampling(batch_list)

    streamLength[0] += batch_size
    # stopping condition is met
    if streamLength[0] >= n:
        stopping_condition.set()


def exact_algorithm(batch_list):
    global frequency_map
    for item in batch_list:
        if item not in frequency_map:
            frequency_map[item] = 1
        else:
            frequency_map[item] += 1


def reservoir_sampling(batch_list, length):
    global reservoir_sample, phi
    m = math.ceil(1/phi)
    for i, item in enumerate(batch_list):
        if len(reservoir_sample) < m:
            reservoir_sample.append(item)
        elif random.random() <= (m / (length + i)):
            index = random.randint(0,m-1)
            reservoir_sample[index] = item


def sticky_sampling(batch_list):
    global sticky_sampling_map, phi, epsilon, delta, n
    r = math.log(1/(delta*phi))/epsilon
    for item in batch_list:
        if item in sticky_sampling_map:
            sticky_sampling_map[item]+=1
        elif random.random() <= r/n:
            sticky_sampling_map[item] = 1


def print_freq_items(items, freq_items):
    for item in items:
        output = str(item) + ' +' if item in freq_items else str(item) + ' -'
        print(output)


if __name__ == '__main__':
    assert len(sys.argv) == 6, "USAGE: python G017HW3.py n, phi, epsilon, delta, portExp"
    conf = SparkConf().setMaster("local[*]").setAppName("G017HW3")

    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])

    print("INPUT PROPERTIES")
    print(f"n = {n} phi = {phi} epsilon = {epsilon} delta = {delta} port = {portExp}")

    streamLength = [0] # Stream length (an array to be passed by reference)
    frequency_map = {} # Hash Table for the exact_algorithm
    reservoir_sample = []
    sticky_sampling_map = {}

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(batch))

    # MANAGING STREAMING SPARK CONTEXT
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)

    # EXACT ALGORITHM
    print("EXACT ALGORITHM")
    print(f"Number of items in the data structure = {len(frequency_map)}")
    exact_frequent_items = []
    for key, value in frequency_map.items():
        if value >= phi*n:
            exact_frequent_items.append(key)
    exact_frequent_items.sort()
    print(f"Number of true frequent items = {len(exact_frequent_items)}")
    print("True frequent items:")
    for fi in exact_frequent_items:
        print(fi)

    # RESERVOIR SAMPLING
    estimated_freq_items = list(set(reservoir_sample))
    estimated_freq_items.sort()
    print("RESERVOIR SAMPLING")
    print(f"Size m of the sample = {len(reservoir_sample)}")
    print(f"Number of estimated frequent items = {len(estimated_freq_items)}")
    print("Estimated frequent items:")
    print_freq_items(estimated_freq_items, exact_frequent_items)

    # STICKY SAMPLING
    sticky_sample = [k for k,v in sticky_sampling_map.items() if v >= (phi - epsilon) * n]
    sticky_sample.sort()
    print("STICKY SAMPLING")
    print(f"Number of items in the Hash Table = {len(sticky_sampling_map)}")
    print(f"Number of estimated frequent items = {len(sticky_sample)}")
    print("Estimated frequent items:")
    print_freq_items(sticky_sample, exact_frequent_items)