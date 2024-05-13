import time

from pyspark import SparkContext, SparkConf
from math import sqrt
import sys


def string_to_point(str_point):
    point_x, point_y = str_point.split(",")
    point = (float(point_x), float(point_y))
    return point


def compute_n3_n7(grid_counts_dict, x, y):
    N3, N7 = 0, 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = (x + i, y + j)
            if p in grid_counts_dict:
                is_n3 = 0 if (abs(i) >= 2) or (abs(j) >= 2) else 1
                count = grid_counts_dict[p]
                N3 += count * is_n3
                N7 += count
    size = grid_counts_dict[(x, y)]
    return (x, y), size, N3, N7


def MRApproxOutliers(points, D, M):
    # Step A
    Lambda = D / (2 * sqrt(2))
    grid_counts_list = (points.map(lambda point: ((int(point[0] // Lambda), int(point[1] // Lambda)), 1))
                        .reduceByKey(lambda x, y: x + y)).collect()

    # Step B can be sequential
    grid_counts_dict = dict((k, v) for k,v in grid_counts_list)
    # Determine for each cell its identifier, count, N3 and N7
    cell_info = map(lambda cell: compute_n3_n7(grid_counts_dict, cell[0], cell[1])
                 ,grid_counts_dict.keys())
    outliers, uncertain_points = 0,0
    for _, count, N3, N7 in cell_info:
        if N7 <= M:
            outliers += count
        elif N3 <= M:
            uncertain_points += count
    print(f"Number of sure outliers = {outliers}")
    print(f"Number of uncertain points = {uncertain_points}")


def distance(p1, p2):
    t0 = p1[0] - p2[0]
    t1 = p1[1] - p2[1]
    return t0*t0 + t1*t1


def SequentialFFT(P, K):
    if len(P) == 0:
        return []

    c_i = P[0]
    # c_1 = random.choice(P)
    C = [c_i]
    distance_table = dict((p, -1) for p in P)
    del distance_table[c_i]
    for i in range(2, K + 1):
        curr_max_dis = -1
        curr_c_i = None
        for p in distance_table.keys():
            curr_dis = distance(p, c_i)
            last_distance = distance_table[p]
            if curr_dis < last_distance or last_distance == -1:
                distance_table[p] = curr_dis
                last_distance = curr_dis

            if last_distance > curr_max_dis:
                curr_max_dis = last_distance
                curr_c_i = p
        c_i = curr_c_i
        C.append(c_i)
        del distance_table[c_i]
    return C


def MRFFT(P, K):
    # Round 1
    start = time.time()
    coreset_rdd = P.mapPartitions(lambda P_i: SequentialFFT(list(P_i), K)).cache()
    coreset_rdd.count()
    end = time.time()
    print(f"Running time of MRFFT Round 1 = {round((end - start) * 1000)} ms")

    # Round 2
    start = time.time()
    coreset = coreset_rdd.collect()
    # print(len(coreset))
    centers = SequentialFFT(coreset, K)
    end = time.time()
    print(f"Running time of MRFFT Round 2 = {round((end - start) * 1000)} ms")
    # Round 3
    start = time.time()
    C = sc.broadcast(centers)
    R = (P.map(lambda point: min([distance(point, c) for c in C.value]))
         .reduce(lambda d1, d2: d1 if d1 > d2 else d2))
    end = time.time()
    print(f"Running time of MRFFT Round 3 = {round((end - start) * 1000)} ms")
    return sqrt(R)


# SPARK SETUP
conf = SparkConf().setAppName('G017HW2')
conf.set("spark.locality.wait", "0s")
sc = SparkContext(conf=conf)

# random.seed(42)


def main():
    assert len(sys.argv) == 5, "Usage: python G017HW2.py <file_path> <M> <K> <L>"
    sc.setLogLevel("WARN")
    try:
        file_path = sys.argv[1]
        M = int(sys.argv[2])
        K = int(sys.argv[3])
        L = int(sys.argv[4])
    except TypeError:
        print("Wrong type inserted")
        return -1

    print(f"{file_path} M={M} K={K} L={L}")

    rawData = sc.textFile(file_path)
    inputPoints = rawData.map(string_to_point).repartition(L).cache()
    number_of_points = inputPoints.count()
    print(f"Number of points = {number_of_points}")

    D = MRFFT(inputPoints, K)
    print(f"Radius = {D}")

    start = time.time()
    MRApproxOutliers(inputPoints, D, M)
    end = time.time()
    print(f"Running time of MRApproxOutliers = {round((end - start) * 1000)} ms")


if __name__ == "__main__":
    main()