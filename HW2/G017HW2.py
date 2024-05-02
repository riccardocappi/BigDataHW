import time

from pyspark import SparkContext, SparkConf
from math import hypot, sqrt
# import random
import sys



def string_to_point(str_point):
    point_x, point_y = str_point.split(",")
    point = (float(point_x), float(point_y))
    return point


def gather_pairs_partitions(points):
    points_dict = {}
    for point in points:
        if point not in points_dict:
            points_dict[point] = 1
        else:
            points_dict[point] += 1
    return [(key, points_dict[key]) for key in points_dict.keys()]


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
    grid_counts = (points.map(lambda point: ((int(point[0] // Lambda), int(point[1] // Lambda)), 1))
                   .reduceByKey(lambda x, y: x + y)).cache()

    # Step B can be sequential
    grid_counts_list = grid_counts.collect()
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
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def SequentialFFT(P, K):
    if len(P) == 0:
        return []

    P_set = set(P)
    c_1 = P[0]
    C = [c_1]
    P_set.remove(c_1)
    distance_table = dict((p, distance(p, c_1)) for p in P_set)
    for i in range(2, K+1):
        c_i = max(distance_table.keys(), key=lambda x: distance_table[x])
        for p in P_set:
            curr_dis = distance(p, c_i)
            if curr_dis < distance_table[p]:
                distance_table[p] = curr_dis
        C.append(c_i)
        P_set.remove(c_i)
        # del distance_table[c_i]
    return C

def MRFFT(P, K):
    coreset = P.mapPartitions(lambda P_i: SequentialFFT(list(P_i), K)).collect()
    centers = SequentialFFT(coreset, K)
    # Round 3
    C = sc.broadcast(centers)
    R = (P.map(lambda point: min([distance(point, c) for c in C.value]))
     .reduce(lambda d1, d2: d1 if d1 > d2 else d2))
    return R


# SPARK SETUP
conf = SparkConf().setAppName('G017HW2')
sc = SparkContext(conf=conf)


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
    start = time.time()
    D = MRFFT(inputPoints, K)
    end = time.time()
    print(D)
    print(f"Running time = {round((end - start) * 1000)} ms")
    MRApproxOutliers(inputPoints, D, M)


if __name__ == "__main__":
    main()
