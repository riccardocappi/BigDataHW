
from pyspark import SparkContext, SparkConf
'''
1) Write a method/function ExactOutliers which implements the Exact algorithm,
through standard sequential code which does not use RDDs.  Specifically, ExactOutliers
takes as input a list of points (ArrayList in Java, or a list in Python) and
parameters 𝐷 (float), 𝑀,𝐾 (integers), and must compute and print the following information.
    The number of (𝐷,𝑀)
-outliers.
The first 𝐾
outliers points 𝑝 in non-decresing order of |𝐵𝑆(𝑝,𝐷)|, one point per line. (If there are less than 𝐾 outlier, it prints all of them.)

'''
import math
from math import hypot
import time

def readinput(filename):
    file = open(filename)
    lines = file.readlines()
    points = []
    for line in lines:
        point_x, point_y = line.split(",")
        point = (float(point_x), float(point_y))
        points.append(point)

    return points


def exactOutliers(points, distance, M):
    inside_points = [0]*len(points)
    for i in range(len(points)-1):
        for j in range(i+1,len(points)):
            dist = hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
            if dist < distance:
                inside_points[i] += 1
                inside_points[j] += 1
    outliers = [(point, count) for point, count in zip(points, inside_points) if count < M]
    return outliers


'''
2) Write a method/function MRApproxOutliers which implements the above approximate algorithm. Specifically, 
MRApproxOutliers must take as input an RDD of points and parameters 𝐷 (float), 𝑀,𝐾 (integers), and can assume 
that the RDD is already subdivided into a suitable number of partitions. MRApproxOutliers consists of two main steps. 
Step A transforms the input RDD into an RDD whose elements corresponds to the non-empty cells and, contain, for each cell, its identifier (𝑖,𝑗) and the number of points of 𝑆 that it contains. The computation must be done by exploiting the Spark partitions, without gathering together all points of a cell (which could be too many). Step B transforms the RDD of cells, resulting from Step A, by attaching to each element, relative to a non-empty cell 𝐶, the values |𝑁3(𝐶)| and |𝑁7(𝐶)|

, as additional info. To this purpose, you can assume that the total number of non-empty cells is small with respect to the capacity of each executor's memory. MRApproxOutliers must eventually compute and print

    The number of sure (𝐷,𝑀)

-outliers.
The number of uncertain points.
For the first 𝐾
non-empty cells,  in non-decreasing order of |𝑁3(𝐶)|, their identifiers and value of |𝑁3(𝐶)|, one line per cell. (If there are less than 𝐾 non-empty cells, it prints the information for all of them.)
'''


def map_point(str_x, l):
    point_x, point_y = str_x.split(",")
    point = (float(point_x)//l, float(point_y)//l)
    return point


def gather_partitions(points):
    points_dict = {}
    for point in points:
        if point not in points_dict:
            points_dict[point] = 1
        else:
            points_dict[point] += 1
    return [(key, points_dict[key]) for key in points_dict.keys()]


def compute_outliers(point_count_map, M):
    outliers = []
    uncertain_outliers = []
    for (x, y) in point_count_map.keys():
        N3, N7 = 0, 0
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = (x + i, y + j)
                if p in point_count_map:
                    is_n3 = 0 if (abs(i) >= 2) or (abs(j) >= 2) else 1
                    count = point_count_map[p]
                    N3 += count * is_n3
                    N7 += count
        if N7 <= M:
            outliers.append(((x, y), point_count_map[(x, y)]))
        elif N3 <= M:
            uncertain_outliers.append(((x, y), point_count_map[(x, y)]))

    return outliers, uncertain_outliers


def MRApproxOutliers(points, D, M, K):
    # Step A
    mapped_points = (((points.map(lambda x: map_point(x, D/(2 * math.sqrt(2)))) # Round 1
                     .mapPartitions(gather_partitions)
                     .groupByKey() # Round 2
                     .mapValues(lambda vals: sum(vals))))).cache()

    # Step B can be sequential
    grid_points_list = mapped_points.collect()
    point_count_map = dict((k, v) for k,v in grid_points_list)
    return compute_outliers(point_count_map, M)




if __name__ == "__main__":
    D, M, K, L = 0.02, 10, 5, 2       # TODO: read from command line
    file_name = "uber-large.csv"
    points = readinput(file_name)
    number_of_points = len(points)
    print("Number of points:", number_of_points)
    if number_of_points < 200000:
        start = time.time()
        outliers = exactOutliers(points, D, M)
        end = time.time()
        outliers.sort(key=lambda x: x[1])
        print("Number of outliers:", len(outliers))
        for outlier in outliers[:K]:
            print("Point: " + str(outlier[0]))
        print(f"Running time of ExactOutliers = {round((end-start)*1000)} ms")

    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    points = sc.textFile(file_name, minPartitions=L).cache()
    points = points.repartition(numPartitions=L)

    start = time.time()
    outliers, uncertain_outliers = MRApproxOutliers(points, D, M, K)
    end = time.time()
    print(f"Number of sure outliers = {sum([x[1] for x in outliers])}")
    print(f"Number of uncertain outliers = {sum([x[1] for x in uncertain_outliers])}")
    outliers.sort(key = lambda x: x[1])
    for outlier in outliers[:K]:
        print(f"Cell: {outlier[0]}  Size = {outlier[1]}")
    print(f"Running time of MRApproxOutliers = {round((end - start) * 1000)} ms")

    # TODO: rename variables

