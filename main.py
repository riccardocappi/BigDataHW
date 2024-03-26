
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

def readinput(filename):
    file = open(filename)
    lines = file.readlines()
    points = []
    for line in lines:
        point_x, point_y = line.split(",")
        point = (float(point_x), float(point_y))
        points.append(point)

    return points


def exactOutliers(points, distance, M, K=10):
    inside_points = [0]*len(points)
    for i in range(len(points)):
        for j in range(i,len(points)):
            dist = hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
            if dist < distance:
                inside_points[i] += 1
                inside_points[j] += 1
    outliers = [point for point, count in zip(points, inside_points) if count < M]
    print(outliers)


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


def compute_neighbors(point_count_map):
    neighbors = []
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
        neighbors.append(((x, y), (N3, N7)))
    return neighbors


def MRApproxOutliers(points, D, M, K):
    # Step A
    mapped_points = (((points.map(lambda x: map_point(x, D/(2 * math.sqrt(2)))) # Round 1
                     .mapPartitions(gather_partitions)
                     .groupByKey() # Round 2
                     .mapValues(lambda vals: sum(vals))))).cache()

    # Step B can be sequential
    grid_points_list = mapped_points.collect()
    point_count_map = dict((k, v) for k,v in grid_points_list)
    print(compute_neighbors(point_count_map))


if __name__ == "__main__":
    points = readinput("uber-10k.csv")
    exactOutliers(points, 1, 3, 10)
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)

    K = 5
    points = sc.textFile("uber-10k.csv", minPartitions=K).cache()
    points = points.repartition(numPartitions=K)
    MRApproxOutliers(points, 0.02,10,5)

