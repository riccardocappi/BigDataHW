from pyspark import SparkContext, SparkConf
from math import hypot, sqrt
import time
import sys


'''
1) Write a method/function ExactOutliers which implements the Exact algorithm,
through standard sequential code which does not use RDDs.  Specifically, ExactOutliers
takes as input a list of points (ArrayList in Java, or a list in Python) and
parameters 𝐷 (float), 𝑀,𝐾 (integers), and must compute and print the following information.
    - The number of (𝐷,𝑀)
    - outliers.
The first 𝐾 outliers points 𝑝 in non-decresing order of |𝐵𝑆(𝑝,𝐷)|, one point per line. 
(If there are less than 𝐾 outlier, it prints all of them.)

'''


def ExactOutliers(points, D, M, K):
    inside_points = [0]*len(points)
    for i in range(len(points)-1):
        for j in range(i+1,len(points)):
            dist = hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
            if dist < D:
                inside_points[i] += 1
                inside_points[j] += 1
    outliers = [(point, count) for point, count in zip(points, inside_points) if count < M]
    outliers.sort(key=lambda x: x[1])
    print("Number of outliers:", len(outliers))
    for outlier in outliers[:K]:
        print("Point: " + str(outlier[0]))


'''
2) Write a method/function MRApproxOutliers which implements the above approximate algorithm. Specifically, 
MRApproxOutliers must take as input an RDD of points and parameters 𝐷 (float), 𝑀,𝐾 (integers), and can assume 
that the RDD is already subdivided into a suitable number of partitions. MRApproxOutliers consists of two main steps. 
Step A transforms the input RDD into an RDD whose elements corresponds to the non-empty cells and, contain, for each cell, 
its identifier (𝑖,𝑗) and the number of points of 𝑆 that it contains. The computation must be done by exploiting the 
Spark partitions, without gathering together all points of a cell (which could be too many). Step B transforms the RDD 
of cells, resulting from Step A, by attaching to each element, relative to a non-empty cell 𝐶, the values |𝑁3(𝐶)| and |𝑁7(𝐶)|, 
as additional info. To this purpose, you can assume that the total number of non-empty cells is small with respect to 
the capacity of each executor's memory. MRApproxOutliers must eventually compute and print
    - The number of sure (𝐷,𝑀)

    - outliers.
    
    - The number of uncertain points.
For the first 𝐾 non-empty cells,  in non-decreasing order of |𝑁3(𝐶)|, their identifiers and value of |𝑁3(𝐶)|, one 
line per cell. (If there are less than 𝐾 non-empty cells, it prints the information for all of them.)
'''


def string_to_point(str_point):
    point_x, point_y = str_point.split(",")
    point = (float(point_x), float(point_y))
    return point


def gather_partitions(points):
    points_dict = {}
    for point in points:
        if point not in points_dict:
            points_dict[point] = 1
        else:
            points_dict[point] += 1
    return [(key, points_dict[key]) for key in points_dict.keys()]


def compute_n3_n7(point_count_map, x, y):
    N3, N7 = 0, 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = (x + i, y + j)
            if p in point_count_map:
                is_n3 = 0 if (abs(i) >= 2) or (abs(j) >= 2) else 1
                count = point_count_map[p]
                N3 += count * is_n3
                N7 += count
    size = point_count_map[(x, y)]
    return (x, y), size, N3, N7


def MRApproxOutliers(points, D, M, K):
    # Step A
    Lambda = D / (2 * sqrt(2))
    grid_counts = (((points.map(lambda point: (point[0] // Lambda, point[1] // Lambda)) # Round 1. Map points to grid
                     .mapPartitions(gather_partitions)
                     .groupByKey() # Round 2
                     .mapValues(lambda vals: sum(vals))))).cache()

    # Step B can be sequential
    grid_counts_list = grid_counts.collect()
    grid_counts_dict = dict((k, v) for k,v in grid_counts_list)
    # Determine for each cell its identifier, count, N3 and N7
    cell_info = map(lambda cell: compute_n3_n7(grid_counts_dict, cell[0], cell[1])
                 ,grid_counts_dict.keys())
    outliers, uncertain_points = 0,0
    for cell, size, N3, N7 in cell_info:
        if N7 <= M:
            outliers += size
        elif N3 <= M:
            uncertain_points += size
    print(f"Number of sure outliers = {outliers}")
    print(f"Number of uncertain points = {uncertain_points}")
    # Since in StepA you generated an RDD of cells with their sizes, the first K
    # non-empty cells in the required order, must be obtained using a map method
    # (the most suitable one), followed by the sortByKey and take methods on this RDD.
    first_k_non_empty_cells = (grid_counts.map(lambda cell_counts: (cell_counts[1], cell_counts[0]))
                               .sortByKey(ascending=True)).take(K)
    for cell in first_k_non_empty_cells:
        print(f"Cell: {cell[1]}  Size = {cell[0]}")


def main():
    assert len(sys.argv) == 6, "Usage: python G017HW1.py <file_path> <D> <M> <K> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G017HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    try:
        file_path = sys.argv[1]
        D = float(sys.argv[2])
        M = int(sys.argv[3])
        K = int(sys.argv[4])
        L = int(sys.argv[5])
    except TypeError:
        print("Wrong type inserted")
        return -1

    print(f"{file_path} D={D} M={M} K={K} L={L}")

    # Create RDD of strings
    rawData = sc.textFile(file_path, minPartitions=L)
    # Transform the string RDD into an RDD of points (pair of floats)
    inputPoints = rawData.map(string_to_point).repartition(L).cache()
    number_of_points = inputPoints.count()
    print(f"Number of points = {number_of_points}")
    if number_of_points < 200000:
        # Downloads the points into a list called listOfPoints
        listOfPoints = inputPoints.collect()
        start = time.time()
        ExactOutliers(listOfPoints, D, M, K)
        end = time.time()
        print(f"Running time of ExactOutliers = {round((end - start) * 1000)} ms")

    start = time.time()
    MRApproxOutliers(inputPoints, D, M, K)
    end = time.time()
    print(f"Running time of MRApproxOutliers = {round((end - start) * 1000)} ms")


if __name__ == "__main__":
    main()

