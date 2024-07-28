# Big Data Computing Homeworks
This repo contains the Homeworks for the Big Data Computing course at University of Padova.

## Homework 1
In the first homework we implemented the approximate outlier detection algorithm presented in Edwin M. Knorr, Raymond T. Ng, V. Tucakov: _Distance-Based Outliers: Algorithms and Applications_, VLDB J. 
8(3-4): 237-253 (2000). The algorithm is implemented in **Apache Spark** and tested against different datasets of increasing size.

## Homework 2
In this homework, we implemented a modified version of the approximation strategy for outlier detection developed in Homework 1, where the distance parameter _D_ is not provided in input by the user, 
but is set equal to the radius of a k-center clustering (for a suitable number _K_ of clusters), that is, the maximum distance of a point from its closest center. We implemented the k-center clustering algorithm in
Spark, exploiting the coreset technique + SequentialFFT algorithm in order to efficiently run the code on very large datasets. The program was tested on [Cloud Veneto](https://cloudveneto.it/).

## Homework 3
In this homework, we used the Spark Streaming API to devise a program which processes a stream of items and compares the effectiveness of two methods to identify frequent items:
- the method based on **reservoir sampling**.
- the method based on **sticky sampling**.
