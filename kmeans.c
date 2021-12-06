#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define DEFAULT_MAX_ITER 200
#define DATATYPE double
#define EPSILON 0.001


DATATYPE **allocateMatrix(int m, int n);
void invalidInput();
void exceptionHandler();
int secureStrtol(char *str);
int countLines(FILE *fp);
int countCoulmns(FILE *fp);
void vectorSum(DATATYPE *src1, DATATYPE *src2, DATATYPE *dst, int size);
void scalarProduct(DATATYPE *src, DATATYPE *dst, DATATYPE scalar, int size);
DATATYPE euclideanDistance(DATATYPE *vector1, DATATYPE *vector2, int size);
int getClosestCluster(DATATYPE *vector, DATATYPE **centroids, int k, int n);


void invalidInput() {
    printf("Invalid Input!\n");
    exit(1);
}


void exceptionHandler() {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* checks if given string is an integer */
int secureStrtol(char *str) {
    char *endptr;
    int result = strtol(str, &endptr, 10);
    if (*endptr != '\0') {
        invalidInput();
    }
    return result;
}


int countLines(FILE *fp) {
    char c;
    int count = 0;
    for (c = fgetc(fp); c != EOF; c = fgetc(fp)) {
        if (c == '\n') {  /* Increment count if this character is newline */
            count ++;
        }
    }
    fseek(fp, 0, SEEK_SET);
    return count;
}


int countColumns(FILE *fp) {
    char c;
    int count = 1; /* # of commas in line + 1 */
    for (c = fgetc(fp); c != '\n'; c = fgetc(fp)) {
        if (c == ',') {
            count++;
        }
    }
    fseek(fp, 0, SEEK_SET);
    return count;
}


/* allocates an m by n matrix (m rows, n columns) as a continunous block in meomry */
DATATYPE **allocateMatrix(int m, int n) {
    int i;
    DATATYPE *ptr;
    DATATYPE **matrix;
    
    matrix = (DATATYPE **)malloc(sizeof(DATATYPE *) * m + sizeof(DATATYPE) * m * n);
    if (matrix == NULL) {
        exceptionHandler();
    }

    /* ptr is now pointing to the first element in the matrix */
    ptr = (DATATYPE *)(matrix + m);
 
    /* for loop to point rows pointer to appropriate location in 2D array */
    for (i = 0; i < m; i++) {
        matrix[i] = (ptr + n * i);
    }
    return matrix;
}

DATATYPE euclideanDistance(DATATYPE *vector1, DATATYPE *vector2, int size){
    int i; 
    DATATYPE sum = 0;
    for (i = 0; i < size; i++) {
        sum += pow(vector1[i] - vector2[i], 2);
    }
    return sqrt(sum);
} 


void vectorSum(DATATYPE *src1, DATATYPE *src2, DATATYPE *dst, int size) {
    int i;
    for (i = 0; i < size; i++) {
        dst[i] = src1[i] + src2[i];
    }
}


void scalarProduct(DATATYPE *src, DATATYPE *dst, DATATYPE scalar, int size) {
    int i;
    dst[0] = 100;
    for (i = 0; i < size; i++) {
        dst[i] = scalar * src[i];
        /*printf("src[%d] = %f, dst[%d] = %f, scalar = %f\n", i, src[i], i, dst[i], scalar);*/
    }
}


int getClosestCluster(DATATYPE *vector, DATATYPE **centroids, int k, int n) {
    int closestIndex = 0, index;
    DATATYPE minDistance, distance;

    minDistance = euclideanDistance(vector, centroids[0], n);
    for (index = 1; index < k; index++) {
        distance = euclideanDistance(vector, centroids[index], n);
        if (distance < minDistance) {
            minDistance = distance;
            closestIndex = index;
        }
    }
    return closestIndex;
}


DATATYPE **kmeans(DATATYPE **datapoints, int k, int maxIter, int m, int n){
    DATATYPE **centroids;
    DATATYPE **prevCentroids;
    DATATYPE **clusterSum;
    int *clustersMapping, *clusterSize;
    int iteration = 0;
    int i, cluster, converges;
    
    centroids = allocateMatrix(k, n);
    prevCentroids = allocateMatrix(k, n);
    clusterSum = allocateMatrix(k, n);
    clusterSize = (int *)malloc(sizeof(int) * k);
    clustersMapping = (int *)malloc(sizeof(int) * m);
    printf("finished alocation memory\n");

    memcpy(&centroids[0][0], &datapoints[0][0], sizeof(DATATYPE) * k * n);

    while (iteration <= maxIter) {
        printf("starting iteration #%d\n", iteration);
        iteration++;
        memcpy(&prevCentroids[0][0], &centroids[0][0], sizeof(DATATYPE) * k * n);
        memset(clusterSize, 0, sizeof(int) * k);
        memset(&clusterSum[0][0], 0, sizeof(DATATYPE) * k * n);

        /* find the closest centroid to each datapoint */
        for (i = 0; i < m; i++) {
            clustersMapping[i] = getClosestCluster(datapoints[i], centroids, k, n);
        }

        /* calculate the new centroids by averging the points in each cluster */
        for (i = 0; i < m; i++) {
            cluster = clustersMapping[i];
            vectorSum(clusterSum[cluster], datapoints[i], clusterSum[cluster], n);
            clusterSize[cluster]++;
        }
        
        for (cluster = 0; cluster < k; cluster++) {
            scalarProduct(clusterSum[cluster], centroids[cluster], 1 / (DATATYPE)clusterSize[cluster], n);
        }

        converges = 1;
        for (cluster = 0; cluster < k; cluster++) {
            if (euclideanDistance(centroids[cluster], prevCentroids[cluster], n) >= EPSILON) {
                converges = 0;
                break;
            }
        }
        if (converges) {
            free(prevCentroids);
            free(clusterSum);
            free(clusterSize);
            free(clustersMapping);
            printf("Converges on iteration: %d\n", iteration);
            return centroids;
        }
    }
    free(prevCentroids);
    free(clusterSum);
    free(clusterSize);
    free(clustersMapping);
    return centroids;
}


int main(int argc, char *argv[]) {
    int k, i, j, rows, columns;
    int maxIter;
    char *inputPath;
    char *outputPath;
    DATATYPE **datapoints, **centroids;
    FILE *fp;

    /* parse commandline arguments */
    if (argc == 5) {
        k = secureStrtol(argv[1]);
        maxIter = secureStrtol(argv[2]);;
        inputPath = argv[3];
        outputPath = argv[4];
    } else if (argc == 4) {
        k = secureStrtol(argv[1]);
        maxIter = DEFAULT_MAX_ITER;
        inputPath = argv[2];
        outputPath = argv[3];
    } else {
        invalidInput();
    }

    /* loading the CSV to a 2D matrix */
    fp = fopen(inputPath, "r");
    if (fp == NULL) {
        exceptionHandler();
    }
    
    rows = countLines(fp);
    columns = countColumns(fp);
    
    datapoints = allocateMatrix(rows, columns);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            fscanf(fp, "%lf", &datapoints[i][j]);
            fgetc(fp);  /* skip 1 char (comma or newline) */
        }
    }
    fclose(fp);

    printf("outputPath=%s\n", outputPath);

    printf("first 9 datapoints:\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n", datapoints[0][0], datapoints[0][1], datapoints[0][2], datapoints[1][0], datapoints[1][1], datapoints[1][2], datapoints[2][0], datapoints[2][1], datapoints[2][2]);
    centroids = kmeans(datapoints, k, maxIter,rows, columns);
    printf("first 9 centroids:\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n", centroids[0][0], centroids[0][1], centroids[0][2], centroids[1][0], centroids[1][1], centroids[1][2], centroids[2][0], centroids[2][1], centroids[2][2]);
    
    free(datapoints);
    free(centroids);
    return 0;
}