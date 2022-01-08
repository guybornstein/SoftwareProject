#define PY_SSIZE_T_CLEAN
#include <Python.h>


#define DATATYPE double


DATATYPE **allocateMatrix(int m, int n);
void exceptionHandler();
void vectorSum(DATATYPE *src1, DATATYPE *src2, DATATYPE *dst, int size);
void scalarProduct(DATATYPE *src, DATATYPE *dst, DATATYPE scalar, int size);
DATATYPE euclideanDistance(DATATYPE *vector1, DATATYPE *vector2, int size);
int getClosestCluster(DATATYPE *vector, DATATYPE **centroids, int k, int n);


void exceptionHandler() {
    printf("An Error Has Occurred\n");
    exit(1);
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


DATATYPE **kmeans(DATATYPE **datapoints, int k, int maxIter, int m, int n, int *observations, double epsilon) {
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
    if (clusterSize == NULL || clustersMapping == NULL) {
        exceptionHandler();
    }

    for (i = 0; i < k; i++) {
        memcpy(&centroids[i][0], &datapoints[observations[i]][0], sizeof(DATATYPE) * n);
    }

    while (iteration < maxIter) {
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
            if (euclideanDistance(centroids[cluster], prevCentroids[cluster], n) > epsilon) {
                converges = 0;
                break;
            }
        }
        if (converges) {
            break;
        }
    }
    free(prevCentroids);
    free(clusterSum);
    free(clusterSize);
    free(clustersMapping);
    return centroids;
}

/* writes the centroids into the file in output_path,
k is the number of centroids and n is the vector size. */
void saveCSV(DATATYPE** centroids, char *output_path, int k, int n) {
    int i, j;
    FILE *fp;

    fp = fopen(output_path, "w");
    if (fp == NULL) {
        exceptionHandler();
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j < n - 1; j++) {
            fprintf(fp, "%.4f,", centroids[i][j]);
        }
        fprintf(fp, "%.4f\n", centroids[i][n - 1]);
    }

    fclose(fp);
}


static PyObject* fit(PyObject* self, PyObject* args) {
    char *filename;
    PyArg_ParseTuple(args, "s", &filename);

    int k, i, j, rows, columns;
    int maxIter;
    double epsilon;
    DATATYPE **datapoints, **centroids;
    int* observations;
    FILE *fp;

    /* loading the CSV to a 2D matrix */
    fp = fopen(filename, "r");
    if (fp == NULL) {
        exceptionHandler();
    }
    
    fscanf(fp, "%i", &rows);
    fgetc(fp);
    fscanf(fp, "%i", &columns);
    fgetc(fp);
    fscanf(fp, "%i", &k);
    fgetc(fp);
    fscanf(fp, "%i", &maxIter);
    fgetc(fp);
    fscanf(fp, "%lf", &epsilon);
    fgetc(fp);

    observations = (int*)malloc(k * sizeof(int));
    
    for (i = 0; i < k; i++) {
        fscanf(fp, "%i", &observations[i]);
        fgetc(fp);  /* skip 1 char (comma or newline) */
    }

    datapoints = allocateMatrix(rows, columns);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            fscanf(fp, "%lf", &datapoints[i][j]);
            fgetc(fp);  /* skip 1 char (comma or newline) */
        }
    }
    fclose(fp);

    centroids = kmeans(datapoints, k, maxIter,rows, columns, observations, epsilon);
    saveCSV(centroids, filename, k, columns);

    free(datapoints);
    free(centroids);
    free(observations);
    
    Py_RETURN_NONE;
}


static PyMethodDef capiMethods[] = {
    {
        "fit",
        (PyCFunction) fit,
        METH_VARARGS,
        PyDoc_STR("A c implementation of kmeans algorithm.")},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    capiMethods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *module;
    module = PyModule_Create(&moduleDef);
    if (!module) {
        return NULL;
    }
    return module;
};