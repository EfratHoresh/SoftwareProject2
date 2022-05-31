#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


void kmeans(int k, int iterNum,double e);
void discover_n_d(char input[], int n_d[2], int line_len);
void fill_mat(char input[], double** vectors, int n, int line_len);
int is_number(char s[]);
void update_u(double* new_centroid, int* cluster_of_vector, double** vectors, int d, int n, int clust_num);
int argmin(double x[], double** centroids, int n, int d);
double occlid_distance(double x1[], double x2[], int d);
void write_in_file(char output[], double** centroids, int k, int d);
void free_double_matrix(double** matrix, int last_index);
int discover_len(char input[]);


static PyObject* fit(PyObject* Py_UNUSED(self), PyObject* args) {
    int k;
    int max_iter;
    double e;
    if (!PyArg_ParseTuple(args, "iid", &k, &max_iter, &e)) {
        Py_RETURN_NONE;
    }
    kmeans(k, max_iter, e);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
        { "fit", &fit, METH_VARARGS, "kmeans function" },
        { NULL, NULL, 0, NULL }
};

static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, // always required
        "mykmeanssp",         // module name
        "calculate kmeans",      // description
        -1,                    // module size (-1 indicates we don't use this feature)
        methods,               // method table
};

PyMODINIT_FUNC PyInit_mykmeanssp() {
    return PyModule_Create(&module_def);
}


void kmeans(int k, int max_iter, double e) {
    char input[] = "vectors_tmp_file.txt";
    char centroid_path[] = "centroids_tmp_file.txt";
    int n = 0;
    int d = 0;
    int n_d[2] = {0, 0};
    int line_len = 0;
    int i = 0;
    int j = 0;
    int num_of_loops = 0;
    double** vectors;
    double** centroids;
    int* cluster_of_vector;
    double* norms;
    double* new_centroid;
    int clust_num = 0;
    int is_bigger_than_e = 0;
    line_len = discover_len(input);
    discover_n_d(input, n_d, line_len);
    n = (int)n_d[0];
    d = n_d[1];
    vectors = malloc(n*sizeof(double*));  /* create empty matrix */
    if(vectors == NULL) {
        printf("An Error Has Occurred");
        free(vectors);
        exit(1);
    }
    for (i = 0; i < n; i++) {
        vectors[i] = malloc(d*sizeof(double));
        if(vectors[i] == NULL) { /* check if there's enough memory for vectors */
            printf("An Error Has Occurred");
            free_double_matrix(vectors, i);  /* frees all lines from 0 to i, including prime matrix */
            exit(1);
        }
    }
    fill_mat(input, vectors, n, line_len); /* fill empty matrix with vectors */
    line_len = discover_len(centroid_path);
    centroids = malloc(k*sizeof(double*));  /* create centroids empty matrix*/
    if(centroids == NULL) {
        printf("An Error Has Occurred");
        free_double_matrix(vectors, n-1);
        free_double_matrix(centroids, -1);
        exit(1);
    }
    for (i = 0; i < k; i++){
        centroids[i] = malloc(d*sizeof(double));
        if(centroids[i] == NULL) {
            printf("An Error Has Occurred");
            free_double_matrix(vectors, n-1);
            free_double_matrix(centroids, i);
            exit(1);
        }
    }
    fill_mat(centroid_path, centroids, k, line_len);  /* fill empty matrix with centroids */
    cluster_of_vector = malloc(n*sizeof(int)); /* cluster_of_vector[i]==cluster of xi vector */
    norms = malloc(k*sizeof(double));  /* oclid distance between new and old centroid */
    new_centroid = malloc(d*sizeof(double));
    if(cluster_of_vector == NULL || norms == NULL || new_centroid == NULL) {
        printf("An Error Has Occurred");
        free_double_matrix(vectors, n-1);
        free_double_matrix(centroids, k-1);
        free(cluster_of_vector);
        free(norms);
        free(new_centroid);
        exit(1);
    }

    while(1 == 1){
        for (i = 0; i < n; i++) {  /* find closest cluster for each vector */
            cluster_of_vector[i] = argmin(vectors[i], centroids, k, d);
        }
        clust_num = 0;
        for (clust_num = 0; clust_num < k; clust_num++) {  /* calculate new centroid and norms */
            for (j = 0; j < d; j++) {
                new_centroid[j] = 0.0;
            }
            update_u(new_centroid, cluster_of_vector, vectors, d, n, clust_num);
            norms[clust_num] = sqrt(occlid_distance(new_centroid, centroids[clust_num], d));  /* oclid distance between new and old centroid */
            for (j = 0; j < d; j++) {  /* update centroid */
                centroids[clust_num][j] = new_centroid[j];
            }
        }
        num_of_loops++;
        is_bigger_than_e = 0;
        for (i = 0; i < k; i++) {  /* check if any norm is bigger than e */
            if (norms[i] >= e) {
                is_bigger_than_e = 1;
                break;
            }
        }
        if (num_of_loops >= max_iter || is_bigger_than_e == 0) {
            break;
        }
    }
    free_double_matrix(vectors, n-1);
    free(cluster_of_vector);
    free(norms);
    free(new_centroid);
    write_in_file("results_tmp_file.txt", centroids, k, d);
    free_double_matrix(centroids, k-1);
    return;
}


void discover_n_d(char input[], int n_d[2], int line_len) {
    char* line;
    int n = 0;
    int d = 0;
    char * token;
    FILE* inputFile;

    line = malloc((line_len + 100)*sizeof(char));  /* extra 100 just in case */
    if(line == NULL) { /* check if there's enough memory for line */
        printf("An Error Has Occurred");
        free(line);
        exit(1);
    }
    inputFile = fopen(input, "r");
    if (inputFile == NULL) {
        printf("An Error Has Occurred");
        free(line);
        exit(1);
    }
    while (fgets(line, line_len + 100, inputFile)) {  /* searching for n and d */
        d = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            d++;
            token = strtok(NULL, ",");
        }
        n++;
    }
    fclose(inputFile);
    free(line);
    n_d[0] = n;
    n_d[1] = d;
    return;
}

void fill_mat(char input[], double** vectors, int n, int line_len) {
    FILE* inputFile;
    char * line;
    int i = 0;
    int j = 0;
    char *token;

    line = malloc((line_len + 100)*sizeof(char));
    if(line == NULL) { /* check if there's enough memory for line */
        printf("An Error Has Occurred");
        free_double_matrix(vectors, n-1);
        exit(1);
    }
    inputFile = fopen(input, "r");
    if (inputFile == NULL) {
        printf("An Error Has Occurred");
        free_double_matrix(vectors, n-1);
        free(line);
        exit(1);
    }
    while (fgets(line, line_len + 100, inputFile)) {
        token = strtok(line, ",");
        while (token != NULL) {
            sscanf(token, "%lf", &vectors[i][j]);
            j++;
            token = strtok(NULL, ",");
        }
        i++;
        j = 0;
    }
    fclose(inputFile);
    free(line);
    return;
}

int is_number(char s[]) {  /* check if string is int */
    int i = 0;
    for (i = 0; i < (int)strlen(s); i++)
        if (isdigit(s[i]) == 0)
            return 0;
    return 1;
}

double occlid_distance(double x1[], double x2[], int d) {
    double distance = 0.0;
    int i = 0;
    for (i = 0; i < d; i++) {
        distance += pow(x1[i]-x2[i], 2);
    }
    return distance;
}

int argmin(double x[], double** centroids, int k, int d) {  /* returns the index of the cluster */
    int index = 0;
    double min_distance = 0.0;
    double distance = 0.0;
    int j = 0;

    min_distance = occlid_distance(x, centroids[0], d);
    for (j = 1; j < k; j++) {  /* find closest centroid */
        distance = occlid_distance(x, centroids[j], d);
        if (distance < min_distance) {
            min_distance = distance;
            index = j;
        }
    }
    return index;
}

void update_u(double* new_centroid, int* cluster_of_vector, double** vectors, int d, int n, int clust_num) {
    int num_vectors_in_cluster = 0;
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {  /* finds all vectors in cluster */
        if (cluster_of_vector[i] == clust_num) {
            num_vectors_in_cluster++;
            for (j = 0; j < d; j++) {
                new_centroid[j] +=vectors[i][j];
            }
        }
    }
    for (i = 0; i < d; i++) {
        new_centroid[i] = new_centroid[i] / (double)num_vectors_in_cluster;
    }
    return;
}

void write_in_file(char output[], double** centroids, int k, int d) {
    int i = 0;
    int j = 0;
    FILE * output_file = fopen(output, "w");
    if (output_file == NULL) {
        printf("An Error Has Occurred");
        free_double_matrix(centroids, k-1);
        exit(1);
    }
    for (i = 0; i < k; i++) {
        for (j = 0; j < d-1; j++) {
            fprintf(output_file, "%.4f,", centroids[i][j]);
        }
        fprintf(output_file, "%.4f\n", centroids[i][d-1]);
    }
    fclose(output_file);
}


void free_double_matrix(double** matrix, int last_index) {  /* frees given matrix which is filled from index 0 to lst_index */
    int j = 0;

    if (matrix != NULL) {
        for (j = 0; j <= last_index; j++) {
            free(matrix[j]);
        }
    }
    free(matrix);
    return;
}


int discover_len(char input[]) {
    char c = 'c';
    int max_length = 0;
    int line_length = 0;
    FILE * input_file;

    input_file = fopen(input, "r");
    if (input_file == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    while (c != EOF) {
        line_length = 0;
        while (1 == 1) {
            c = getc(input_file);
            line_length++;
            if (c == '\n' || c == EOF) {
                break;
            }
        }
        if (line_length > max_length) {
            max_length = line_length;
        }
    }
    fclose(input_file);
    return max_length;
}
