#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "symnmf.h"


const double EPSILON = 1e-4;
const int MAX_ITER = 300;
const double DENOMINATOR_EPSILON = 1e-6;
const double BETA = 0.5;


double** malloc_matrix(int n, int m) {
    double** A;
    int i;

    A = (double**)malloc(n * sizeof(double*));
    if (A == NULL) {
        printf("An Error Has Occurred1\n");
        exit(1);
    }

    for (i = 0; i < n; i++) {
        A[i] = (double*)malloc(m * sizeof(double));
        if (A[i] == NULL) {
            printf("An Error Has Occurred2\n");
            exit(1);
        }
    }

    return A;
}


void free_matrix(double** A, int n) {
    int i;

    for (i = 0; i < n; i++) {
        free(A[i]);
    }

    free(A);
}


double** transpose(double** A, int n, int m) {
    double** B;
    int i, j;
    
    B = malloc_matrix(m, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            B[j][i] = A[i][j];
        }
    }

    return B;
}


double** matrix_multiplication(double** A, double** B, int n, int r, int m) {
    double** C;
    int i, j, k;

    C = malloc_matrix(n, m);

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            C[i][j] = 0;

            for (k = 0; k < r; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}


double euclidean_distance(double* a, double* b, int d) {
    double result;
    int i;

    result = 0;

    for (i = 0; i < d; i++) {
        result += pow((a[i] - b[i]), 2);
    }

    return result;
}


double frobenius_norm(double** A, int n, int m) {
    double result;
    int i, j;

    result = 0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            result += pow(A[i][j], 2);
        }
    }

    return result;
}


double** sym_c(double** X, int n, int d) {
    double** A;
    int i, j;

    A = malloc_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0;
            }
            else {
                A[i][j] = exp(-(euclidean_distance(X[i], X[j], d))/2);
            }
        }
    }

    return A;
}


double** ddg_c(double** X, int n, int d) {
    double** D;
    double** A;
    int i, j;
    double sum;

    A = sym_c(X, n, d);
    D = malloc_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][j] = 0;
        }
    }

    for (i = 0; i < n; i++) {
        sum = 0;
        
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }

        D[i][i] = sum;
    }

    free_matrix(A, n);

    return D;
}


double** norm_c(double** X, int n, int d) {
    double** W;
    double** D;
    double** A;
    int i, j;
    double denominator;

    A = sym_c(X, n, d);
    D = ddg_c(X, n, d);
    W = malloc_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            denominator = sqrt(D[i][i] * D[j][j]);
            if (denominator == 0) {
                denominator = DENOMINATOR_EPSILON;
            }

            W[i][j] = A[i][j] / denominator;
        }
    }

    free_matrix(A, n);
    free_matrix(D, n);

    return W;
}


void symnmf_c_step(double** H_t, double** H_t1, double** W, int n, int k) {
    double** WH;
    double** HT;
    double** HTH;
    double** HHTH;
    int i, j;

    WH = matrix_multiplication(W, H_t, n, n, k);
    HT = transpose(H_t, n, k);
    HTH = matrix_multiplication(HT, H_t, k, n, k);
    HHTH = matrix_multiplication(H_t, HTH, n, k, k);

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (HHTH[i][j] == 0) {
                HHTH[i][j] += DENOMINATOR_EPSILON;
            }

            H_t1[i][j] = H_t[i][j] * (1 - BETA + (BETA * (WH[i][j] / HHTH[i][j])));
        }
    }

    free_matrix(WH, n);
    free_matrix(HT, k);
    free_matrix(HTH, k);
    free_matrix(HHTH, n);
}


double** symnmf_c(double** H_0, double** W, int n, int k) {
    double** H_t;
    double** H_t1;
    double** delta;
    int i, j;
    int iter;

    H_t = malloc_matrix(n, k);
    H_t1 = malloc_matrix(n, k);
    delta = malloc_matrix(n, k);

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H_t[i][j] = H_0[i][j];
        }
    }

    for (iter = 0; iter < MAX_ITER; iter++) {
        symnmf_c_step(H_t, H_t1, W, n, k);
        
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                delta[i][j] = H_t1[i][j] - H_t[i][j];
            }
        }

        if (frobenius_norm(delta, n, k) < EPSILON) {
            break;
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_t[i][j] = H_t1[i][j];
            }
        }
    }

    free_matrix(H_t1, n);
    free_matrix(delta, n);

    return H_t;
}


double** proccess_input_file(char* file_name, int* n, int* d) {
    FILE* file;
    char c;
    double** A;
    double value;
    int i, j;
    
    file = fopen(file_name, "r");

    if (file == NULL) {
        printf("An Error Has Occurred3\n");
        exit(1);
    }

    *n = 0;
    *d = 0;
    
    while ((c = fgetc(file)) != EOF) {
        if (c == ',') {
            (*d)++;
        }
        else if (c == '\n') {
            (*n)++;
        }
    }

    *d /= *n;
    (*d)++;

    fseek(file, 0, SEEK_SET);

    A = malloc_matrix(*n, *d);
        
    for (i = 0; i < *n; i++) {
        for (j = 0; j < *d; j++) {
            if (fscanf(file, "%lf", &value) != 1) {
                printf("An Error Has Occurred4\n");
                exit(1);
            }

            A[i][j] = value;
            
            /* skip commas and newlines */
            fgetc(file);
        }
    }

    fclose(file);

    return A;
}


void print_matrix(double** A, int n, int m) {
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%.4f", A[i][j]);

            if (j < m - 1) {
                printf(",");
            }
        }

        printf("\n");
    }
}


int main(int argc, char* argv[]) {
    char* goal;
    char* file_name;
    double** X;
    double** result;
    int n, d;
    
    if (argc != 3) {
        printf("Usage: ./symnmf <goal> <file_name>\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];
    
    X = proccess_input_file(file_name, &n, &d);

    if (strcmp(goal, "sym") == 0) {
        result = sym_c(X, n, d);
    }
    else if (strcmp(goal, "ddg") == 0) {
        result = ddg_c(X, n, d);
    }
    else if (strcmp(goal, "norm") == 0) {
        result = norm_c(X, n, d);
    }
    else {
        printf("An Error Has Occurred5\n");
        exit(1);
    }

    print_matrix(result, n, n);
    free_matrix(result, n);
    
    return 0;
}
