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
    /* Allocate memory for a n * m matrix of doubles */

    double** A;
    int i;

    /* Allocate memory for an array of pointers to the cells and check for errors */
    A = (double**)malloc(n * sizeof(double*));
    if (A == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }

    /* Allocate the memory line by line */
    for (i = 0; i < n; i++) {
        A[i] = (double*)malloc(m * sizeof(double));
        if (A[i] == NULL) {
            printf("An Error Has Occurred\n");
            exit(1);
        }
    }

    return A;
}


void free_matrix(double** A, int n) {
    /* Free all memory used by a matrix */
    
    int i;

    /* Go through every line and free its memory */
    for (i = 0; i < n; i++) {
        free(A[i]);
    }

    free(A);
}


double** transpose(double** A, int n, int m) {
    /* Transpose an n x m matrix */

    double** B;
    int i, j;

    /* Allocate memory for matrix */
    B = malloc_matrix(m, n);

    /* Calculate the transpose of the given matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            B[j][i] = A[i][j];
        }
    }

    return B;
}


double** matrix_multiplication(double** A, double** B, int n, int r, int m) {
    /* Multiply to matrices of size n x r and r x m, respectivley */
    double** C;
    int i, j, k;

    /* Allocate memory for matrix */
    C = malloc_matrix(n, m);

    /* Calculate matrix multiplication */
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
    /* Euclidean distance squared between two points of length d */

    double result;
    int i;

    /* Sum starts at 0 */
    result = 0;

    /* Calculate distance between two data points */
    for (i = 0; i < d; i++) {
        result += pow((a[i] - b[i]), 2);
    }

    return result;
}


double frobenius_norm(double** A, int n, int m) {
    /* Frobenius norm squared of a matrix of size n x m */
    double result;
    int i, j;

    /* Sum starts at 0 */
    result = 0;

    /* Calculate sum of squares of the cells of the given matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            result += pow(A[i][j], 2);
        }
    }

    return result;
}


double** sym_c(double** X, int n, int d) {
    /* Calculate the similarity matrix */

    double** A;
    int i, j;

    /* Allocate memory for matrix */
    A = malloc_matrix(n, n);

    /* Calculate the similarity matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) { /* 0 if on the main diagonal */
                A[i][j] = 0;
            }
            else { /* Otherwise, calculate similarity */
                A[i][j] = exp(-(euclidean_distance(X[i], X[j], d))/2);
            }
        }
    }

    return A;
}


double** ddg_c(double** X, int n, int d) {
    /* Calculate the diagonal degree matrix */

    double** D;
    double** A;
    int i, j;
    double sum;

    /* Calculate similarity matrix */
    A = sym_c(X, n, d);

    /* Allocate memory for matrix */
    D = malloc_matrix(n, n);

    /* Initialize matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][j] = 0;
        }
    }

    /* Calculate the D matrix */
    for (i = 0; i < n; i++) {
        /* Calculate the sum for each column */
        sum = 0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }

        /* Set the value on the diagonal to be the sum */
        D[i][i] = sum;
    }

    /* Free the memory */
    free_matrix(A, n);

    return D;
}


double** norm_c(double** X, int n, int d) {
    /* Calculate the normalized similarity matrix */

    double** W;
    double** D;
    double** A;
    int i, j;
    double denominator;

    /* Calculate A and D matrices */
    A = sym_c(X, n, d);
    D = ddg_c(X, n, d);

    /* Allocate memory for matrix */
    W = malloc_matrix(n, n);

    /* Calculate W */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* Calculate the denominator (D^-1/2 is diagonal so we get that this needs to be divided by to get
            D ^ -1/2 * A * D ^ -1/2) */
            denominator = sqrt(D[i][i] * D[j][j]);
            if (denominator == 0) { /* cant divide by 0, make it a small epsilon */
                denominator = DENOMINATOR_EPSILON;
            }

            /* Calculate the value in W */
            W[i][j] = A[i][j] / denominator;
        }
    }

    /* Free memory */
    free_matrix(A, n);
    free_matrix(D, n);

    return W;
}


void symnmf_c_step(double** H_t, double** H_t1, double** W, int n, int k) {
    /* Calculate a step in symnmf */

    double** WH;
    double** HT;
    double** HTH;
    double** HHTH;
    int i, j;

    /* Calculate W * H */
    WH = matrix_multiplication(W, H_t, n, n, k);
    /* Calculate H transposed (H^t)*/
    HT = transpose(H_t, n, k);
    /* Calculate H^t * H */
    HTH = matrix_multiplication(HT, H_t, k, n, k);
    /* Calculate H * H^t * H */
    HHTH = matrix_multiplication(H_t, HTH, n, k, k);

    /* Calculate one step of symNMF */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (HHTH[i][j] == 0) { /* cant divide by 0, make it epsilon */
                HHTH[i][j] += DENOMINATOR_EPSILON;
            }

            /* Calculate the cell in new H */
            H_t1[i][j] = H_t[i][j] * (1 - BETA + (BETA * (WH[i][j] / HHTH[i][j])));
        }
    }

    /* Free memory */
    free_matrix(WH, n);
    free_matrix(HT, k);
    free_matrix(HTH, k);
    free_matrix(HHTH, n);
}


double** symnmf_c(double** H_0, double** W, int n, int k) {
    /* Find an optimized H */
    double** H_t;
    double** H_t1;
    double** delta;
    int i, j;
    int iter;
    H_t = malloc_matrix(n, k);
    H_t1 = malloc_matrix(n, k);
    delta = malloc_matrix(n, k);
    /* Initialize H_t to be H_0 */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H_t[i][j] = H_0[i][j];
        }
    }
    /* Do symnmf step until convergence or max_iter reached */
    for (iter = 0; iter < MAX_ITER; iter++) {
        symnmf_c_step(H_t, H_t1, W, n, k);
        /* Calculate difference between H_t1 and H_t for frobenius norm */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                delta[i][j] = H_t1[i][j] - H_t[i][j];
            }
        }
        /* Move H_t1 to H_t before convergence check */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_t[i][j] = H_t1[i][j];
            }
        }
        /* Check convergence */
        if (frobenius_norm(delta, n, k) < EPSILON) {
            break;
        }
    }
    free_matrix(H_t1, n);
    free_matrix(delta, n);
    return H_t;
}

void calculate_dimensions(FILE* file, int* n, int* d) {
    /* Calculate the dimensions of a matrix from file */

    char c;

    /* Check if file was opened correctly */
    if (file == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }

    /* Initialize dimensions */
    *n = 0;
    *d = 0;

    /* Count the number of ',' and '\n' to calculate the dimensions from */
    while ((c = fgetc(file)) != EOF) {
        if (c == ',') {
            (*d)++;
        }
        else if (c == '\n') {
            (*n)++;
        }
    }

    /* Calculate the number of elements in a line, the amount of ',' divided by the number of lines + 1*/
    *d /= *n;
    (*d)++;
}


double** proccess_input_file(char* file_name, int* n, int* d) {
    /* Proccess an input file */
    FILE* file;
    double** A;
    double value;
    int i, j;
    
    /* Open the wanted file */
    file = fopen(file_name, "r");

    /* Calculate the dimensions of the matrix in the file */
    calculate_dimensions(file, n, d);

    /* Start reading from the beginning of the file after calculating the dimensions */
    fseek(file, 0, SEEK_SET);

    /* Allocate space to save the matrix from the file to */
    A = malloc_matrix(*n, *d);
    
    /* Load values into A */
    for (i = 0; i < *n; i++) {
        for (j = 0; j < *d; j++) {
            if (fscanf(file, "%lf", &value) != 1) {
                printf("An Error Has Occurred\n");
                fclose(file);
                exit(1);
            }

            A[i][j] = value;
            
            /* skip commas and newlines */
            fgetc(file);
        }
    }

    /* Close the file */
    fclose(file);

    return A;
}


void print_matrix(double** A, int n, int m) {
    /* Print an n x m matrix */
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
    
    /* Check correct number of args */
    if (argc != 3) {
        printf("Usage: ./symnmf <goal> <file_name>\n");
        return 1;
    }

    /* Proccess args*/
    goal = argv[1];
    file_name = argv[2];
    /* Get matrix from input file */
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
        printf("An Error Has Occurred\n");
        exit(1);
    }

    /* Print the result matrix */
    print_matrix(result, n, n);
    /* Free memory */
    free_matrix(result, n);
    free_matrix(X, n);
    return 0;
}
