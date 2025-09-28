#ifndef SYMNMF_H
#define SYMNMF_H

void free_matrix(double** A, int n);
double** sym_c(double** X, int n, int d);
double** ddg_c(double** X, int n, int d);
double** norm_c(double** X, int n, int d);
double** symnmf_c(double** H_0, double** W, int n, int k);

#endif