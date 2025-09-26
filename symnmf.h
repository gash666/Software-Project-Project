void free_matrix(double** A, int n);
double** sym_c(double** X, int n, int d);
double** ddg_c(double** X, int n, int d);
double** norm_c(double** X, int n, int d);
double** symnmf_c(double** H_0, double** W, int n, int k);

const double EPSILON = 1e-4;
const int MAX_ITER = 300;
const double DENOMINATOR_EPSILON = 1e-6;
const double BETA = 0.5;