#include <cstdio>
#include <cstdlib>

// Test one function from BLAS and one function LAPACK
// Contains code from
// https://www.ibm.com/support/knowledgecenter/en/SSLTBW_2.1.0/com.ibm.zos.v2r1.cbcpx01/atlasexample1.htm
// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgels_ex.c.htm
extern "C"
{
    int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, 
               double *, int *, double *, double *, int *);

    // DGELS prototype 
    void dgels(char* trans, int* m, int* n, int* nrhs, double* a, int* lda,
        double* b, int* ldb, double* work, int* lwork, int* info);
}

/* Auxiliary routine: printing a matrix */
void print_matrix(char* desc, int m, int n, double* a, int lda) {
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.2f", a[i + j*lda]);
        printf("\n");
    }
}

/* Auxiliary routine: printing norms of matrix columns */
void print_vector_norm(char* desc, int m, int n, double* a, int lda) {
    int i, j;
    double norm;
    printf("\n %s\n", desc);
    for (j = 0; j < n; j++) {
        norm = 0.0;
        for (i = 0; i < m; i++) norm += a[i + j*lda] * a[i + j*lda];
        printf(" %6.2f", norm);
    }
    printf("\n");
}

void dgemm_test()
{
    double A[9] = {1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0};

    double B[9] = {2.276, 1.926, 1.978,
                   2.013, 1.928, 2.270,
                   2.148, 2.387, 1.165};

    double C[9] = {0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0};

    char transA = 'N';
    char transB = 'N';
    double one = 1.0;
    double zero = 0.0;
    int rowsA = 3;
    int colsA = 3;
    int colsB = 3;

    dgemm_(&transA, &transB, &rowsA, &colsB, &colsA, &one, A, &rowsA, B, &colsA, &zero, C, &rowsA);
}

/* Parameters */
#define M 6
#define N 4
#define NRHS 2
#define LDA M
#define LDB M

void dgels_test()
{
    /* Locals */
    int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork;
    double wkopt;
    double* work;
    /* Local arrays */
    double a[LDA*N] = {
        1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
        -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
        -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
        4.53,  3.83, -6.64,  2.06, -2.47,  4.70
    };
    double b[LDB*NRHS] = {
        8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
        9.35, -4.43, -0.70, -0.26, -7.36, -2.52
    };
    /* Executable statements */
    printf(" DGELS Example Program Results\n");
    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgels("No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
        &info);
    lwork = (int)wkopt;
    work = (double*)malloc(lwork * sizeof(double));
    /* Solve the equations A*X = B */
    dgels("No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
        &info);
    /* Check for the full rank */
    if (info > 0) {
        printf("The diagonal element %i of the triangular factor ", info);
        printf("of A is zero, so that A does not have full rank;\n");
        printf("the least squares solution could not be computed.\n");
        exit(1);
    }
    /* Print least squares solution */
    print_matrix("Least squares solution", n, nrhs, b, ldb);
    /* Print residual sum of squares for the solution */
    print_vector_norm("Residual sum of squares for the solution", m - n, nrhs,
        &b[n], ldb);
    /* Print details of QR factorization */
    print_matrix("Details of QR factorization", m, n, a, lda);
    /* Free workspace */
    free((void*)work);
}

int main()
{
    dgemm_test();
    dgels_test();
}