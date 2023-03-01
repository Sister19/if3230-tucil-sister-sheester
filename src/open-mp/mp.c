#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    #pragma omp parallel for
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = 0.0, arg, exponent;
    #pragma omp parallel for private(arg) reduction(+:element)
    for (int m = 0; m < mat->size; m++) {
        for (int n = 0; n < mat->size; n++) {
            arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
            exponent = cexp(-2.0I * M_PI * arg);
            element += mat->mat[m][n] * exponent;
        }
    }
    return element / (double) (mat->size*mat->size);
}



int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    
    #pragma omp parallel for
    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = dft(&source, k, l);

    double complex sum = 0.0;
    double complex el;
    #pragma omp parallel for private(el) reduction(+:sum)
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
            sum += el;
        }
        printf("\n");
    }
    sum /= source.size;
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));

    return 0;
}
