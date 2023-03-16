#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    cuDoubleComplex mat[MAX_N][MAX_N];
};

cuDoubleComplex cuCexp(cuDoubleComplex x)
{
 double factor = exp(x.x);
 return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

cuDoubleComplex dft(struct Matrix *mat, int k, int l) {
    cuDoubleComplex element = make_cuDoubleComplex(0.0, 0.0);
    for (int m = 0; m < mat->size; m++) {
        for (int n = 0; n < mat->size; n++) {
            cuDoubleComplex arg      = make_cuDoubleComplex(k*m / (double) mat->size + l*n / (double) mat->size, 0.0);
            cuDoubleComplex exponent = cuCexp(make_cuDoubleComplex(0.0, -2.0 * M_PI * arg.x));
            cuDoubleComplex value = make_cuDoubleComplex(mat->mat[m][n], 0.0);
            element = cuCadd(element, cuCmul(value, exponent));
        }
    }
    element = cuCdiv(element, make_cuDoubleComplex(mat->size*mat->size, 0.0));
    return element;
}

int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    
    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = dft(&source, k, l);

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            cuDoubleComplex el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", el.x, el.y);
            sum = cuCadd(sum, el);
        }
        printf("\n");
    }
    sum = cuCdiv(sum, make_cuDoubleComplex(freq_domain.size, 0.0));
    printf("Average : (%lf, %lf)", sum.x, sum.y);

    return 0;
}




//alternatives

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};

__device__ double complex dft_element(struct Matrix *mat, int k, int l, int m, int n) {
    double complex arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
    double complex exponent = cexp(-2.0I * M_PI * arg);
    return mat->mat[m][n] * exponent;
}

__global__ void dft_kernel(struct Matrix *mat, struct FreqMatrix *freq_domain) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < freq_domain->size && l < freq_domain->size) {
        double complex element = 0.0;
        for (int m = 0; m < mat->size; m++) {
            for (int n = 0; n < mat->size; n++) {
                element += dft_element(mat, k, l, m, n);
            }
        }
        freq_domain->mat[k][l] = element / (double) (mat->size*mat->size);
    }
}

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;

    readMatrix(&source);
    freq_domain.size = source.size;

    // Allocate memory on the device
    struct Matrix     *d_source;
    struct FreqMatrix *d_freq_domain;
    cudaMalloc(&d_source, sizeof(struct Matrix));
    cudaMalloc(&d_freq_domain, sizeof(struct FreqMatrix));

    // Copy the data to the device
    cudaMemcpy(d_source, &source, sizeof(struct Matrix), cudaMemcpyHostToDevice);

    // Launch the kernel to compute the DFT
    int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(ceil(freq_domain.size / (float) block_size), ceil(freq_domain.size / (float) block_size));
    dft_kernel<<<dimGrid, dimBlock>>>(d_source, d_freq_domain);

    // Copy the results back to the host
    cudaMemcpy(&freq_domain, d_freq_domain, sizeof(struct FreqMatrix), cudaMemcpyDeviceToHost);

    double complex sum = 0.0;
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            double complex el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
            sum += el;
        }
        printf("\n");
    }
    sum /= source.size;
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));

    return 0;
}
