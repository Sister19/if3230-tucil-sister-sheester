#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m,int world_size, int world_rank) {
    int i;
    int j;
    int k;
    int x = 2;
    if(world_rank==0){
        scanf("%d", &(m->size));
        for (k = 1; k < world_size; k++)
        {   
            MPI_Send(&(m->size),1,MPI_INT,k,0,MPI_COMM_WORLD); 
        }
        for (i = 0; i < m->size; i++){
            for (j = 0; j < m->size; j++){
                scanf("%lf", &(m->mat[i][j]));

                for (int k = 1;k < world_size; k++)
                {
                    MPI_Send(&(m->mat[i][j]),1,MPI_DOUBLE,k,0,MPI_COMM_WORLD); 
                }
            }
        }
                
    }else{
        MPI_Recv(&(m->size),1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for (i = 0; i < m->size; i++)
            for (j = 0; j < m->size; j++){
                {
                    MPI_Recv(&(m->mat[i][j]),1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
            }
        
    }
   
}

   double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = 0.0;
    for (int m = 0; m < mat->size; m++) {
        for (int n = 0; n < mat->size; n++) {
            double complex arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element += mat->mat[m][n] * exponent;
        }
    }
    return element / (double) (mat->size*mat->size);
    }
// double complex dft(struct Matrix *mat, int k, int l, int world_rank,int world_size ){
//     double complex element = 0.0;
//     double complex add; 
    
 
    // if(world_rank!=0){
    //     for (int m = 0; m < mat->size; m++) {
    //         for (int n = 0; n < mat->size; n++) {
    //             if(((m*mat->size)+n)%(3) == world_rank-1 ){
    //                 double complex arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
    //                 double complex exponent = cexp(-2.0I * M_PI * arg);

    //                 add = mat->mat[m][n] * exponent;
    //                 MPI_Send(&add,1,MPI_C_DOUBLE_COMPLEX,0,0,MPI_COMM_WORLD); 
    //             }
    //         }
    //     }
    //     return 0.0;
    // }else{
        
    //     for (int i = 0; i < (mat->size)*(mat->size); i++)
    //     {   
    //         MPI_Recv(&add,1,MPI_C_DOUBLE_COMPLEX,((i)%(3)+1),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //         element +=add;
    //     }
        
    //     // for (int m = 0; m < mat->size; m++) {
    //     //     for (int n = 0; n < mat->size; n++) {
    //     //         MPI_Recv(&add,1,MPI_C_DOUBLE_COMPLEX,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //     //         element +=add;
    //     //     }
    //     // }
    //     return element / (double) (mat->size*mat->size);
    // }

    
// }



int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;
        
    MPI_Init(NULL, NULL);
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    readMatrix(&source,world_size,world_rank);
    freq_domain.size = source.size;

    if(world_rank==0){
        for (int k = 0; k < source.size; k++){
            for (int l = 0; l < source.size; l++){
                MPI_Recv(&(freq_domain.mat[k][l]),1,MPI_C_DOUBLE_COMPLEX,(k*source.size+l)%(3)+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
    }else{
        for (int k = 0; k < source.size; k++){
            for (int l = 0; l < source.size; l++){
                if(((k*source.size)+l)%(3) == world_rank-1 ){
                    freq_domain.mat[k][l]= dft(&source, k, l);
                    MPI_Send(&(freq_domain.mat[k][l]),1,MPI_C_DOUBLE_COMPLEX,0,0,MPI_COMM_WORLD);
                }
            }
        }
    }
    
    if(world_rank==0){
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
    }

    MPI_Finalize();
    return 0;
}