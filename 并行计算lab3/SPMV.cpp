// mpicxx -o test test.cpp
// mpirun -np 2 ./test 1000
// M * N = P
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include"mpi.h"
#include<chrono>

using namespace std;
#define EPS 0.001
int dimension;

void gemm_baseline(float **A, float *B, float *C) {
    for (auto i = 0; i < dimension; i++){
        C[i] = 0.0;
        for (auto k = 0; k < dimension; k++)
            C[i] += A[i][k] * B[k];
    }
}

bool verify(float *A, float *B) {
    for (auto i = 0; i < dimension; i++)
        if (abs(A[i] - B[i]) > EPS){
            cout << "Error!" << endl;
            return 0;
        }
    return 1;
}

void Init(float *a) {
    int i, j;
    srand(((int) time(0)) * 100);
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            a[i * dimension + j] = rand() % 2;
        }
    }
}

void Init_2(float *a) {
    int i, j;
    srand(((int) time(0)) * 100);
    for (i = 0; i < dimension; i++) {
       a[i] = rand() % 2;
    }
}

int main(int argc, char *argv[]) {

    float *M, *N, *P, *buffer, *ans;
    int my_rank, numprocs;
    int m = 1, i, line, j, k;
    float temp = 0.0;
    double start_time, end_time, sub_start, sub_end;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (argc != 2) {
        if (my_rank == 0)
            printf("Error!\n");
        MPI_Finalize();
        return 0;
    }

    dimension = atoi(argv[1]);
    line = dimension / (numprocs - 1);
    M = (float *) malloc(sizeof(float) * dimension * dimension);
    N = (float *) malloc(sizeof(float) * dimension);
    P = (float *) malloc(sizeof(float) * dimension);

    buffer = (float *) malloc(sizeof(float) * dimension * line);
    ans = (float *) malloc(sizeof(float) * line);

    if (my_rank == 0) {

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        printf("The num of process is %d\n", numprocs);
        printf("The DIM is %d\n", dimension);
        Init(M);
        Init_2(N);

        for (i = 1; i < numprocs; i++)
            MPI_Send(N, dimension, MPI_FLOAT, i, 0, MPI_COMM_WORLD);

        for (m = 1; m < numprocs; m++)
            MPI_Send(M + (m - 1) * line * dimension, dimension * line, MPI_FLOAT, m, 1, MPI_COMM_WORLD);


        for (k = 1; k < numprocs; k++) {
            MPI_Recv(ans, line, MPI_FLOAT, k, 3, MPI_COMM_WORLD, &status);
            for (i = 0; i < line; i++) {
                P[(k - 1) * line + i] = ans[i];
            }
        }

        for (i = (numprocs - 1) * line; i < dimension; i++) {
            temp = 0.0;
            for (k = 0; k < dimension; k++)
                temp += M[i * dimension + k] * N[k];
            P[i] = temp;
        }

//        for(auto l = 0; l < dimension; l++)
//            cout << P[l] << " ";
//        cout << endl;

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        cout << "my_rank = " << my_rank << " time = " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;
    }
    else {
        sub_start = MPI_Wtime();
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        MPI_Recv(N, dimension, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(buffer, dimension * line, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);

        for (i = 0; i < line; i++) {
            temp = 0.0;
            for (k = 0; k < dimension; k++)
                temp += buffer[i * dimension + k] * N[k];
            ans[i] = temp;

        }
        MPI_Send(ans, line, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
        sub_end = MPI_Wtime();
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        cout << "my_rank = " << my_rank << " time = " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;
    }
    MPI_Finalize();
    return 0;
}