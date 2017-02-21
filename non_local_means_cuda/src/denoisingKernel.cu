#include <math.h>
#include <stdio.h>

// Array access macros
#define b(i,j) B[(i) + (j)*m*m]

__global__ void Denoising(double *I,double *B, int m, int n, int patchSize, double filtSigma) {
        // Get pixel (x,y) in input
        // I: image as a vector
        // B: neighborCube
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int linearPixelId = i+j*m;
        double D = 0;

        if(linearPixelId<m*n) {
                int l=0;

                double sumI = 0;
                double Z = 0;
                // neighborhood index (row in B, start from top)
                for(l=0; l<m*n; l++) {
                        D = 0;
                        // neighbor pixel index, left to right
                        for(int k=0; k<patchSize*patchSize; k++) {
                                // calculate euclidean distance between neighborhoods (dissimilarity)
                                // and add them to sum
                                D+=(b(linearPixelId,k)-b(l,k))*(b(linearPixelId,k)-b(l,k));
                        }
                        Z +=  exp(-sqrt(D)*sqrt(D)/filtSigma);
                        sumI += exp(-sqrt(D)*sqrt(D)/filtSigma)*I[l];
                }
                I[linearPixelId] = sumI/Z;
        }
}
