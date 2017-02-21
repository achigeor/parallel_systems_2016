#include <math.h>
#include <stdio.h>

// Array access macros
#define INPUT(i,j) A[(i) + (j)*(m+patchSize-1)]
#define OUTPUT(i,j) B[(i) + (j)*m*m]
#define FILTER(i) H[(i)]

__global__ void neighborCube(double const * const A, double *B, double *H, int m, int n, int patchSize) {
        // Get pixel (x,y) in input
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i>=((patchSize - 1) / 2) && i<=m+((patchSize - 1) / 2) && j>=((patchSize - 1) / 2) && j<=m+((patchSize - 1) / 2)) { // Only scan pixels of original image, skip padded region
                // Scan the neighbourhood of i,j pixel in an area patchSize x patchSize, starting from the row above, left to right
                for (int k = -(patchSize - 1) / 2; k <= (patchSize - 1) / 2; k++) { // RowAbove --> SameRow --> RowBelow
                        for (int l = -(patchSize - 1) / 2; l <= (patchSize - 1) / 2; l++) { // FarLeft --> Center --> FarRight
                                OUTPUT(i - ((patchSize - 1) / 2) + m * (j - ((patchSize - 1) / 2)),
                                       k + ((patchSize - 1) / 2) + (l + ((patchSize - 1) / 2)) * patchSize) = INPUT(i + k, j + l); // populate neighbor cube (as a matrix)
                                OUTPUT(i - ((patchSize - 1) / 2) + m * (j - ((patchSize - 1) / 2)),
                                       k + ((patchSize - 1) / 2) + (l + ((patchSize - 1) / 2)) * patchSize) *= (FILTER(k + ((patchSize - 1) / 2) + (l + ((patchSize - 1) / 2)) * patchSize)); // filter the value
                        }
                }
        }
}
