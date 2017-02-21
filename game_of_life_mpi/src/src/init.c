#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#include "../include/game-of-life.h"

void generate_table(int *board, int N1, int N2, double threshold, int tasks_num) {
    int i, j;
    //int counter = 0;

    // use rand_r() because it is thread safe
    unsigned int seed;
    seed = time(NULL);
    // do not use openMP for initialize because rand() is not thread safe
#pragma omp parallel for default(none) shared(board, N1, N2, threshold, tasks_num) private(i ,j, seed) collapse(2)
    for (i = 0; i < N1; i++){
        for (j = 0; j < N2; j++){
            if ((i==0 || i==N1-1) && (tasks_num!=1) ) { //linking rows
                Board(i, j) = 0;
            }
            else{
                seed = seed ^ omp_get_thread_num(); //seed every thread from different point
                Board(i,j) = ((float)rand_r(&seed) / (float)RAND_MAX) < threshold;
            }
        }
    }
}


