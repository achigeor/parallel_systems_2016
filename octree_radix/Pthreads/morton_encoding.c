#include "stdio.h"
#include "stdlib.h"
#include <pthread.h>
#include "string.h"
#include "utils.h"


#define DIM 3

typedef struct{
    int N;
    unsigned long int *mcodes;
    unsigned int *codes;
    long start;
    long end;
}Codes;


inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

void* parallel_morton_codes(void* arg){

    Codes *my_codes = (Codes*) arg;
    //my_codes = malloc((my_codes->N)*sizeof(Codes));

    for (long i = my_codes->start ; i<my_codes->end ; ++i){

        my_codes->mcodes[i] = mortonEncode_magicbits(
                my_codes->codes [i*DIM],
                my_codes->codes [i*DIM +1],
                my_codes->codes [i*DIM +2]
        );
    }

    //free(my_codes);
    pthread_exit((void*) 0);
}


void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level){

    //Codes *gcodes;
    //gcodes = malloc(THREADS*sizeof(Codes));
    Codes gcodes[THREADS];
    //gcodes->mcodes=mcodes;

    pthread_t *threads = malloc(THREADS*sizeof(pthread_t));
    //pthread_t threads[THREADS];
    pthread_attr_t tattr;
    void* status;
    long i;

    pthread_attr_init(&tattr);
    pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);

    int chunk = N / THREADS;

    for (i = 0 ; i<THREADS ; ++i) {
        (gcodes+i)->N = chunk;

        long start = i * chunk;
        long end = (i+1) * chunk;
        (gcodes+i)->codes = codes;
        (gcodes+i)->mcodes = mcodes;
        (gcodes+i)->start = start;
        (gcodes+i)->end = end;


//        (gcodes + i)->codes = malloc(chunk * sizeof(unsigned int));
//
//        memcpy((gcodes + i)->codes, codes + start, chunk * sizeof(unsigned int));

        pthread_create(&threads[i],&tattr,parallel_morton_codes,(void*)&gcodes[i]);
    }

    pthread_attr_destroy(&tattr);
    for ( i = 0; i < THREADS; ++i)
    {
        pthread_join(threads[i], &status);

    }
    //free(gcodes);
    free(threads);
}
