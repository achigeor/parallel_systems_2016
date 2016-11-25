#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <pthread.h>
#include "utils.h"

#define MAXBINS 8

void truncated_radix_sort(unsigned long int *morton_codes,
                          unsigned long int *sorted_morton_codes,
                          unsigned int *permutation_vector,
                          unsigned int *index,
                          unsigned int *level_record,
                          int N,
                          int population_threshold,
                          int sft, int lv);

struct thread_data{
    unsigned long int *morton_codes;
    unsigned long int *sorted_morton_codes;
    unsigned int *permutation_vector;
    unsigned int *index;
    unsigned int *level_record;
    int N;
    int population_threshold;
    int sft;
    int lv;
};

struct thread_data *my_data;

pthread_mutex_t mutex_mortons = PTHREAD_MUTEX_INITIALIZER;

void *threaded_truncated_radix_sort(void *threadarg){

  my_data = (struct thread_data *) threadarg;
  truncated_radix_sort(my_data->morton_codes, my_data->sorted_morton_codes, my_data->permutation_vector, my_data->index, my_data->level_record, my_data->N, my_data->population_threshold, my_data->sft, my_data->lv);

}

inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void truncated_radix_sort(unsigned long int *morton_codes,
                          unsigned long int *sorted_morton_codes,
                          unsigned int *permutation_vector,
                          unsigned int *index,
                          unsigned int *level_record,
                          int N,
                          int population_threshold,
                          int sft, int lv){

  int rc;
  void *status;

  struct thread_data thread_data_array[NUM_THREADS];
  pthread_t threads[NUM_THREADS];

  pthread_attr_t attribute;

  pthread_attr_init(&attribute);
  pthread_attr_setdetachstate(&attribute, PTHREAD_CREATE_JOINABLE);

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;

  if(thread_data_array->N<=0){

    return;
  }
  else if(thread_data_array->N<=population_threshold || thread_data_array->sft < 0) { // Base case. The node is a leaf

    thread_data_array->level_record[0] = thread_data_array->lv; // record the level of the node
    memcpy(thread_data_array->permutation_vector, thread_data_array->index, thread_data_array->N*sizeof(unsigned int)); // Copy the permutation vector
    memcpy(thread_data_array->sorted_morton_codes, thread_data_array->morton_codes, thread_data_array->N*sizeof(unsigned long int)); // Copy the Morton codes

    return;
  }
  else{

    thread_data_array->level_record[0] = thread_data_array->lv;
    // Find which child each point belongs to
    for(int j=0; j<N; j++){
      unsigned int ii = (thread_data_array->morton_codes[j]>>thread_data_array->sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code) //hope it's false
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }

    for(int j=0; j<thread_data_array->N; j++){
      unsigned int ii = (thread_data_array->morton_codes[j]>>thread_data_array->sft) & 0x07;
      thread_data_array->permutation_vector[BinCursor[ii]] = thread_data_array->index[j];
      thread_data_array->sorted_morton_codes[BinCursor[ii]] = thread_data_array->morton_codes[j];
      BinCursor[ii]++;
    }

    //swap the index pointers
    swap(&thread_data_array->index, &thread_data_array->permutation_vector);

    //swap the code pointers
    swap_long(&thread_data_array->morton_codes, &thread_data_array->sorted_morton_codes);

    int i;
    /* Call the function recursively to split the lower levels */
    for(i=0; i<NUM_THREADS; i++) {
      int offset = (i > 0) ? BinSizes[i - 1] : 0;
      int size = BinSizes[i] - offset;
      thread_data_array[i].morton_codes = &morton_codes[offset];
      thread_data_array[i].sorted_morton_codes = &sorted_morton_codes[offset];
      thread_data_array[i].permutation_vector = &permutation_vector[offset];
      thread_data_array[i].index = &index[offset];
      thread_data_array[i].level_record = &level_record[offset];
      thread_data_array[i].N = size;
      thread_data_array[i].population_threshold = population_threshold;
      thread_data_array[i].sft = sft - 3;
      thread_data_array[i].lv = lv + 1;

      rc = pthread_create(&threads[i], &attribute , threaded_truncated_radix_sort, (void *) &thread_data_array[i]);

      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }

    }

    pthread_attr_destroy(&attribute);
    int k=0;

    for (k=0; k<NUM_THREADS; k++){
      rc = pthread_join(threads[k], &status);
      if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }

      /* truncated_radix_sort(&morton_codes[offset],
                           &sorted_morton_codes[offset],
                           &permutation_vector[offset],
                           &index[offset], &level_record[offset],
                           size,
                           population_threshold,
                           sft-3, lv+1);      */
    }
  }
}