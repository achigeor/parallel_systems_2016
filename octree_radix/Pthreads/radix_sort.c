#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <pthread.h>
#include "utils.h"

#define MAXBINS 8

typedef struct{
    unsigned long int *morton_codes;
    int sft;
    int *bin_sizes;
    int start;
    int end;
    pthread_spinlock_t *lock;

}bin_data;

typedef struct{
    unsigned long int *morton_codes;
    unsigned long int *sorted_morton_codes;
    unsigned int *permutation_vector;
    unsigned int *index;
    unsigned int *level_record;
    int N;
    int population_threshold;
    int sft;
    int lv;
}recursion_data;

void *parallel_bin_sizes(void* arg){

    bin_data *my_data = (bin_data*) arg;
    int sft = my_data->sft;
    int LocalBinSizes[MAXBINS] = {0};
    unsigned long int *morton_codes = my_data->morton_codes;

    for (int j = my_data->start ; j < my_data->end ; ++j)
    {
        unsigned int ii = ( morton_codes[j] >> sft) & 0x07;
        LocalBinSizes[ii]++;
    }

    // avoid racing conditions
    pthread_spin_lock(my_data->lock);

    for(int i = 0; i < MAXBINS; i++){
        (my_data->bin_sizes)[i]+=LocalBinSizes[i];
    }

    pthread_spin_unlock(my_data->lock);

}

void *parallel_radix_sort(void *arg)
{

    recursion_data *my_recursion_data = (recursion_data *)arg;
    truncated_radix_sort(my_recursion_data->morton_codes,
                         my_recursion_data->sorted_morton_codes,
                         my_recursion_data->permutation_vector,
                         my_recursion_data->index,
                         my_recursion_data->level_record,
                         my_recursion_data->N,
                         my_recursion_data->population_threshold,
                         my_recursion_data->sft,
                         my_recursion_data->lv);
} // End of pthread Radix Callback

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

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;


  if(N<=0){

    return;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes

    return;
  }
  else{
      level_record[0] = lv;
      if(lv<1){

          //printf("fuck");
          // Initiliaze the static array containing
          // the identities of threads.
          pthread_t threads[THREADS];
          // Declare an attribute for the above threads.
          pthread_attr_t tattr;
          void *status;

          // Declare the spin lock and initialize it.
          pthread_spinlock_t lock ;

          pthread_spin_init(&lock,PTHREAD_PROCESS_PRIVATE);

          // Initialize the attribute so that the threads we create are
          // joinable.
          pthread_attr_init(&tattr);
          pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);

          int chunk = N / THREADS ;

          bin_data binData[THREADS];

          // Initialize the data that will be passed to every thread.
          for ( long i = 0 ; i < THREADS ; i++)
          {
              (binData+i)->morton_codes = morton_codes ;
              (binData+i)->sft = sft ;
              (binData+i)->bin_sizes = &BinSizes[0] ;
              (binData+i)->lock = &lock;
              (binData+i)->start = i*chunk ;
              (binData+i)->end = (i+1)*chunk ;

              pthread_create(&threads[i] , &tattr , parallel_bin_sizes , (void*) &binData[i] );
          }

          pthread_attr_destroy(&tattr);
          for (long i = 0 ;i <THREADS ; i++){
              pthread_join( threads[i], &status);
          }

          // scan prefix (must change this code)
          int offset = 0;
          for(int i=0; i<MAXBINS; i++){
              int ss = BinSizes[i];
              BinCursor[i] = offset;
              offset += ss;
              BinSizes[i] = offset;
          }

          for(int j=0; j<N; j++){
              unsigned int ii = (morton_codes[j]>>sft) & 0x07;
              permutation_vector[BinCursor[ii]] = index[j];
              sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
              BinCursor[ii]++;
          }

          //swap the index pointers
          swap(&index, &permutation_vector);

          //swap the code pointers
          swap_long(&morton_codes, &sorted_morton_codes);

          // Initiliaze the static array containing
          // the threads that will be used to split the recursive calls
          // and perform them in parallel.
          pthread_t r_threads[MAXBINS];
          // Declare an attribute for the above threads.
          pthread_attr_t rattr;

          // Initialize the attribute so that the threads we create are
          // joinable.
          pthread_attr_init(&rattr);
          pthread_attr_setdetachstate(&rattr, PTHREAD_CREATE_JOINABLE);


          recursion_data rData[MAXBINS];

          for (long i = 0 ; i < MAXBINS ; i++ )
          {
              int offset = (i>0) ? BinSizes[i-1] : 0;
              int size = BinSizes[i] - offset;
              rData[i].morton_codes = &morton_codes[offset];
              rData[i].sorted_morton_codes = &sorted_morton_codes[offset];
              rData[i].permutation_vector = &permutation_vector[offset];
              rData[i].index = &index[offset];
              rData[i].level_record = &level_record[offset];
              rData[i].N = size;
              rData[i].population_threshold = population_threshold;
              rData[i].sft = sft - 3;
              rData[i].lv = lv + 1;

              pthread_create( &r_threads[i] , &rattr ,
                              parallel_radix_sort , (void*) &rData[i] );
          }

          pthread_attr_destroy(&rattr);

          for( long i = 0 ; i < MAXBINS ; i++)
          {
              // Join thread #threadIt .
              pthread_join( r_threads[i], &status);
          }
      }


      else{
          // Find which child each point belongs to
          for(int j=0; j<N; j++){
              unsigned int ii = (morton_codes[j]>>sft) & 0x07;
              BinSizes[ii]++;
          }

          // scan prefix (must change this code)
          int offset = 0;
          for(int i=0; i<MAXBINS; i++){
              int ss = BinSizes[i];
              BinCursor[i] = offset;
              offset += ss;
              BinSizes[i] = offset;
          }

          for(int j=0; j<N; j++){
              unsigned int ii = (morton_codes[j]>>sft) & 0x07;
              permutation_vector[BinCursor[ii]] = index[j];
              sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
              BinCursor[ii]++;
          }

          //swap the index pointers
          swap(&index, &permutation_vector);

          //swap the code pointers
          swap_long(&morton_codes, &sorted_morton_codes);

          /* Call the function recursively to split the lower levels */
          for(int i=0; i<MAXBINS; i++){
              int offset = (i>0) ? BinSizes[i-1] : 0;
              int size = BinSizes[i] - offset;

              truncated_radix_sort(&morton_codes[offset],
                                   &sorted_morton_codes[offset],
                                   &permutation_vector[offset],
                                   &index[offset], &level_record[offset],
                                   size,
                                   population_threshold,
                                   sft-3, lv+1);
          }




  }

//      // Initiliaze the static array containing
//      // the identities of threads.
//      pthread_t threads[THREADS];
//      // Declare an attribute for the above threads.
//      pthread_attr_t tattr;
//      void *status;
//
//      // Declare the spin lock and initialize it.
//      pthread_spinlock_t lock ;
//
//      pthread_spin_init(&lock,PTHREAD_PROCESS_PRIVATE);
//
//      // Initialize the attribute so that the threads we create are
//      // joinable.
//      pthread_attr_init(&tattr);
//      pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);
//
//      int chunk = N / THREADS ;
//
//      bin_data binData[THREADS];
//
//      // Initialize the data that will be passed to every thread.
//      for ( long i = 0 ; i < THREADS ; i++)
//      {
//          binData[i].morton_codes = morton_codes ;
//          binData[i].sft = sft ;
//          binData[i].bin_sizes = &BinSizes[0] ;
//          binData[i].lock = &lock;
//          binData[i].start = i*chunk ;
//          binData[i].end = (i+1)*chunk ;
//
//          pthread_create(&threads[i] , &tattr , parallel_bin_sizes , (void*) &binData[i] );
//      }
//
//      pthread_attr_destroy(&tattr);
//      for (long i = 0 ;i <THREADS ; i++){
//          pthread_join( threads[i], &status);
//      }


//    // Find which child each point belongs to
//    for(int j=0; j<N; j++){
//      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
//      BinSizes[ii]++;
//    }

//    // scan prefix (must change this code)
//    int offset = 0;
//    for(int i=0; i<MAXBINS; i++){
//      int ss = BinSizes[i];
//      BinCursor[i] = offset;
//      offset += ss;
//      BinSizes[i] = offset;
//    }
//
//    for(int j=0; j<N; j++){
//      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
//      permutation_vector[BinCursor[ii]] = index[j];
//      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
//      BinCursor[ii]++;
//    }
//
//    //swap the index pointers
//    swap(&index, &permutation_vector);
//
//    //swap the code pointers
//    swap_long(&morton_codes, &sorted_morton_codes);
//
//      if (lv<1){
//          // Initiliaze the static array containing
//          // the threads that will be used to split the recursive calls
//          // and perform them in parallel.
//          pthread_t r_threads[THREADS];
//          // Declare an attribute for the above threads.
//          pthread_attr_t rattr;
//          void* status;
//
//
//          // Initialize the attribute so that the threads we create are
//          // joinable.
//          pthread_attr_init(&rattr);
//          pthread_attr_setdetachstate(&rattr, PTHREAD_CREATE_JOINABLE);
//
//
//          recursion_data rData[THREADS];
//
//          for (long i = 0 ; i < THREADS ; i++ )
//          {
//              int offset = (i>0) ? BinSizes[i-1] : 0;
//              int size = BinSizes[i] - offset;
//              rData[i].morton_codes = &morton_codes[offset];
//              rData[i].sorted_morton_codes = &sorted_morton_codes[offset];
//              rData[i].permutation_vector = &permutation_vector[offset];
//              rData[i].index = &index[offset];
//              rData[i].level_record = &level_record[offset];
//              rData[i].N = size;
//              rData[i].population_threshold = population_threshold;
//              rData[i].sft = sft - 3;
//              rData[i].lv = lv + 1;
//
//              pthread_create( &r_threads[i] , &rattr ,
//                              parallel_radix_sort , (void*) &rData[i] );
//          }
//
//          pthread_attr_destroy(&rattr);
//
//          for( long i = 0 ; i < THREADS ; i++)
//          {
//              // Join thread #threadIt .
//              pthread_join( r_threads[i], &status);
//          }
//      }
//      else{
//          /* Call the function recursively to split the lower levels */
//          for(int i=0; i<THREADS; i++){
//              int offset = (i>0) ? BinSizes[i-1] : 0;
//              int size = BinSizes[i] - offset;
//
//              truncated_radix_sort(&morton_codes[offset],
//                                   &sorted_morton_codes[offset],
//                                   &permutation_vector[offset],
//                                   &index[offset], &level_record[offset],
//                                   size,
//                                   population_threshold,
//                                   sft-3, lv+1);
//          }
//
//      }
//
////      // Initiliaze the static array containing
////      // the threads that will be used to split the recursive calls
////      // and perform them in parallel.
////      pthread_t r_threads[MAXBINS];
////      // Declare an attribute for the above threads.
////      pthread_attr_t rattr;
////
////
////      // Initialize the attribute so that the threads we create are
////      // joinable.
////      pthread_attr_init(&rattr);
////      pthread_attr_setdetachstate(&rattr, PTHREAD_CREATE_JOINABLE);
////
////
////      recursion_data rData[MAXBINS];
////
////      for (long i = 0 ; i < MAXBINS ; i++ )
////      {
////          int offset = (i>0) ? BinSizes[i-1] : 0;
////          int size = BinSizes[i] - offset;
////          rData[i].morton_codes = &morton_codes[offset];
////          rData[i].sorted_morton_codes = &sorted_morton_codes[offset];
////          rData[i].permutation_vector = &permutation_vector[offset];
////          rData[i].index = &index[offset];
////          rData[i].level_record = &level_record[offset];
////          rData[i].N = size;
////          rData[i].population_threshold = population_threshold;
////          rData[i].sft = sft - 3;
////          rData[i].lv = lv + 1;
////
////          pthread_create( &r_threads[i] , &rattr ,
////                          parallel_radix_sort , (void*) &rData[i] );
////      }
////
////      pthread_attr_destroy(&rattr);
////
////      for( long i = 0 ; i < MAXBINS ; i++)
////      {
////          // Join thread #threadIt .
////          pthread_join( r_threads[i], &status);
////      }
//
////    /* Call the function recursively to split the lower levels */
////    for(int i=0; i<MAXBINS; i++){
////      int offset = (i>0) ? BinSizes[i-1] : 0;
////      int size = BinSizes[i] - offset;
////
////      truncated_radix_sort(&morton_codes[offset],
////			   &sorted_morton_codes[offset],
////			   &permutation_vector[offset],
////			   &index[offset], &level_record[offset],
////			   size,
////			   population_threshold,
////			   sft-3, lv+1);
////    }
//


  }
}

