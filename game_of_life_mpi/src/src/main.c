/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "sys/time.h"

#include "../include/game-of-life.h"

// to set the size of the board later

int main (int argc, char *argv[]) {
    int *board, *newboard, i;

    if (argc != 6) { // Check if the command line arguments are correct
        printf("Usage: %s Ν1 Ν2 thres disp iter thr t\n"
                       "where\n"
                       "  N1    : number of board's rows\n"
                       "  N2    : number of board's columns\n"
                       "  thres : propability of alive cell\n"
                       "  disp  : {1: display output, 0: hide output}\n"
                       "  iter  : number of generations\n", argv[0]);
        return (1);
    }

    // Input command line arguments
    int N1 = atoi(argv[1]);                   // Rows number
    int N2 = atoi(argv[2]);                   // Column number
    double thres = atof(argv[3]);             // Probability of life cell
    int disp = atoi(argv[4]);                 // Display output?
    int t = atoi(argv[5]);                    // Three generations on each execution

    printf("Size %dx%d with propability: %0.1f%%\n", N1, N2, thres * 100);

    int  tasks_num, rank, init_flag; //vars needed for MPI

    struct timeval startwtime, endwtime;

    init_flag = MPI_Init(&argc, &argv); //initiate MPI execution environment
    if (init_flag != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, init_flag);
    }


    //TIC
    gettimeofday (&startwtime, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("This is rank: %i of %i tasks_num \n", rank, tasks_num);

    board = NULL;
    newboard = NULL;
    //each process works on a different part of the overall matrix
    //so we slice the board horizontally, depending of the number of tasks
    //we do that by changing N1 when the number of processes are more than one
    if (tasks_num != 1){
        N1 = N1/tasks_num + 2; //we need to create a new row on the top and on the bottom for information exchange (cyclic boundary conditions)
    }
    board = (int *) malloc(N1 * N2 * sizeof(int));
    if (board == NULL) {
        printf("\nERROR: Memory allocation did not complete successfully!\n");
        return (1);
    }

    /* second pointer for updated result */
    newboard = (int *) malloc(N1 * N2 * sizeof(int));

    if (newboard == NULL) {
        printf("\nERROR: Memory allocation did not complete successfully!\n");
        return (1);
    }

    /*functions contained in init.c */
    //initialize_board(board, N1, N2); //useless
    generate_table(board, N1, N2, thres, tasks_num);
    printf("Board generated\n");

    /* play game of life t times */
    /* t generations */

    // each rank executes play on the slice of the board it was assigned
    for (i = 0; i < t; i++) { //can't parallelize  that, it's the generations
        if (disp) {
            display_table(board, N1, N2);//display the new condition of the board
        }
        play(board, newboard, N1, N2, rank, tasks_num);
    }


    printf("Game finished after %d generations.\n\n", t);
    //TOK
    gettimeofday (&endwtime, NULL);
    double total_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    //free space
    free(board);
    free(newboard);
    MPI_Finalize(); // terminate MPI execution environment

    printf("Total execution time for all processes : %fs\n", total_time);

    return(0);

}