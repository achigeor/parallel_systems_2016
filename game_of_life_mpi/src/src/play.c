#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include "sys/time.h"


#include "../include/game-of-life.h"

void play (int *board, int *newboard, int N1, int N2, int rank, int tasks_num) {
    /*
      (copied this from some web page, hence the English spellings...)
      1.STASIS : If, for a given cell, the number of on neighbours is
      exactly two, the cell maintains its status quo into the next
      generation. If the cell is on, it stays on, if it is off, it stays off.
      2.GROWTH : If the number of on neighbours is exactly three, the cell
      will be on in the next generation. This is regardless of the cell's
      current state.
      3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will
      be off in the next generation.
    */

    int i, j, a;
    omp_set_num_threads(8); // number of threads for openMP
    char my_cpu_name[BUFSIZ];
    int my_name_length;
    MPI_Get_processor_name(my_cpu_name, &my_name_length);

    printf("This is processor %s with rank %d\n", my_cpu_name, rank);

    if (tasks_num == 1){ //no MPI here, only one task
        /* for each cell, apply the rules of Life */
#pragma omp parallel for default(none) shared(newboard,board, N2, N1) private(i, a, j) collapse(2)
        for (i = 0; i < N1; i++) {
            for (j = 0; j < N2; j++) {
                a = adjacent_to(board, i, j, N1, N2);
                if (a == 2) NewBoard(i, j) = Board(i, j);
                if (a == 3) NewBoard(i, j) = 1;
                if (a < 2) NewBoard(i, j) = 0;
                if (a > 3) NewBoard(i, j) = 0;
            }
        }
        /* copy the new board back into the old board */
#pragma omp parallel for default(none) shared(newboard, board, N2, N1) private(i, j) collapse(2)
        for (i=0; i < N1; i++) {
            for (j = 0; j < N2; j++) {
                Board(i, j) = NewBoard(i, j);
            }
        }
    }
    else { // MPI code here

        int tag1 = 1;
        int tag2 = 2;

        // set previous and next rank to facilitate communication between them
        int prev = rank - 1;
        int next = rank + 1;

        // cyclic boundary conditions
        if (rank == 0) prev = tasks_num - 1;
        if (rank == (tasks_num - 1)) next = 0;

        MPI_Request reqs[4];
        MPI_Status stats[4];

        //send and receive buffers
        int out1[N2], out2[N2], in1[N2], in2[N2];

#pragma omp parallel for default(none) shared(board, N1, N2, out1, out2) private(j)
        for (j = 0; j < N2; j++) {
            out1[j] = Board(0, j);
            out2[j] = Board(N1 - 1, j);
        }

        //Send data to the previous slice of board
        MPI_Isend(out1, N2, MPI_INT, prev, tag1, MPI_COMM_WORLD, &reqs[0]);

        //Send data to the next slice of board
        MPI_Isend(out2, N2, MPI_INT, next, tag2, MPI_COMM_WORLD, &reqs[1]);

        //Receive data from the previous slice of board
        MPI_Irecv(in2, N2, MPI_INT, prev, tag2, MPI_COMM_WORLD, &reqs[2]);

        //Receive data from the next slice of board
        MPI_Irecv(in1, N2, MPI_INT, next, tag1, MPI_COMM_WORLD, &reqs[3]);



        /* for each cell, apply the rules of Life */
#pragma omp parallel for default(none) shared(board, N2, N1, newboard) private(i, j, a) collapse(2)
        //do NOT include the first and last row in play this time since they are used for the communication between tasks
        //asynchronous application of rules of life with respect to the whole board
        //meaning that each task applies the rules of life on its slice's cells minus "linking" rows, while waiting for the communication to end
        for (i = 2; i < N1-2; i++){
            for (j = 0; j < N2; j++){

                a = adjacent_to(board, i, j, N1, N2);
                if (a == 2) NewBoard(i, j) = Board(i, j);
                if (a == 3) NewBoard(i, j) = 1;
                if (a < 2) NewBoard(i, j) = 0;
                if (a > 3) NewBoard(i, j) = 0;

            }
        }

        // ensure all communication is finished
        MPI_Waitall(4, reqs, stats);

#pragma omp parallel for default(none) shared(board, newboard, N1, N2, in1, in2) private(j, a)
        for (j = 0; j < N2; j++){

            // copy what i received to my board slice (first and last row)
            Board(N1 - 1, j) = in1[j];
            Board(0, j) = in2[j];

            //calculate neighbors for the received rows

            //first row
            a = adjacent_to(board, 1, j, N1, N2);
            if (a == 2) NewBoard(1, j) = Board(1, j);
            if (a == 3) NewBoard(1, j) = 1;
            if (a < 2) NewBoard(1, j) = 0;
            if (a > 3) NewBoard(1, j) = 0;

            //last row
            a = adjacent_to(board, N1-1, j, N1, N2);
            if (a == 2) NewBoard(N1-1, j) = Board(N1-1, j);
            if (a == 3) NewBoard(N1-1, j) = 1;
            if (a < 2) NewBoard(N1-1, j) = 0;
            if (a > 3) NewBoard(N1-1, j) = 0;

        }

        /* copy the new board back into the old board */
#pragma omp parallel for default(none) shared(board, newboard, N2, N1) private(i, j, a) collapse(2)
        for (i=1; i < N1-1; i++) {
            for (j = 0; j < N2; j++){
                Board(i, j) = NewBoard(i, j);
            }
        }

    }

} // end of play function
