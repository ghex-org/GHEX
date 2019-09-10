#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

int ongoing_comm = 0;

int main(int argc, char *argv[])
{
    int rank, peer_rank, mode;
    int niter, buff_size, inflight;
    MPI_Comm mpi_comm;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &rank);
    peer_rank = (rank+1)%2;

    {
	int **buffers = new int *[inflight];
	int *vals = new int[inflight];
	MPI_Request *req = new MPI_Request[inflight];
	
	/* allocate buffers and mark comm requests as available */
	for(int j=0; j<inflight; j++){
	    MPI_Alloc_mem(buff_size*sizeof(int), MPI_INFO_NULL, &buffers[j]);
	    for(int i=0; i<buff_size; i++) buffers[j][i] = 0;
	    req[j] = MPI_REQUEST_NULL;
	    vals[j] = peer_rank;
	}
	
	/* pre-post recv */
	if(rank==1){
	    for(int j=0; j<inflight; j++){
		MPI_Irecv(buffers[j], buff_size, MPI_INT, peer_rank, j, mpi_comm, req+j);
	    }
	}

	int i = 0, sent = 0, recv = 0;
	while(i<niter){

	    /* submit new comm requests for available / completed) entries */
	    if(rank==0){
		for(int j=0; j<inflight; j++){
		    if(req[j] == MPI_REQUEST_NULL){

			if(rank==0 && i%(niter/10)==0) fprintf(stderr, "%d iters\n", i);
			i++;
			sent++;
			ongoing_comm++;
			buffers[j][0] = vals[j]++;
			MPI_Isend(buffers[j], buff_size, MPI_INT, peer_rank, j, mpi_comm, req+j);
		    }
		    if(i==niter) break;
		}
	    }

	    for(int j=0; j<inflight; j++){
		int flag;
		MPI_Test(req+j, &flag, MPI_STATUS_IGNORE);
		if(!flag) continue;
		if(rank==0) ongoing_comm--;
		if(rank==1){
		    int received = buffers[j][0];
		    if(vals[j]+1 != received)
			printf("j %d vals %d recv %d\n", j, vals[j], received);
		    vals[j] = received;

		    /* post recv */
		    i++;
		    recv++;
		    MPI_Irecv(buffers[j], buff_size, MPI_INT, peer_rank, j, mpi_comm, req+j);
		}
	    }

	    if(i==niter) break;
	}

	printf("rank %d sent %d recv %d\n", rank, sent, recv);

	if(rank==0) MPI_Waitall(inflight, req, MPI_STATUS_IGNORE);
	if(rank==1) {
	    for(int j=0; j<inflight; j++)
		MPI_Cancel(req+j);
	}

	printf("rank %d finished\n", rank);

	/* cleanup */
	for(int j=0; j<inflight; j++){
	    MPI_Free_mem(buffers[j]);
	}
	delete req;
	delete buffers;
    }

    MPI_Finalize();
}
