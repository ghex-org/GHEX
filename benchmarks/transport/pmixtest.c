#include <stdio.h>
#include <pmix.h>
#include <mpi.h>

static pmix_proc_t allproc = {};
static pmix_proc_t myproc = {};

#define ERR(msg, ...)							\
    do {								\
	time_t tm = time(NULL);						\
	char *stm = ctime(&tm);						\
	stm[strlen(stm)-1] = 0;						\
	fprintf(stderr, "%s ERROR: %s:%d  " msg "\n", stm, __FILE__, __LINE__, ## __VA_ARGS__); \
	exit(1);							\
    } while(0);


int pmi_set_string(const char *key, void *data, size_t size)
{
    int rc;
    pmix_value_t value;

    PMIX_VALUE_CONSTRUCT(&value);
    value.type = PMIX_BYTE_OBJECT;
    value.data.bo.bytes = data;
    value.data.bo.size  = size;
    if (PMIX_SUCCESS != (rc = PMIx_Put(PMIX_GLOBAL, key, &value))) {
        ERR("Client ns %s rank %d: PMIx_Put failed: %d\n", myproc.nspace, myproc.rank, rc);
    }

    /* protect the data */
    value.data.bo.bytes = NULL;
    value.data.bo.size  = 0;
    PMIX_VALUE_DESTRUCT(&value);
    printf("PMIx_Put on %s\n", key);

    if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
        ERR("Client ns %s rank %d: PMIx_Commit failed: %d\n", myproc.nspace, myproc.rank, rc);
    }
    printf("PMIx_Commit on %s\n", key);

    return 0;
}

int pmi_get_string(uint32_t peer_rank, const char *key, void **data_out, size_t *data_size_out)
{
    int rc;
    pmix_proc_t proc;
    pmix_value_t *pvalue;

    PMIX_PROC_CONSTRUCT(&proc);
    (void)strncpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
    proc.rank = peer_rank;
    if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, key, NULL, 0, &pvalue))) {
        ERR("Client ns %s rank %d: PMIx_Get %s: %d\n", myproc.nspace, myproc.rank, key, rc);
    }
    if(pvalue->type != PMIX_BYTE_OBJECT){
        ERR("Client ns %s rank %d: PMIx_Get %s: got wrong data type\n", myproc.nspace, myproc.rank, key);
    }
    *data_out = pvalue->data.bo.bytes;
    *data_size_out = pvalue->data.bo.size;

    /* protect the data */
    pvalue->data.bo.bytes = NULL;
    pvalue->data.bo.size = 0;
    PMIX_VALUE_RELEASE(pvalue);
    PMIX_PROC_DESTRUCT(&proc);

    printf("PMIx_get %s returned %zi bytes\n", key, data_size_out[0]);

    return 0;
}

int main(int argc, char *argv[])
{
    char data[256];
    char *data_out;
    size_t size_out;
    int rc;
    pmix_value_t *pvalue;

    // MPI_Init(&argc, &argv);

    if (PMIX_SUCCESS != (rc = PMIx_Init(&myproc, NULL, 0))) {
	ERR("PMIx_Init failed");
        exit(1);
    }
    if(myproc.rank == 0) printf("PMIx initialized\n");

    sprintf(data, "this is data for rank %d\n", myproc.rank);
    pmi_set_string("ghex-rank-address", data, 256);
    pmi_get_string((myproc.rank+1)%2, "ghex-rank-address", (void**)&data_out, &size_out);

    if (PMIX_SUCCESS != (rc = PMIx_Finalize(NULL, 0))) {
        ERR("Client ns %s rank %d:PMIx_Finalize failed: %d\n", myproc.nspace, myproc.rank, rc);
    }
    if(myproc.rank == 0) printf("PMIx finalized\n");

    // MPI_Finalize();
}
