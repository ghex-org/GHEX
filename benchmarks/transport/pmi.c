#include <stdio.h>
#include <pmix.h>
#include <pthread.h>

#include <ghex/common/debug.hpp>

static pmix_proc_t allproc = {};
static pmix_proc_t myproc = {};
static uint32_t nprocs = 0;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    pmix_status_t   status;
    volatile bool   active;
} cblock_t;

#define LOCK_INIT(l)                            \
    do {                                        \
        pthread_mutex_init(&(l)->mutex, NULL);  \
        pthread_cond_init(&(l)->cond, NULL);    \
        (l)->active = false;                    \
        (l)->status = PMIX_SUCCESS;             \
    } while(0)

#define LOCK_DEL(l)                             \
    do {                                        \
        pthread_mutex_destroy(&(l)->mutex);     \
        pthread_cond_destroy(&(l)->cond);       \
    } while(0)

#define LOCK_WAIT(l)					\
    do {						\
        pthread_mutex_lock(&(l)->mutex);		\
        (l)->active = true;				\
        while ((l)->active) {				\
            pthread_cond_wait(&(l)->cond, &(l)->mutex);	\
        }						\
        pthread_mutex_unlock(&(l)->mutex);		\
    } while(0)

#define LOCK_SIGNAL(l)				\
    do {					\
        pthread_mutex_lock(&(l)->mutex);	\
        (l)->active = false;			\
        pthread_cond_broadcast(&(l)->cond);	\
        pthread_mutex_unlock(&(l)->mutex);	\
    } while(0)


int pmi_get_rank()
{
    return myproc.rank;
}

int pmi_get_size()
{
    return nprocs;
}

int pmi_init()
{
    int rc;
    pmix_value_t *pvalue;

    if (PMIX_SUCCESS != (rc = PMIx_Init(&myproc, NULL, 0))) {
	ERR("PMIx_Init failed");
        exit(1);
    }
    if(myproc.rank == 0) LOG("PMIx initialized");

    /* job-related info is found in our nspace, assigned to the
     * wildcard rank as it doesn't relate to a specific rank. Setup
     * a name to retrieve such values */
    PMIX_PROC_CONSTRUCT(&allproc);
    (void)strncpy(allproc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
    allproc.rank = PMIX_RANK_WILDCARD;

    /* get the number of procs in our job */
    if (PMIX_SUCCESS != (rc = PMIx_Get(&allproc, PMIX_JOB_SIZE, NULL, 0, &pvalue))) {
        ERR("Client ns %s rank %d: PMIx_Get job size failed: %d\n", myproc.nspace, myproc.rank, rc);
	exit(1);
    }
    nprocs = pvalue->data.uint32;
    PMIX_VALUE_RELEASE(pvalue);

    return 0;
}

int pmi_set_string(const char *key, void *data, size_t size)
{
    int rc;
    pmix_value_t value;

    /* printf("pmi_set_string: "); */
    /* for(int i=0; i<size; i++) printf("%x", ((unsigned char*)data)[i]); */
    /* printf("\n"); */

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
    LOG("PMIx_Put on %s", key);

    if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
        ERR("Client ns %s rank %d: PMIx_Commit failed: %d\n", myproc.nspace, myproc.rank, rc);
    }
    LOG("PMIx_Commit on %s", key);

    return 0;
}

int pmi_set_uint64(const char *key, uint64_t data)
{
    int rc;
    pmix_value_t value;

    PMIX_VALUE_CONSTRUCT(&value);
    value.type = PMIX_UINT64;
    value.data.uint64 = data;
    if (PMIX_SUCCESS != (rc = PMIx_Put(PMIX_GLOBAL, key, &value))) {
        ERR("Client ns %s rank %d: PMIx_Put failed: %d\n", myproc.nspace, myproc.rank, rc);
    }

    /* protect the data */
    PMIX_VALUE_DESTRUCT(&value);
    LOG("PMIx_Put on %s", key);

    if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
        ERR("Client ns %s rank %d: PMIx_Commit failed: %d\n", myproc.nspace, myproc.rank, rc);
    }
    LOG("PMIx_Commit on %s", key);

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

    /* printf("pmi_get_string: "); */
    /* for(int i=0; i<*data_size_out; i++) printf("%x", ((unsigned char*)*data_out)[i]); */
    /* printf("\n"); */

    /* protect the data */
    pvalue->data.bo.bytes = NULL;
    pvalue->data.bo.size = 0;
    PMIX_VALUE_RELEASE(pvalue);
    PMIX_PROC_DESTRUCT(&proc);

    LOG("PMIx_get %s returned %zi bytes", key, data_size_out[0]);

    return 0;
}

int pmi_get_uint64(uint32_t peer_rank, const char *key, uint64_t *data_out)
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
    if(pvalue->type != PMIX_UINT64){
        ERR("Client ns %s rank %d: PMIx_Get %s: got wrong data type\n", myproc.nspace, myproc.rank, key);
    }
    *data_out = pvalue->data.uint64;

    /* protect the data */
    PMIX_VALUE_RELEASE(pvalue);
    PMIX_PROC_DESTRUCT(&proc);

    return 0;
}

static void fence_status_cb(pmix_status_t status, void *cbdata)
{
    cblock_t *lock = (cblock_t*)cbdata;
    lock->status = status;
    lock->active = false;
    LOCK_SIGNAL(lock);
}

int pmi_exchange()
{
    int rc;
    pmix_info_t info;
    cblock_t lock;
    bool flag;

    LOCK_INIT(&lock);
    PMIX_INFO_CONSTRUCT(&info);
    flag = true;
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);
    if (PMIX_SUCCESS != (rc = PMIx_Fence_nb(&allproc, 1, &info, 1, fence_status_cb, &lock))){
        ERR("Client ns %s rank %d: PMIx_Fence_nb failed: %d\n", myproc.nspace, myproc.rank, rc);
    }
    PMIX_INFO_DESTRUCT(&info);

    /* wait for completion */
    LOCK_WAIT(&lock);
    if (PMIX_SUCCESS != lock.status){
        ERR("Client ns %s rank %d: PMIx_Fence_nb finished with an error: %d\n", myproc.nspace, myproc.rank, rc);
    }
    LOG("PMIx_Fence_nb");    

    return 0;
}

int pmi_finalize()
{
    int rc;
    if (PMIX_SUCCESS != (rc = PMIx_Finalize(NULL, 0))) {
        ERR("Client ns %s rank %d:PMIx_Finalize failed: %d\n", myproc.nspace, myproc.rank, rc);
    } else {
        if(myproc.rank == 0) LOG("PMIx finalized");
    }
    return 0;
}
