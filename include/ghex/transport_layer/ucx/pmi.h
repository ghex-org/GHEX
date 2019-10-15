#ifndef _PMI_H
#define _PMI_H

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
    int pmi_init();
    int pmi_get_rank();
    int pmi_get_size();
    int pmi_set_string(const char *key, void *data, size_t size);
    int pmi_get_string(uint32_t peer_rank, const char *key, void **data_out, size_t *data_size_out);
    int pmi_set_uint64(const char *key, uint64_t data);
    int pmi_get_uint64(uint32_t peer_rank, const char *key, uint64_t *data_out);
    int pmi_exchange();
    int pmi_finalize();
#ifdef __cplusplus
}
#endif

#endif /* _PMI_H */
