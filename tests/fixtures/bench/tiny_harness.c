/* tiny_harness.c - emits a hardcoded JSONL line; no timing. */
#include <stdio.h>

extern int noop(void);

int main(void) {
    int v = noop();
    fprintf(stderr, "tiny harness ran; noop returned %d\n", v);
    printf("{\"kernel\":\"noop\",\"median_ns\":42,\"p10_ns\":41,\"p90_ns\":43,\"n_inner\":1,\"n_runs\":1}\n");
    return 0;
}
