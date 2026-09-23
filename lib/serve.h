// Residency policies for serving a MoE model whose weights exceed the budget.
#ifndef GTIER_SERVE_H
#define GTIER_SERVE_H
#include <stdint.h>

// The five are arranged so each isolates one decision.  none is the floor;
// lru is what a cache does with no oracle and no notion of phase; lru_phase
// differs from it in one line -- prefill does not admit -- so the gap between
// them is the cost of letting a prefill evict what decode needs; per_layer is
// the static oracle ordering; prefix adds the shared prefix's union pinned.
typedef enum {
    SERVE_NONE = 0,       // stream every expert, hold nothing
    SERVE_LRU,            // LRU over (layer,expert); prefill admits too
    SERVE_LRU_PHASE,      // same LRU, but prefill does not admit (sec 3.6)
    SERVE_PERLAYER,       // static, per-layer popularity order (sec 3.4)
    SERVE_PREFIX,         // PERLAYER + the shared prefix's union pinned (sec 3.5)
    SERVE_ONLINE,         // per-layer LFU learned online -- no oracle
    SERVE_ONLINE_PREFIX,  // ONLINE + prefix pinned, both learned online
    SERVE_MULTIPREFIX,    // several prefixes, LRU over prefixes when they do not fit
    SERVE_POLICY_COUNT
} serve_policy;

#endif
