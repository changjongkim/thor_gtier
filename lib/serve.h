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
    // Prefix pinning buys prefill and costs decode, and per-layer popularity
    // does the reverse, because the two were made to bid for the same bytes
    // from separate pools.  They do not have to: a unit is worth what it
    // saves, and what it saves is the reads it prevents in *both* phases.
    //   value(u) = prefill_hits(u) + decode_hits(u)
    // Ranking by that and filling once removes the tradeoff rather than
    // trading it off.
    SERVE_UNIFIED,        // one value per unit, oracle counts (sec 3.7)
    SERVE_UNIFIED_ONLINE, // the same value learned from what is observed
    // Reimplementations of the published ideas, as policies over the same
    // budget.  They are the ideas, not the systems.
    SERVE_MOEINF,         // sequence-level activation matrix + LRU
    SERVE_MIXTRAL,        // LRU + speculative next-layer load
    SERVE_POLICY_COUNT
} serve_policy;

#endif
