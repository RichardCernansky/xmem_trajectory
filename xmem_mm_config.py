from typing import Dict

def xmem_mm_config(
    mem_every: int = 3,
    min_mid: int = 5, max_mid: int = 10,
    num_prototypes: int = 128,
    max_long_term: int = 10000,
    enable_long_term: bool = True,
    deep_update_every: int = 10**9,  # effectively disable deep update at t=0
    top_k: int = 30,
    single_object: bool = False,
    hidden_dim: int = 256,
) -> Dict:
    return {
        "mem_every": mem_every,
        "min_mid_term_frames": min_mid,
        "max_mid_term_frames": max_mid,
        "num_prototypes": num_prototypes,
        "max_long_term_elements": max_long_term,
        "enable_long_term": enable_long_term,
        "deep_update_every": deep_update_every,
        "top_k": top_k,
        "benchmark": False,
        "enable_long_term_count_usage": False,
        "single_object": single_object,
        "hidden_dim": hidden_dim,
    }