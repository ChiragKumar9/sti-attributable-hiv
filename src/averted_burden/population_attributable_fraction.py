# recall PAF = p_c (RR_adj - 1) / RR_adj


def paf(RR_adj, prevalence_cases):
    return prevalence_cases * (RR_adj - 1) / RR_adj
