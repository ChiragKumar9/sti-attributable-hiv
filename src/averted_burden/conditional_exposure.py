def p_a_given_b(p_a, p_b, rr_a_b, allow_invalid=False):
    # recall that the RR = P(A|B) / P(A|~B)
    # We also know that P(A) = P(A|B)P(B) + P(A|~B)P(~B)
    # We can substitute for P(A|~B) in terms of RR and P(A|B):
    # P(A) = P(A|B)P(B) + (P(A|B) / RR) * (1 - P(B))
    # P(A) = P(A|B) * [P(B) + (1 - P(B)) / RR]
    # Rearranging to solve for P(A|B):
    # P(A|B) = P(A) / [P(B) + (1 - P(B)) / RR]
    p_a_given_b = p_a / (p_b + (1 - p_b) / rr_a_b)
    if allow_invalid:
        return p_a_given_b
    else:
        return min(p_a_given_b, 1.0)


def p_a_given_not_b(p_a, p_b, rr_a_b):
    """
    Calculate P(A|~B) given P(A), P(B), and RR = P(A|B)/P(A|~B)

    Parameters:
    -----------
    p_a : float
        Overall probability of A
    p_b : float
        Probability of B
    rr_a_b : float
        Relative risk = P(A|B) / P(A|~B)

    Returns:
    --------
    float
        P(A|~B) - probability of A given NOT B
    """
    # First get P(A|B), allowing > 1 so the cap in p_a_given_b applies cleanly
    p = p_a_given_b(p_a, p_b, rr_a_b, allow_invalid=True)
    # Then use RR definition: RR = P(A|B) / P(A|~B)
    # So: P(A|~B) = P(A|B) / RR
    return p / rr_a_b
