# recall that the RR = P(A|B) / P(A|~B)
# If we want to solve for P(A|B), we can rearrange to get:
# P(A|B) = RR * P(A|~B)
# We also know that P(A) = P(A|B)P(B) + P(A|~B)P(~B)
# We can substitute for P(A|~B) in terms of RR and P(A|B):
# P(A) = P(A|B)P(B) + (P(A|B) / RR) * (1 - P(B))
# P(A) = P(A|B) * [P(B) + (1 - P(B)) / RR]
# Rearranging to solve for P(A|B):
# P(A|B) = P(A) / [P(B) + (1 - P(B)) / RR]


def p_a_given_b(p_a, p_b, rr_a_b):
    return p_a / (p_b + (1 - p_b) / rr_a_b)
