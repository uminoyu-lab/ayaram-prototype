"""Learning: Modern Hopfield continuous update + classical Hebb rule.

Design decision #2 (Aya + Yu, 2026-06-17):

  - Primary (option B): Modern Hopfield Network continuous update following
    Ramsauer 2020, "Hopfield Networks is All You Need" (arXiv:2008.02217).
    This is the rule that is mathematically equivalent to Transformer
    attention and that demos/attention_test.py must verify against
    Theorem 3.

  - Secondary (option C): classical Hebb rule, kept in parallel for
    comparison experiments. The original requirements doc named only the Hebb
    rule, but Aya + Yu corrected this on 2026-06-17 and made Modern Hopfield
    the main learning rule.

M1: implement both rules with a shared signature so demos can swap them.
"""
