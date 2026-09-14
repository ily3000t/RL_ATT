"""Explicit objectives for the frozen-policy gradient baselines."""


def margin(logits, clean_action):
    others = [i for i in range(3) if i != clean_action]
    return logits[others].max() - logits[clean_action]


