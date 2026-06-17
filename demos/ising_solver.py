"""Demo: Ising-formulated NP problem solver.

Following Lucas 2014, "Ising formulations of many NP problems"
(Frontiers in Physics 2:5). Maps a small NP-problem instance onto the
3-layer Hopfield network, runs the 4-phase cycle (decision #1) with
sleep-mode fluctuation (decision #4) as the annealing mechanism, and reads
out the solution.

M1: pick one small Lucas-2014 instance (e.g. Max-Cut on ~8 nodes that fits
in the 64-cell output layer) and implement the encode -> cycle -> decode
pipeline.
"""


def main() -> None:
    """Entry point. M1 will populate this."""
    raise NotImplementedError("Ising solver demo is planned for M1.")


if __name__ == "__main__":
    main()
