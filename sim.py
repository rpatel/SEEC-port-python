# coding=utf-8
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DEFAULT_RNG_BG = np.random.PCG64


def get_rng(seed=None):
    seed_seq = np.random.SeedSequence(seed)
    bg = DEFAULT_RNG_BG(seed_seq)
    rng = np.random.Generator(bg)

    return (rng, seed_seq.entropy)


def get_site_alphabet_pdf(seq_arr, e, h, site_to_mutate):
    alphabet_size, num_sites = h.shape
    probs = np.zeros((alphabet_size,))

    for site_num in range(site_to_mutate):
        probs += e[
            site_num * alphabet_size + seq_arr[site_num],
            site_to_mutate * alphabet_size : site_to_mutate * alphabet_size
            + alphabet_size,
        ]

    for site_num in range(site_to_mutate + 1, num_sites):
        probs += e[
            site_to_mutate * alphabet_size : site_to_mutate * alphabet_size
            + alphabet_size,
            site_num * alphabet_size + seq_arr[site_num],
        ]

    probs += h[:, site_to_mutate]

    pdf = (exp_probs := np.exp(probs)) / np.sum(exp_probs)

    return pdf


@dataclass
class SequenceEvolutionSimulation:
    e: np.ndarray
    h: np.ndarray

    alphabet_size: int = field(init=False)
    num_sites: int = field(init=False)

    def __post_init__(self):
        self.alphabet_size, self.num_sites = self.h.shape
        self.alphabet_array = np.arange(self.alphabet_size, dtype=np.uint8)

    def run_simulation(
        self,
        start_seq_arr: np.ndarray,
        num_generations: int = 30000,
        random_seed: int = None,
    ):
        (num_sites,) = start_seq_arr.shape
        assert num_sites == self.num_sites

        rng, random_seed = get_rng(random_seed)

        trajectory = np.zeros((num_generations, num_sites), dtype=np.uint8)
        trajectory[0, :] = start_seq_arr

        mutation_types = np.zeros((num_generations, num_sites), dtype=np.uint8)

        for i in range(1, num_generations):
            site_to_mutate = rng.integers(num_sites)
            site_pdf = get_site_alphabet_pdf(
                trajectory[i - 1, :], self.e, self.h, site_to_mutate
            )
            mutate_to = rng.choice(self.alphabet_array, p=site_pdf)

            trajectory[i, :] = trajectory[i - 1, :]
            trajectory[i, site_to_mutate] = mutate_to

            mutation_types[i, site_to_mutate] = (
                2
                if (trajectory[i, site_to_mutate] != trajectory[i - 1, site_to_mutate])
                else 1
            )

        event_times = stats.poisson.rvs(
            10, size=num_generations, random_state=rng
        ).astype(np.uint8)

        return trajectory, event_times, mutation_types, random_seed
