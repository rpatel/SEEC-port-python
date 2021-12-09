# coding=utf-8
import numpy as np

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
ALPHABET2INT_MAP = dict((aa, ix) for ix, aa in enumerate(ALPHABET))


def transcode_int2alphabet_seq(int_seq):
    assert not np.any(
        (int_seq < 0) | (int_seq >= len(ALPHABET))
    ), f"Invalid sequence. Encoding requires values >=0 and <{len(ALPHABET)}."
    return [ALPHABET[item] for item in int_seq]


def transcode_alphabet2int_seq(alphabet_seq):
    return [ALPHABET2INT_MAP[item] for item in alphabet_seq]

