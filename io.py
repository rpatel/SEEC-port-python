# coding=utf-8
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as spio

def _get_raw_seec_variables_from_mat_file(file_path, extract_variables=None, **kwargs):
    return spio.loadmat(
        Path(file_path).resolve(strict=True),
        squeeze_me=True,
        variable_names=extract_variables,
    )


def _shift_sequence_variables(data_dict, _VARS_TO_SHIFT=("align", "Native"), **kwargs):
    for vts in _VARS_TO_SHIFT:
        if vts in data_dict:
            data_dict[vts] = data_dict[vts] - 1


def _get_shifted_seec_variables_from_mat_file(*args, **kwargs):
    mat_variables = _get_raw_seec_variables_from_mat_file(*args, **kwargs)
    _shift_sequence_variables(mat_variables, **kwargs)

    return mat_variables


def get_potts_pararmeters_from_mat_file(file_path):
    return _get_shifted_seec_variables_from_mat_file(
        file_path, extract_variables=("e", "h")
    )
