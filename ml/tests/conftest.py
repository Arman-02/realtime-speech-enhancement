from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_num_threads(4)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def crn():
    from ans.models import build_model

    torch.manual_seed(0)
    return build_model("ans_crn").eval()
