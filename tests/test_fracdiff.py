import numpy as np
import pandas as pd

from settlex.features.fracdiff import frac_diff_ffd, frac_weights


def test_frac_weights_d0_is_unit():
    w = frac_weights(0.0, 5)
    assert w[-1] == 1.0
    assert np.allclose(w[:-1], 0.0)


def test_frac_diff_d0_is_identity():
    s = pd.Series(np.arange(50, dtype=float) + 10)
    fd = frac_diff_ffd(s, 0.0)
    assert np.allclose(fd.to_numpy(), s.to_numpy())


def test_frac_diff_d1_is_first_difference():
    s = pd.Series(np.cumsum(np.ones(50)) + 5.0)  # linear ramp
    fd = frac_diff_ffd(s, 1.0, thres=1e-6).dropna()
    assert np.allclose(fd.to_numpy(), 1.0, atol=1e-6)


def test_frac_diff_reduces_nonstationarity():
    rng = np.random.default_rng(0)
    random_walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
    fd = frac_diff_ffd(random_walk, 0.4).dropna()
    assert fd.std() < random_walk.std()
