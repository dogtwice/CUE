import torch

from src.lib.evidence import UDF


def make_alpha(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_multisource_pcr6_is_permutation_invariant():
    udf = UDF(num_views=3, n_clusters=2)

    alpha0 = make_alpha([[8.0, 2.0], [4.0, 3.0]])
    alpha1 = make_alpha([[2.5, 7.0], [3.5, 5.0]])
    alpha2 = make_alpha([[6.0, 3.0], [7.0, 2.5]])

    alpha_a_012, u_012 = udf.pcr6_combine({0: alpha0, 1: alpha1, 2: alpha2})
    alpha_a_210, u_210 = udf.pcr6_combine({0: alpha2, 1: alpha1, 2: alpha0})

    assert torch.allclose(alpha_a_012, alpha_a_210, atol=1e-6)
    assert torch.allclose(u_012, u_210, atol=1e-6)
