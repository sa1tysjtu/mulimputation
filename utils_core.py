import numpy as np
import torch


def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "prelu":
        return torch.nn.PReLU()
    if activation == "tanh":
        return torch.nn.Tanh()
    if activation in (None, "none"):
        return torch.nn.Identity()
    raise NotImplementedError


def pick_coeffs(X, idxs_obs, idxs_nas):
    d = len(idxs_obs)
    coeffs = torch.randn(d, len(idxs_nas))
    Wx = X[:, idxs_obs].mm(coeffs)
    sd = Wx.std(dim=0, keepdim=True)
    coeffs /= sd
    return coeffs


def fit_intercepts(X, coeffs, p):
    n, d = X.shape
    intercepts = torch.zeros(d)
    for j in range(d):
        def f(x):
            return torch.sigmoid(X @ coeffs[:, j] + x).mean() - p

        intercepts[j] = torch.tensor(
            float(torch.special.ndtr((p - 0.5) * 2))
        )
        for _ in range(5):
            intercepts[j] = intercepts[j] - f(intercepts[j]) / (0.25 + 1e-6)
    return intercepts


def MAR_mask(X, p, p_obs):
    n, d = X.shape
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps
    return mask


def MNAR_self_mask_logistic(X, p):
    n, d = X.shape
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
    coeffs = torch.randn(d)
    Wx = X * coeffs
    intercepts = torch.zeros(d)
    for j in range(d):
        intercepts[j] = torch.tensor(float(torch.special.ndtr((p - 0.5) * 2)))
    ps = torch.sigmoid(Wx + intercepts)
    ber = torch.rand(n, d)
    mask = ber < ps
    return mask


def MNAR_mask_logistic(X, p, p_params=0.3, exclude_inputs=True):
    n, d = X.shape
    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    d_na = d - d_params if exclude_inputs else d
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)
    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)
    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps
    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p
    return mask


def produce_NA(X, p_miss, mecha="MCAR", n_row=None, n_col=None, opt=None, p_obs=None, q=None):
    if mecha == "Random":
        unif_random_matrix = np.random.uniform(0.0, 1.0, size=X.shape[0])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
    elif mecha == "MCAR":
        unif_random_matrix = np.random.uniform(0.0, 1.0, size=[n_row, n_col])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
    elif mecha == "MAR":
        mask = MAR_mask(X.view(n_row, n_col), p_miss, p_obs).double()
        mask = mask == 0
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X.view(n_row, n_col), p_miss).double()
    elif mecha == "MNAR" and opt == "logistic":
        if torch.is_tensor(X):
            mask = MNAR_mask_logistic(X.double(), p_miss, p_obs).double()
        else:
            mask = MNAR_mask_logistic(torch.from_numpy(X).double(), p_miss, p_obs).double()
        mask = mask == 0
    else:
        raise ValueError("Missing mechanism not implemented")
    return mask.view(-1)
