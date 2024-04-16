from typing import Union, List

import numpy as np
import torch


def continuous_to_bin(values, num_bins, min_value=0.0, max_value=1.0, border_type='periodic', **kwargs):
    assert border_type in ['periodic', 'clamp']
    z = (values - min_value) / (max_value - min_value)
    if border_type == 'periodic':
        z = z % 1.0
    elif border_type == 'clamp':
        if isinstance(z, np.ndarray):
            z = np.clip(z, min_value, max_value)
            z[z == max_value] = min_value
        elif isinstance(z, torch.Tensor):
            z = torch.clamp(z, min_value, max_value)
            z[z == max_value] = min_value
        else:
            z = max_value if z > max_value else z
            z = min_value if (z == max_value or z < min_value) else z
    if isinstance(values, np.ndarray):
        return np.floor(z * num_bins).astype(np.int32) % num_bins
    elif isinstance(values, torch.Tensor):
        return torch.floor(z * num_bins).to(torch.long) % num_bins
    else:
        return int(np.floor(z * num_bins)) % num_bins


def bin_to_continuous(bins, num_bins, min_value=0.0, max_value=1.0, **kwargs):
    bins = bins % num_bins
    z = (bins + 0.5) / num_bins
    return z * (max_value - min_value) + min_value


class MultiBin:
    """Converts between continuous values and bin indices."""

    def __init__(self, num_bins=40, min_value=0.0, max_value=1.0, border_type='periodic'):
        assert border_type in ['periodic', 'clamp']

        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.border_type = border_type

    def value_to_bin(self, values: Union[List[float], List[int], np.ndarray, float, int]):
        if isinstance(values, list):
            return [self._value_to_bin(v) for v in values]
        else:
            return self._value_to_bin(values)

    def _value_to_bin(self, values):
        z = (values - self.min_value) / (self.max_value - self.min_value)
        if self.border_type == 'periodic':
            z = z % 1.0
        elif self.border_type == 'clamp':
            if isinstance(np.ndarray):
                z = np.clip(z, self.min_value, self.max_value)
                z[z == self.max_value] = self.min_value
            else:
                z = self.max_value if z > self.max_value else z
                z = self.min_value if (z == self.max_value or z < self.min_value) else z
        if isinstance(values, np.ndarray):
            return np.floor(z * self.num_bins).astype(np.int32)
        else:
            return int(np.floor(z * self.num_bins))

    def bin_to_value(self, bins: Union[List[int], np.ndarray, int]):
        if isinstance(bins, list):
            return [self._bin_to_value(b) for b in bins]
        else:
            return self._bin_to_value(bins)

    def _bin_to_value(self, bins):
        z = (bins + 0.5) / self.num_bins
        return z * (self.max_value - self.min_value) + self.min_value
