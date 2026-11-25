"""Fast numeric annotation readers; None asks the caller to use its old parser."""
from collections import OrderedDict
import re
import numpy as np
import torch

_NUMBER = rb'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_TABLE_PATTERNS = {}
_UINT_PATTERN = re.compile(rb'\s*\d+(?:\s*[,\r\n]\s*\d+)*\s*')


def read_float_table(path, ncols=4):
    try:
        with open(path, 'rb') as handle:
            raw = handle.read().strip()
        if ncols not in _TABLE_PATTERNS:
            row = rb'[ \t]*' + _NUMBER + (rb'[ \t]*,[ \t]*' + _NUMBER) * (ncols - 1) + rb'[ \t]*'
            _TABLE_PATTERNS[ncols] = re.compile(row + rb'(?:\r?\n' + row + rb')*')
        if not _TABLE_PATTERNS[ncols].fullmatch(raw):
            return None
        values = np.fromstring(raw.replace(b'\r', b'').replace(b'\n', b',').decode('ascii'), dtype=np.float64, sep=',')
        return values.reshape(-1, ncols).astype(np.float32)
    except (OSError, ValueError, UnicodeError):
        return None


def read_uint_vec(path):
    try:
        with open(path, 'rb') as handle:
            raw = handle.read().strip()
        if not _UINT_PATTERN.fullmatch(raw):
            return None
        flattened = raw.replace(b'\r', b'').replace(b'\n', b',')
        if any(not field.strip() for field in flattened.split(b',')):
            return None
        values = np.fromstring(flattened.decode('ascii'), dtype=np.int64, sep=',')
        if not values.size or values.min() < 0 or values.max() > 255:
            return None
        return values.astype(np.uint8)
    except (OSError, ValueError, UnicodeError):
        return None


class SequenceInfoCache:
    """Worker-local, byte-bounded cache. Dataset frame access must clone annotations."""
    def __init__(self, max_bytes=64 * 2**20):
        self.max_bytes = max_bytes
        self._data = OrderedDict()
        self._nbytes = 0

    @staticmethod
    def _size(info):
        return 64 + sum(v.numel() * v.element_size() for v in info.values() if torch.is_tensor(v))

    def get(self, key):
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key, value):
        size = self._size(value)
        if size > self.max_bytes:
            return
        old = self._data.pop(key, None)
        if old is not None:
            self._nbytes -= self._size(old)
        self._data[key] = value
        self._nbytes += size
        while self._nbytes > self.max_bytes:
            _, removed = self._data.popitem(last=False)
            self._nbytes -= self._size(removed)
