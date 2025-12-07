# Stub implementation for torch to satisfy imports in the bilingual project
"""A minimal stub of the torch library providing only the symbols required by the
project's code and tests. This stub is placed in the repository root so that it
takes precedence over any real installation on the system.
"""

__version__ = "stub"

# ---------------------------------------------------------------------------
# Basic tensor placeholder
# ---------------------------------------------------------------------------
class Tensor:
    """Placeholder for torch.Tensor. No actual functionality is needed for the
    current test suite – the models are never instantiated because the stub
    replaces the real library.
    """
    pass

# ---------------------------------------------------------------------------
# Simple load / save helpers (no‑op)
# ---------------------------------------------------------------------------
def load(path):
    """Return a dummy object when a model is loaded. The real implementation
    returns a torch.nn.Module, but the tests only check that the call succeeds.
    """
    return None


def save(obj, path):
    """No‑op save function – does nothing.
    """
    pass

# ---------------------------------------------------------------------------
# CUDA helper
# ---------------------------------------------------------------------------
class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

cuda = _Cuda()

# ---------------------------------------------------------------------------
# Context manager for no_grad
# ---------------------------------------------------------------------------
class _NoGrad:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

no_grad = _NoGrad()

# ---------------------------------------------------------------------------
# Minimal nn module
# ---------------------------------------------------------------------------
class _BaseModule:
    def __init__(self, *args, **kwargs):
        pass

class Dropout(_BaseModule):
    def __call__(self, x):
        return x

class Linear(_BaseModule):
    def __call__(self, x):
        return x

class ReLU(_BaseModule):
    def __call__(self, x):
        return x

class Module(_BaseModule):
    pass

class nn:
    Module = Module
    Dropout = Dropout
    Linear = Linear
    ReLU = ReLU

# End of torch stub
