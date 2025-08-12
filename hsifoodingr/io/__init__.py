"""I/O subpackage.

Avoid importing submodules at package import time to prevent side effects in tests
(e.g., mocking external dependencies like 'spectral'). Import submodules directly:

    from hsifoodingr.io.envi_reader import read_envi_hsi
    from hsifoodingr.io.rgb_reader import read_rgb
    from hsifoodingr.io.json_reader import read_annotation
"""

__all__ = []
