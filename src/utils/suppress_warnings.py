"""
Utility to suppress MediaPipe and TensorFlow warnings.
Import this module before importing MediaPipe to suppress verbose logging.
"""

import os
import sys
import warnings
import logging

# Suppress TensorFlow/MediaPipe C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress absl logging (used by MediaPipe)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

# Redirect stderr temporarily during MediaPipe import
class SuppressStderr:
    """Context manager to suppress stderr output."""
    def __enter__(self):
        self.null = open(os.devnull, 'w')
        self.old_stderr = sys.stderr
        sys.stderr = self.null
        return self
    
    def __exit__(self, *args):
        sys.stderr = self.old_stderr
        self.null.close()
