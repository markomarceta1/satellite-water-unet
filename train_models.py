"""Programmatic training helper.

Provides a convenience function to invoke the existing `train.py` main
from other Python code or from tests.
"""
from typing import List, Optional
import sys


def train_from_argv(argv: Optional[List[str]] = None) -> int:
    """Run the `train.py` main() using the provided argv list.

    If `argv` is None, `sys.argv` is used. Returns the exit code from main
    or 0 on success.
    """
    if argv is not None:
        old_argv = sys.argv
        sys.argv = [old_argv[0]] + argv
        try:
            # import inside function to avoid side-effects at module import time
            from train import main

            main()
        finally:
            sys.argv = old_argv
    else:
        from train import main

        main()
    return 0


if __name__ == "__main__":
    # simple passthrough so this file can be called directly
    import sys

    sys.exit(train_from_argv(sys.argv[1:]))
