"""Direct Windows entrypoint for launching FrFruit Studio Workstation (.exe or python)."""

import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from frfruit.desktop.win_shell import main  # noqa: E402

if __name__ == "__main__":
    main()
