#!/usr/bin/env python3
"""
Legacy two-stage training entrypoint.
"""

import sys


def main() -> None:
    message = """
[DEPRECATED] `two_stage_training.py` 属于旧版训练路线。

这个文件保留用于历史追溯，不再作为默认实验入口。

请改用：
  python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000

相关说明：
  - cases/self-discovery/README.md
  - cases/self-discovery/LEGACY_NOTES.md
  - cases/self-discovery/legacy/README.md
""".strip()
    raise SystemExit(message)


if __name__ == "__main__":
    main()
