#!/usr/bin/env python3
"""
Legacy entrypoint for the old improved training prototype.
"""

import sys


def main() -> None:
        message = """
[DEPRECATED] `run_improved.py` 属于历史试验入口，默认不再使用。

如果你要验证当前采用的指数定义，请改用：
    python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000

历史说明见：
    - cases/self-discovery/LEGACY_NOTES.md
    - cases/self-discovery/legacy/README.md
""".strip()
        raise SystemExit(message)

if __name__ == "__main__":
        main()
