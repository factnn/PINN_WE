#!/usr/bin/env python3
"""
Legacy entrypoint for the old self-discovery prototype.
"""

import sys


def main() -> None:
        message = """
[DEPRECATED] `run_discovery.py` 已不再是推荐入口。

请改用新的规范化主线：
    python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000

更多说明见：
    - cases/self-discovery/README.md
    - cases/self-discovery/canonical/README.md
    - cases/self-discovery/LEGACY_NOTES.md
""".strip()
        raise SystemExit(message)

if __name__ == "__main__":
    main()
