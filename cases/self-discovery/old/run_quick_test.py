#!/usr/bin/env python3
"""
Legacy quick-test entrypoint for the old self-discovery prototype.
"""

import sys


def main() -> None:
    message = """
[DEPRECATED] `run_quick_test.py` 是旧主线的快速实验脚本。

当前推荐的最小验证方式：
  python cases/self-discovery/canonical/nonpinn_fit.py --geometry spherical
  python cases/self-discovery/canonical/pinn_inverse.py --geometry spherical --epochs 1500

完整对比入口：
  python cases/self-discovery/canonical/run_compare.py --geometry spherical --pinn-epochs 3000
""".strip()
    raise SystemExit(message)

if __name__ == "__main__":
    main()
