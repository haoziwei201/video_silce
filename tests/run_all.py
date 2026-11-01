"""
一键运行全部构造输入测试（不调用任何外部 API）。
用法：在 video_silce/ 目录下执行  python tests/run_all.py
"""
import os
import sys
import traceback
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import _common  # noqa: F401  （设置 sys.path / 日志）

MODULES = [
    "test_data_merger",
    "test_evidence_chain",
    "test_commitment_rules",
    "test_doc_ingest",
]


def main():
    failed = 0
    for mod_name in MODULES:
        print(f"\n[{mod_name}]")
        try:
            mod = importlib.import_module(mod_name)
            mod.run()
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"  ✗ {mod_name} 失败")
    print("\n" + ("全部测试通过 ✅" if failed == 0 else f"有 {failed} 个测试模块失败 ❌"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
