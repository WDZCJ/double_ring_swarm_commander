"""demo 单测（tests/test_demos.py）。

验证 3 个 demo 能完整跑通（不崩溃 + 产出 plan + 收集 telemetry）。
"""

from __future__ import annotations

import pytest

from munition_system.demos.demo_adhoc_events import main as adhoc_main
from munition_system.demos.demo_cooperative_attack import main as attack_main
from munition_system.demos.demo_cooperative_search import main as search_main


class TestDemos:
    """3 个 demo 都能完整跑通。"""

    def test_cooperative_search_runs(self, capsys):
        """协同搜索 demo 能跑通。"""
        search_main()
        captured = capsys.readouterr()
        assert "协同搜索 demo" in captured.out
        assert "final_phase" in captured.out

    def test_cooperative_attack_runs(self, capsys):
        """协同攻击 demo 能跑通。"""
        attack_main()
        captured = capsys.readouterr()
        assert "协同攻击 demo" in captured.out
        assert "final_phase" in captured.out

    def test_adhoc_events_runs(self, capsys):
        """临机事件 demo 能跑通。"""
        adhoc_main()
        captured = capsys.readouterr()
        assert "临机事件 demo" in captured.out
        assert "final_phase" in captured.out
