# -*- coding: utf-8 -*-
"""FastAPI 接口集成测试（需要 server 在 8765 运行）。"""
import socket

import pytest
import requests

BASE = "http://localhost:8765"


def _server_running() -> bool:
    """检查 8765 端口是否有服务监听。"""
    try:
        with socket.create_connection(("localhost", 8765), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_running(),
    reason="需要服务运行在 localhost:8765（CI 环境跳过）")


class TestState:
    def test_state_basic(self):
        r = requests.get(f"{BASE}/state")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 24
        assert "occupied" in data
        assert "free" in data
        assert "spaces" in data

    def test_docs_page(self):
        r = requests.get(f"{BASE}/docs")
        assert r.status_code == 200
        assert "swagger" in r.text.lower()

    def test_index_page(self):
        r = requests.get(BASE)
        assert r.status_code == 200
        assert "智能社区" in r.text

    def test_snapshot(self):
        r = requests.get(f"{BASE}/snapshot", timeout=10)
        if r.status_code == 500:
            pytest.skip("服务端无摄像头（Docker/Linux 环境）")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 10000


class TestEntryExit:
    @pytest.fixture(autouse=True)
    def reset(self):
        requests.post(f"{BASE}/reset")

    def test_entry_success(self, sample_images):
        with open(sample_images["front"], "rb") as f:
            r = requests.post(f"{BASE}/entry", files={"file": f})
        assert r.status_code == 200
        data = r.json()
        assert data["plate"] == "沪AD07979"
        assert data["space"].startswith("A")

    def test_entry_duplicate(self, sample_images):
        with open(sample_images["front"], "rb") as f:
            requests.post(f"{BASE}/entry", files={"file": f})
        with open(sample_images["front"], "rb") as f:
            r = requests.post(f"{BASE}/entry", files={"file": f})
        assert r.status_code == 409
        assert "重复" in r.json()["detail"]

    def test_entry_no_plate(self, sample_images):
        with open(sample_images["blank"], "rb") as f:
            r = requests.post(f"{BASE}/entry", files={"file": f})
        assert r.status_code == 422
        assert "未检测到" in r.json()["detail"]

    def test_exit_success(self, sample_images):
        with open(sample_images["front"], "rb") as f:
            requests.post(f"{BASE}/entry", files={"file": f})
        with open(sample_images["front"], "rb") as f:
            r = requests.post(f"{BASE}/exit", files={"file": f})
        assert r.status_code == 200
        assert r.json()["plate"] == "沪AD07979"

    def test_exit_no_record(self, sample_images):
        with open(sample_images["front"], "rb") as f:
            r = requests.post(f"{BASE}/exit", files={"file": f})
        assert r.status_code == 404
        assert "无入场记录" in r.json()["detail"]

    def test_full_lot(self, sample_images):
        """预填 24 条记录后再进场应 409 车位已满。"""
        import openpyxl
        from openpyxl.styles import Font
        from config import settings

        # 直接写 24 条假记录到 Excel（服务每次请求都会重新读盘）
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "In Lot"
        ws.append(["License Plate", "Entry Time", "Space"])
        for r in "ABCD":
            for c in range(1, 7):
                ws.append([f"TEST{r}{c}", "2026-10-01 00:00:00", f"{r}{c}"])
        wb.save(settings.EXCEL_PATH)

        with open(sample_images["front"], "rb") as f:
            r = requests.post(f"{BASE}/entry", files={"file": f})
        assert r.status_code == 409
        assert "车位已满" in r.json()["detail"]
        requests.post(f"{BASE}/reset")  # 清理假记录


class TestReset:
    def test_reset(self, sample_images):
        with open(sample_images["front"], "rb") as f:
            requests.post(f"{BASE}/entry", files={"file": f})
        r = requests.post(f"{BASE}/reset")
        assert r.status_code == 200
        state = requests.get(f"{BASE}/state").json()
        assert state["occupied"] == 0


class TestSSE:
    def test_sse_initial_state(self):
        r = requests.get(f"{BASE}/stream", stream=True, timeout=5)
        assert r.status_code == 200
        lines = []
        for line in r.iter_lines(decode_unicode=True):
            if line:
                lines.append(line)
            if len(lines) >= 2:
                break
        # 第一条 event: state，第二条 data: {...}
        assert any("event: state" in l for l in lines)
        data_line = [l for l in lines if l.startswith("data:")][0]
        assert '"total": 24' in data_line
