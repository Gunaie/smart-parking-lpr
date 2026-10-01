# -*- coding: utf-8 -*-
"""TTS 语音播报模块。

主路径：edge-tts（在线，微软神经语音）生成 mp3 → playsound3 播放。
兜底：若 edge-tts 不可用，用 pywin32 调用系统 SAPI 直接朗读（离线）。

对外函数：welcome(plate, space) / goodbye(plate) / speak(text)
"""
import asyncio
import hashlib
from pathlib import Path

from config import settings
from log_utils import get_logger

logger = get_logger("tts")


def _cache_path(text: str) -> Path:
    h = hashlib.md5(f"{settings.TTS_VOICE}:{text}".encode("utf-8")).hexdigest()
    return settings.TTS_DIR / f"{h}.mp3"


def _generate_edge(text: str) -> Path | None:
    """用 edge-tts 生成 mp3，成功返回文件路径，失败返回 None。"""
    out = _cache_path(text)
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        import edge_tts

        async def _run():
            c = edge_tts.Communicate(text, settings.TTS_VOICE)
            await c.save(str(out))

        asyncio.run(_run())
        return out if out.exists() else None
    except Exception as e:
        logger.warning(f"[TTS] edge-tts 生成失败，转离线兜底: {e}")
        return None


def _play(path: Path) -> bool:
    try:
        from playsound3 import playsound

        playsound(str(path))
        return True
    except Exception as e:
        logger.warning(f"[TTS] 播放失败: {e}")
        return False


def _speak_sapi(text: str) -> bool:
    """离线兜底：调用 Windows SAPI 直接朗读。"""
    try:
        import win32com.client

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Speak(text)
        return True
    except Exception as e:
        logger.error(f"[TTS] SAPI 兜底也失败: {e}")
        return False


def speak(text: str) -> bool:
    """朗读一段文字，优先 edge-tts，失败用 SAPI。"""
    if not settings.TTS_ENABLED:
        return False
    logger.info(f"[TTS] {text}")
    mp3 = _generate_edge(text)
    if mp3 is not None:
        return _play(mp3)
    return _speak_sapi(text)


def welcome(plate: str, space: str = "") -> bool:
    """进场欢迎语，例如：沪AD07979，欢迎光临，A1号车位。"""
    text = f"{plate}，欢迎光临"
    if space:
        text += f"，{space}号车位"
    return speak(text)


def goodbye(plate: str) -> bool:
    """离场送别语，例如：沪AD07979，一路顺风。"""
    return speak(f"{plate}，一路顺风")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS 测试")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--welcome", metavar="PLATE", help="欢迎语")
    group.add_argument("--goodbye", metavar="PLATE", help="送别语")
    group.add_argument("--say", metavar="TEXT", help="自定义文本")
    args = parser.parse_args()

    settings.ensure_runtime_dirs()
    if args.welcome:
        welcome(args.welcome, "A1")
    elif args.goodbye:
        goodbye(args.goodbye)
    else:
        speak(args.say or "语音测试正常")
