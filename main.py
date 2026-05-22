"""
FastAPI 入口 - 独立语音问答服务
提供 /ask 接口，支持文本或语音输入，返回文本或语音输出
"""

import os
import uuid
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.config_loader import load_config
from config.logger import setup_logging
from core.connection import Asker
from core.utils.gc_manager import get_gc_manager

TAG = "main"
logger = setup_logging()

# 加载配置
config = load_config()

# 初始化 Asker（全局单例，避免重复加载模型）
asker: Optional[Asker] = None

# FastAPI 应用
app = FastAPI(
    title="Voice Q&A Service",
    description="独立的语音问答服务，支持 ASR → LLM → TTS",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskTextRequest(BaseModel):
    text: str


class AskTextResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global asker
    logger.bind(tag=TAG).info("正在初始化 Asker...")
    try:
        asker = Asker(config)
        logger.bind(tag=TAG).info("Asker 初始化成功")
    except Exception as e:
        logger.bind(tag=TAG).error(f"Asker 初始化失败: {e}")
        raise

    gc_manager = get_gc_manager(interval_seconds=300)
    await gc_manager.start()
    logger.bind(tag=TAG).info("GC 管理器已启动")


@app.on_event("shutdown")
async def shutdown_event():
    global asker
    if asker:
        await asker.close()
        logger.bind(tag=TAG).info("Asker 已关闭")
    gc_manager = get_gc_manager()
    await gc_manager.stop()
    logger.bind(tag=TAG).info("GC 管理器已停止")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ask/text", response_model=AskTextResponse)
async def ask_text(request: AskTextRequest):
    if not asker:
        raise HTTPException(status_code=503, detail="服务未就绪")
    try:
        answer = await asker.ask_text(request.text)
        return AskTextResponse(text=answer, success=True)
    except Exception as e:
        logger.bind(tag=TAG).error(f"文本问答失败: {e}")
        return AskTextResponse(text="", success=False, error=str(e))


@app.post("/ask/audio")
async def ask_audio(
    audio: UploadFile = File(...),
    response_format: str = Form("wav"),
):
    """
    语音问答接口
    上传音频文件（支持常见格式），返回合成的音频文件（默认 wav）
    """
    if not asker:
        raise HTTPException(status_code=503, detail="服务未就绪")

    MAX_SIZE = 10 * 1024 * 1024
    contents = await audio.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="文件过大，最大 10MB")

    # 保存临时文件
    suffix = os.path.splitext(audio.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        import asyncio
        from pydub import AudioSegment

        def convert_to_pcm(file_path: str) -> bytes:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            return audio.raw_data

        pcm_bytes = await asyncio.to_thread(convert_to_pcm, tmp_path)
        audio_bytes = await asker.ask_audio(pcm_bytes, sample_rate=16000)

        # 转换输出格式
        output_bytes = audio_bytes
        media_type = "audio/wav"

        if response_format == "mp3":
            seg = AudioSegment(
                data=audio_bytes,
                sample_width=2,
                frame_rate=16000,
                channels=1
            )
            output_bytes = seg.export(format="mp3").read()
            media_type = "audio/mpeg"
        elif response_format == "opus":
            # 简单返回 PCM（Opus 编码需要额外实现）
            media_type = "audio/ogg"
        else:  # wav
            import io
            import wave
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            output_bytes = wav_io.getvalue()
            media_type = "audio/wav"

        return Response(content=output_bytes, media_type=media_type)

    except Exception as e:
        logger.bind(tag=TAG).error(f"语音问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/ask/audio_raw")
async def ask_audio_raw(request: Request):
    """
    直接接收 PCM 原始数据的接口（用于自定义客户端）
    请求体必须是纯 PCM 字节流（16kHz 单声道 16bit）
    """
    if not asker:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        pcm_bytes = await request.body()
        if not pcm_bytes:
            raise HTTPException(status_code=400, detail="空的 PCM 数据")

        audio_bytes = await asker.ask_audio(pcm_bytes, sample_rate=16000)

        # 转换为 WAV 返回
        import io
        import wave
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        output_bytes = wav_io.getvalue()
        return Response(content=output_bytes, media_type="audio/wav")
    except Exception as e:
        logger.bind(tag=TAG).error(f"语音问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.get("server", {}).get("ip", "0.0.0.0"),
        port=config.get("server", {}).get("port", 8000),
        reload=False,
        log_level="info"
    )