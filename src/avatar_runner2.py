import asyncio
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)

sys.path.insert(0, str(Path(__file__).parent))
from wave_viz import WaveformVisualizer

logger = logging.getLogger("avatar-example-neural")

class MockNeuralModel(nn.Module):
    """
    Mock neural network model that takes audio chunks and generates video frames.
    In practice, this would be replaced with a real model like:
    - Audio-to-Avatar model (e.g., for lip-sync)
    - Speech-to-face generation model
    - Audio-driven animation model
    """
    def __init__(self, audio_features: int = 800, video_height: int = 720, video_width: int = 1280):
        super().__init__()
        self.audio_features = audio_features
        self.video_height = video_height
        self.video_width = video_width

        # Simple architecture for demonstration
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.frame_decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, video_height * video_width * 4),  # RGBA
            nn.Sigmoid(),
        )

    def forward(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """
        Generate video frame from audio chunk

        Args:
            audio_chunk: [batch_size, audio_features] or [audio_features]

        Returns:
            video_frame: [batch_size, height, width, 4] or [height, width, 4]
        """
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)

        # Encode audio features
        audio_encoded = self.audio_encoder(audio_chunk)

        # Generate video frame
        frame_flat = self.frame_decoder(audio_encoded)
        frame = frame_flat.view(-1, self.video_height, self.video_width, 4)

        # Convert from [0,1] to [0,255] and ensure RGBA format
        frame = (frame * 255).clamp(0, 255).byte()

        return frame.squeeze(0) if frame.size(0) == 1 else frame

class NeuralAudioWaveGenerator(VideoGenerator):
    def __init__(self, options: AvatarOptions, model_path: Optional[str] = None):
        self._options = options
        self._audio_queue = asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._audio_resampler: Optional[rtc.AudioResampler] = None

        # Initialize device and model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model(model_path)

        # Fallback visualizer for when neural model fails or for comparison
        self._fallback_canvas = np.zeros((options.video_height, options.video_width, 4), dtype=np.uint8)
        self._fallback_canvas.fill(255)
        self._wave_visualizer = WaveformVisualizer(sample_rate=options.audio_sample_rate)

        # Audio processing setup
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=options.audio_sample_rate // options.video_fps,
        )
        self._frame_ts: deque[float] = deque(maxlen=options.video_fps)

        # Thread pool for GPU operations to avoid blocking the main async loop
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-inference")

        # Performance monitoring
        self._inference_times = deque(maxlen=100)
        self._gpu_memory_usage = deque(maxlen=100)

        logger.info(f"Neural model initialized on device: {self._device}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create the neural model"""
        audio_chunk_size = self._options.audio_sample_rate // self._options.video_fps
        model = MockNeuralModel(
            audio_features=audio_chunk_size,
            video_height=self._options.video_height,
            video_width=self._options.video_width
        )

        if model_path and Path(model_path).exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self._device))
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")

        model = model.to(self._device)
        model.eval()  # Set to evaluation mode
        return model

    def _preprocess_audio(self, audio_frame: rtc.AudioFrame | None) -> torch.Tensor:
        """Convert audio frame to model input tensor"""
        if audio_frame is None:
            # Generate silence
            audio_chunk_size = self._options.audio_sample_rate // self._options.video_fps
            audio_data = np.zeros(audio_chunk_size, dtype=np.float32)
        else:
            # Convert int16 to float32 and normalize
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32767.0  # Normalize to [-1, 1]

            # Handle mono/stereo
            if audio_frame.num_channels > 1:
                audio_data = audio_data.reshape(-1, audio_frame.num_channels).mean(axis=1)

        return torch.from_numpy(audio_data).to(self._device)

    def _run_inference(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Run neural model inference on GPU (blocking operation)"""
        start_time = time.time()

        try:
            with torch.no_grad():
                # Run inference
                video_frame_tensor = self._model(audio_tensor)

                # Convert to numpy
                video_frame = video_frame_tensor.cpu().numpy().astype(np.uint8)

                # Record performance metrics
                inference_time = time.time() - start_time
                self._inference_times.append(inference_time)

                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1e6  # MB
                    self._gpu_memory_usage.append(gpu_memory)

                # Check if inference is too slow
                target_frame_time = 1.0 / self._options.video_fps
                if inference_time > target_frame_time * 0.8:  # Use 80% of frame time as threshold
                    logger.warning(f"Inference time ({inference_time*1000:.1f}ms) approaching frame deadline ({target_frame_time*1000:.1f}ms)")

                return video_frame

        except Exception as e:
            logger.error(f"Neural model inference failed: {e}")
            # Fall back to traditional visualization
            return self._generate_fallback_frame(audio_tensor)

    def _generate_fallback_frame(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Generate fallback frame when neural model fails"""
        canvas = self._fallback_canvas.copy()

        # Convert tensor back to numpy for traditional processing
        if audio_tensor.numel() == 0:
            audio_data = np.zeros((1, self._options.audio_channels))
        else:
            audio_data = audio_tensor.cpu().numpy()
            audio_data = (audio_data * 32767).astype(np.int16)
            audio_data = audio_data.reshape(-1, 1)  # Make it 2D for visualizer

        self._wave_visualizer.draw(canvas, audio_samples=audio_data, fps=self._get_fps())
        return canvas

    async def _generate_frame_async(self, audio_frame: rtc.AudioFrame | None) -> rtc.VideoFrame:
        """Generate video frame using neural model (async wrapper)"""
        # Preprocess audio on main thread (fast operation)
        audio_tensor = self._preprocess_audio(audio_frame)

        # Run inference on GPU thread to avoid blocking
        loop = asyncio.get_event_loop()
        video_frame_data = await loop.run_in_executor(
            self._gpu_executor, 
            self._run_inference, 
            audio_tensor
        )

        # Create VideoFrame
        video_frame = rtc.VideoFrame(
            width=video_frame_data.shape[1],
            height=video_frame_data.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=video_frame_data.tobytes(),
        )
        return video_frame

    # -- VideoGenerator abstract methods --

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Called by the runner to push audio frames to the generator."""
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        """Called by the runner to clear the audio buffer"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """
        Generate a continuous stream of video and audio frames.
        """
        return self._video_generation_impl()

    # -- End of VideoGenerator abstract methods --

    async def _video_generation_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        while True:
            try:
                # Timeout slightly shorter than frame interval to maintain frame rate
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self._options.video_fps
                )
                self._audio_queue.task_done()
            except asyncio.TimeoutError:
                # Generate frame without audio (silence state)
                video_frame = await self._generate_frame_async(None)
                yield video_frame
                self._frame_ts.append(time.time())
                continue

            audio_frames: list[rtc.AudioFrame] = []
            if isinstance(frame, rtc.AudioFrame):
                # Resample audio if needed
                if not self._audio_resampler and (
                    frame.sample_rate != self._options.audio_sample_rate
                    or frame.num_channels != self._options.audio_channels
                ):
                    self._audio_resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=self._options.audio_sample_rate,
                        num_channels=self._options.audio_channels,
                    )

                if self._audio_resampler:
                    for f in self._audio_resampler.push(frame):
                        audio_frames += self._audio_bstream.push(f.data)
                else:
                    audio_frames += self._audio_bstream.push(frame.data)
            else:
                if self._audio_resampler:
                    for f in self._audio_resampler.flush():
                        audio_frames += self._audio_bstream.push(f.data)

                audio_frames += self._audio_bstream.flush()

            # Generate video frames using neural model
            for audio_frame in audio_frames:
                video_frame = await self._generate_frame_async(audio_frame)
                yield video_frame
                yield audio_frame
                self._frame_ts.append(time.time())

            # Send the AudioSegmentEnd back to notify the playback finished
            if isinstance(frame, AudioSegmentEnd):
                yield AudioSegmentEnd()

    def _get_fps(self) -> float | None:
        if len(self._frame_ts) < 2:
            return None
        return (len(self._frame_ts) - 1) / (self._frame_ts[-1] - self._frame_ts[0])

    def get_performance_stats(self) -> dict:
        """Get performance statistics for monitoring"""
        stats = {
            "device": str(self._device),
            "avg_inference_time_ms": np.mean(self._inference_times) * 1000 if self._inference_times else 0,
            "max_inference_time_ms": np.max(self._inference_times) * 1000 if self._inference_times else 0,
            "target_frame_time_ms": 1000 / self._options.video_fps,
        }

        if self._gpu_memory_usage:
            stats.update({
                "gpu_memory_mb": self._gpu_memory_usage[-1] if self._gpu_memory_usage else 0,
                "avg_gpu_memory_mb": np.mean(self._gpu_memory_usage),
            })

        return stats

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, '_gpu_executor'):
            self._gpu_executor.shutdown(wait=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@utils.log_exceptions(logger=logger)
async def main(api_url: str, api_token: str, model_path: Optional[str] = None):
    # Connect to the room
    room = rtc.Room()
    await room.connect(api_url, api_token)
    should_stop = asyncio.Event()

    # Stop when disconnect from the room or the agent disconnects
    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            logging.info(f"Agent {participant.identity} disconnected, stopping worker")
            should_stop.set()

    @room.on("disconnected")
    def _on_disconnected():
        logging.info("Room disconnected, stopping worker")
        should_stop.set()

    # Define the avatar options and start the runner
    avatar_options = AvatarOptions(
        video_width=1280,
        video_height=720,
        video_fps=30,
        audio_sample_rate=24000,
        audio_channels=1,
    )

    video_gen = NeuralAudioWaveGenerator(avatar_options, model_path=model_path)
    runner = AvatarRunner(
        room, audio_recv=DataStreamAudioReceiver(room), video_gen=video_gen, options=avatar_options
    )

    # Performance monitoring task
    async def monitor_performance():
        while not should_stop.is_set():
            await asyncio.sleep(10)  # Monitor every 10 seconds
            stats = video_gen.get_performance_stats()
            logger.info(f"Performance stats: {stats}")

    try:
        await runner.start()

        # Run until stopped or the runner is complete/failed
        tasks = [
            asyncio.create_task(runner.wait_for_complete()),
            asyncio.create_task(should_stop.wait()),
            asyncio.create_task(monitor_performance()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await utils.aio.cancel_and_wait(*tasks)
        await runner.aclose()
        await room.disconnect()
        logger.info("Neural avatar runner stopped")

if __name__ == "__main__":
    import os

    from livekit.agents.cli.log import setup_logging

    setup_logging("DEBUG", devmode=True, console=True)

    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_token = os.getenv("LIVEKIT_TOKEN")
    model_path = os.getenv("MODEL_PATH")  # Optional path to pre-trained model

    assert livekit_url and livekit_token, "LIVEKIT_URL and LIVEKIT_TOKEN must be set"

    asyncio.run(main(livekit_url, livekit_token, model_path)) 