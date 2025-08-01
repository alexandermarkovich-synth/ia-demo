import asyncio
import contextlib
import logging

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc

# Configuration constants
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
AMPLITUDE = 2 ** 8 - 1
AUDIO_CHUNK_SIZE = 480  # 10ms at 48kHz
VIDEO_FPS = 16
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

load_dotenv(".env.local")
logger = logging.getLogger(__name__)


async def audio_processor(track: rtc.Track, queue: asyncio.Queue):
    audio_stream = rtc.AudioStream(track)
    async for audio_stream_event in audio_stream:
        await queue.put(audio_stream_event)
    await audio_stream.aclose()

async def video_processor(track: rtc.Track, queue: asyncio.Queue):
    """Process incoming video frames and put them in queue for syncing"""
    video_stream = rtc.VideoStream(track)
    async for video_frame_event in video_stream:
        await queue.put(video_frame_event)
    await video_stream.aclose()

async def audio_publisher(audio_source: rtc.AudioSource, audio_queue: asyncio.Queue):
    while True:

        try:

            audio_frame_event = None

            try:
                audio_frame_event = audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

            if audio_frame_event is not None:
                raw_bytes = audio_frame_event.frame.data
                array = np.frombuffer(raw_bytes, dtype=np.int16)

                # Apply simple DSP filter - low-pass filter to reduce high frequency noise
                # Simple moving average filter (low-pass)
                window_size = 3
                if len(array) >= window_size:
                    logger.info("Applying low-pass filter")
                    # Pad the array for edge handling
                    padded = np.pad(array, (window_size//2, window_size//2), mode='edge')
                    # Apply moving average
                    filtered = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
                    array = filtered.astype(np.int16)

                new_audio_frame = rtc.AudioFrame(
                    array.tobytes(),
                    audio_frame_event.frame.sample_rate,
                    audio_frame_event.frame.num_channels,
                    audio_frame_event.frame.samples_per_channel
                )

                await audio_source.capture_frame(new_audio_frame)


        except Exception as e:
            logger.error(f"Error in stream synchronizer: {e}")
            await asyncio.sleep(0.01)

async def video_publisher(video_source: rtc.VideoSource, video_queue: asyncio.Queue):
    while True:

        try:

            video_frame_event: rtc.VideoFrameEvent | None = None

            try:
                video_frame_event = video_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

            if video_frame_event is not None:
                await video_source.capture_frame(video_frame_event.frame)

        except Exception as e:
            logger.error(f"Error in stream synchronizer: {e}")
            await asyncio.sleep(0.01)

async def av_publisher(av_sync: rtc.AVSynchronizer, audio_queue: asyncio.Queue, video_queue: asyncio.Queue):
    while True:
        try:

            audio_frame = None
            video_frame = None

            # Try to get audio frame (non-blocking)
            with contextlib.suppress(asyncio.QueueEmpty):
                audio_frame = audio_queue.get_nowait().frame

            # Try to get video frame (non-blocking)
            with contextlib.suppress(asyncio.QueueEmpty):
                video_frame = video_queue.get_nowait().frame

            # Push frames to AVSynchronizer if available
            if audio_frame:
                await av_sync.push(audio_frame)
                logger.debug("Audio frame pushed to synchronizer")

            if video_frame:
                await av_sync.push(video_frame)
                logger.debug("Video frame pushed to synchronizer")

            # Small delay to prevent busy waiting
            if not audio_frame and not video_frame:
                await asyncio.sleep(0.001)  # 1ms

        except Exception as e:
            logger.error(f"Error in stream synchronizer: {e}")
            await asyncio.sleep(0.01)

async def entrypoint(ctx: agents.JobContext):

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)

    # Create publishing audio track
    queue_size_ms = 50
    audio_source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, queue_size_ms=queue_size_ms)
    audio_track = rtc.LocalAudioTrack.create_audio_track("name", audio_source)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.room.local_participant.publish_track(audio_track, audio_options)

    video_source = rtc.VideoSource(384, 384)
    video_track = rtc.LocalVideoTrack.create_video_track("video_track", video_source)
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(video_track, video_options)

    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source, video_source=video_source,
        video_fps=30
    )

    audio_queue = asyncio.Queue[rtc.AudioFrameEvent | None](maxsize=400)
    video_queue = asyncio.Queue[rtc.VideoFrameEvent | None](maxsize=100)

    async_tasks = []

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_task = asyncio.create_task(audio_processor(track, audio_queue))
            async_tasks.append(audio_task)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            video_task = asyncio.create_task(video_processor(track, video_queue))
            async_tasks.append(video_task)


    task = asyncio.create_task(av_publisher(av_sync, audio_queue, video_queue))
    # task = asyncio.create_task(video_publisher(video_source, video_queue))
    async_tasks.append(task)


    await asyncio.gather(*async_tasks)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
