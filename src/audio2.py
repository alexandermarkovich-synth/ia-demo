import asyncio
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


async def audio_processor(audio_stream: rtc.AudioStream, audio_queue: asyncio.Queue):
    """Process incoming audio frames and put them in queue for syncing"""
    async for audio_frame in audio_stream:
        # Identity processing - just pass through the audio frame
        processed_frame = audio_frame
        await audio_queue.put(processed_frame)
        logger.debug(f"Audio frame processed: {audio_frame.samples_per_channel} samples")


async def video_processor(video_stream: rtc.VideoStream, video_queue: asyncio.Queue):
    """Process incoming video frames and put them in queue for syncing"""
    async for video_frame in video_stream:
        # Identity processing - just pass through the video frame
        processed_frame = video_frame
        await video_queue.put(processed_frame)
        logger.debug(f"Video frame processed: {video_frame.width}x{video_frame.height}")


async def stream_synchronizer_and_publisher(
    audio_queue: asyncio.Queue,
    video_queue: asyncio.Queue,
    av_sync: rtc.AVSynchronizer
):
    """Synchronize audio and video streams and publish them"""
    logger.info("Starting stream synchronizer and publisher")
    
    while True:
        try:
            # Get frames from both queues with timeout
            audio_frame = None
            video_frame = None
            
            # Try to get audio frame (non-blocking)
            try:
                audio_frame = audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            
            # Try to get video frame (non-blocking)
            try:
                video_frame = video_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            
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
    logger.info(f"JobContext room: {ctx.room}")
    logger.info(f"JobContext process: {ctx.proc}")

    # Connect to the room first
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)

    # Step 4: Create synchronized output streams
    # Set up audio and video sources for output
    queue_size_ms = 100
    audio_source = rtc.AudioSource(
        sample_rate=SAMPLE_RATE, 
        num_channels=NUM_CHANNELS, 
        queue_size_ms=queue_size_ms
    )
    video_source = rtc.VideoSource(VIDEO_WIDTH, VIDEO_HEIGHT)

    # Create and publish tracks
    audio_track = rtc.LocalAudioTrack.create_audio_track("processed_audio", audio_source)
    video_track = rtc.LocalVideoTrack.create_video_track("processed_video", video_source)
    
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_codec=rtc.VideoCodec.H264
    )
    
    await ctx.room.local_participant.publish_track(audio_track, audio_options)
    await ctx.room.local_participant.publish_track(video_track, video_options)

    # Step 5: Set up AVSynchronizer
    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=VIDEO_FPS,
        video_queue_size_ms=queue_size_ms
    )

    # Step 2: Set up async queues for processing
    audio_queue = asyncio.Queue(maxsize=100)
    video_queue = asyncio.Queue(maxsize=100)

    # Store processing tasks
    processing_tasks = []

    # Step 1: Subscribe to both video and audio tracks
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track subscribed: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            task = asyncio.create_task(
                audio_processor(audio_stream, audio_queue)
            )
            processing_tasks.append(task)
            logger.info("Audio processing task created")
            
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            video_stream = rtc.VideoStream(track)
            task = asyncio.create_task(
                video_processor(video_stream, video_queue)
            )
            processing_tasks.append(task)
            logger.info("Video processing task created")

    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")

    # Start the synchronizer and publisher
    sync_task = asyncio.create_task(
        stream_synchronizer_and_publisher(audio_queue, video_queue, av_sync)
    )
    processing_tasks.append(sync_task)
    
    logger.info("Audio and video streaming app started")
    logger.info("Waiting for participants to join and share audio/video...")

    # Keep the app running
    try:
        await asyncio.gather(*processing_tasks)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean up tasks
        for task in processing_tasks:
            if not task.done():
                task.cancel()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
