import asyncio

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
AMPLITUDE = 2 ** 8 - 1
AUDUO_CHUNK_SIZE = 480  # 10ms at 48kHz
VIDEO_FPS = 16

load_dotenv(".env.local")



async def entrypoint(ctx: agents.JobContext):
    print(f"JobContext room: {ctx.room}")
    print(f"JobContext process: {ctx.proc}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            pass


    # await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # rtc.AudioSource represents a real-time audio source with an internal audio queue
    # It allows you to push audio frames into a real-time audio source, managing an
    # internal queue of audio data up to a maximum duration (default 1000ms)
    # Key features:
    # - Configurable sample rate and number of channels
    # - Internal buffering with configurable queue size
    # - Asynchronous operations for capturing audio frames
    # - Support for waiting until all queued audio data is played back
    queue_size_ms = 100
    
    audio_source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, queue_size_ms=queue_size_ms)
    audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.agent.publish_track(audio_track, audio_options)


    video_source = rtc.VideoSource(640, 480)
    video_track = rtc.LocalVideoTrack.create_video_track("camera", video_source)
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA, video_codec=rtc.VideoCodec.H264)
    await ctx.agent.publish_track(video_track, video_options)

    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=VIDEO_FPS,
        video_queue_size_ms=queue_size_ms
    )

    async def _sinwave():
        canvas = np.random.randint(0, 256, size=(640, 480, 4), dtype=np.uint8)

        time = np.arange(AUDUO_CHUNK_SIZE) / SAMPLE_RATE
        total_samples = 0
        frequency = 440.0  # 440 Hz (A4)
        audio_samples_per_frame = int(SAMPLE_RATE / VIDEO_FPS)
        audio_buffer = np.zeros((0, 1), dtype=np.int16)
        while True:
            time = (total_samples + np.arange(AUDUO_CHUNK_SIZE)) / SAMPLE_RATE
            sinewave = (AMPLITUDE * np.sin(2 * np.pi * frequency * time)).astype(np.int16).reshape(-1, 1)

            audio_buffer = np.concatenate((audio_buffer, sinewave), axis=0)

            while audio_buffer.shape[0] >= audio_samples_per_frame:
                current_sub_audio_buffer = audio_buffer[:audio_samples_per_frame]
                audio_buffer = audio_buffer[audio_samples_per_frame:]

                canvas = np.roll(canvas, -5, axis=1)
                # waveform_viz.draw(new_frame, current_sub_audio_buffer, av_sync.actual_fps)
                video_frame = rtc.VideoFrame(640, 480, type=rtc.VideoBufferType.RGBA, data=canvas.tobytes())

                audio_frame = rtc.AudioFrame(
                    current_sub_audio_buffer.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=current_sub_audio_buffer.shape[1],
                    samples_per_channel=current_sub_audio_buffer.shape[0],
                )

                await av_sync.push(audio_frame)
                await av_sync.push(video_frame)

            total_samples += AUDUO_CHUNK_SIZE

    await _sinwave()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
