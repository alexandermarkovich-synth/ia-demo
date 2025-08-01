<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents Starter - Python

A voice AI assistant built with [LiveKit Agents for Python](https://github.com/livekit/agents), featuring OpenAI integration and avatar support.

## Quick Start

### Using Docker Compose (Recommended)

Run the complete stack with a single command:

```bash
docker-compose up
```

**Services:**
- **LiveKit Server** (`livekit:7880`) - WebRTC media server
- **Frontend** (`localhost:3000`) - Next.js web interface for testing
- **Agent** - Python voice AI agent with OpenAI integration
- **Dispatcher** (`localhost:8089`) - Avatar management service

### Manual Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment:
Copy `.env.example` to `.env.local` and set:
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY` 
- `CARTESIA_API_KEY`
- LiveKit credentials (if not using docker-compose)

3. Download required models:
```bash
uv run python src/agent.py download-files
```

4. Run the agent:
```bash
# Console mode (terminal interaction)
uv run python src/agent.py console

# Development mode (for frontend/telephony)
uv run python src/agent.py dev
```

## Features

- Voice AI pipeline with OpenAI, Cartesia TTS, and Deepgram STT
- Avatar integration with visual representation
- Turn detection for natural conversations
- Real-time audio processing
- Web frontend for testing
- **Avatar Runner** (`src/avatar_runner.py`) - Standalone visual avatar that displays animated waveforms synchronized with speech audio

## License

MIT License - see [LICENSE](LICENSE) file for details.