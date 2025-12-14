# OpenAI Models (Normalized)

This document outlines the available OpenAI models, their capabilities, and relevant API information, derived solely from provided sources.

## Introduction

OpenAI offers a diverse range of models, from general-purpose reasoning to specialized capabilities for specific tasks, including open-weight options.
https://platform.openai.com/docs/models

## Authentication

Unknown from provided sources.

## Endpoints/Methods

Model interactions primarily occur via API calls, with specific mentions of:

*   **Chat Completions API**: Used for models like [gpt-audio](https://platform.openai.com/docs/models/gpt-audio) and [GPT-4o Search Preview](https://platform.openai.com/docs/models/gpt-4o-search-preview) for audio inputs/outputs and web search, respectively.
*   **Codex CLI**: Optimized for models like the deprecated [codex-mini-latest](https://platform.openai.com/docs/models/codex-mini-latest).

## Core Workflows

Model capabilities support various core workflows:

*   **Coding and Agentic Tasks**: Advanced GPT-series models are optimized for code generation and executing complex, multi-step agentic workflows.
*   **Reasoning and General Intelligence**: Models designed for complex problem-solving and intelligent responses.
*   **Content Generation**:
    *   **Video**: Generating video with synced audio.
    *   **Image**: State-of-the-art image creation.
*   **Audio Processing**:
    *   **Text-to-Speech (TTS)**: Converting text into natural-sounding speech.
    *   **Speech-to-Text (STT)**: Transcribing spoken language into text, including speaker diarization.
    *   **Realtime Audio/Text**: Processing and generating text and audio inputs/outputs in real-time.
*   **Research**: Powerful models designed for deep research tasks.
*   **Web Search**: Specialized models for web search functionality within chat completions.
*   **Moderation**: Identifying potentially harmful content in text and images.

## Model Catalog

Models are categorized by their primary function and capabilities.

### GPT Series (Frontier, Coding, Reasoning)

*   [GPT-5.2](https://platform.openai.com/docs/models/gpt-5.2) (New): The best model for coding and agentic tasks across industries.
*   [GPT-5.2 pro](https://platform.openai.com/docs/models/gpt-5.2-pro): Version of GPT-5.2 that produces smarter and more precise responses.
*   [GPT-5.1](https://platform.openai.com/docs/models/gpt-5.1): The best model for coding and agentic tasks with configurable reasoning effort.
*   [GPT-5.1 Codex](https://platform.openai.com/docs/models/gpt-5.1-codex): A version of GPT-5.1 optimized for agentic coding in Codex.
*   [GPT-5.1-Codex-Max](https://platform.openai.com/docs/models/gpt-5.1-codex-max): Our most intelligent coding model optimized for long-horizon, agentic coding tasks.
*   [GPT-5.1 Codex mini](https://platform.openai.com/docs/models/gpt-5.1-codex-mini): Smaller, more cost-effective, less-capable version of GPT-5.1-Codex.
*   [GPT-5](https://platform.openai.com/docs/models/gpt-5): Previous intelligent reasoning model for coding and agentic tasks with configurable reasoning effort.
*   [GPT-5-Codex](https://platform.openai.com/docs/models/gpt-5-codex): A version of GPT-5 optimized for agentic coding in Codex.
*   [GPT-5 pro](https://platform.openai.com/docs/models/gpt-5-pro): Version of GPT-5 that produces smarter and more precise responses.
*   [GPT-5 mini](https://platform.openai.com/docs/models/gpt-5-mini): A faster, cost-efficient version of GPT-5 for well-defined tasks.
*   [GPT-5 nano](https://platform.openai.com/docs/models/gpt-5-nano): Fastest, most cost-efficient version of GPT-5.
*   [GPT-4o](https://platform.openai.com/docs/models/gpt-4o): Fast, intelligent, flexible GPT model.
*   [GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini): Fast, affordable small model for focused tasks.
*   [GPT-4.1](https://platform.openai.com/docs/models/gpt-4.1): Smartest non-reasoning model.
*   [GPT-4.1 mini](https://platform.openai.com/docs/models/gpt-4.1-mini): Smaller, faster version of GPT-4.1.
*   [GPT-4.1 nano](https://platform.openai.com/docs/models/gpt-4.1-nano): Fastest, most cost-efficient version of GPT-4.1.
*   [GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4-turbo): An older high-intelligence GPT model.
*   [GPT-4](https://platform.openai.com/docs/models/gpt-4): An older high-intelligence GPT model.
*   [GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3.5-turbo): Legacy GPT model for cheaper chat and non-chat tasks.

### Open-Weight Models (Apache 2.0 License)

*   [gpt-oss-120b](https://platform.openai.com/docs/models/gpt-oss-120b): Most powerful open-weight model, fits into an H100 GPU.
*   [gpt-oss-20b](https://platform.openai.com/docs/models/gpt-oss-20b): Medium-sized open-weight model for low latency.

### Specialized Models

#### Video Generation

*   [Sora 2](https://platform.openai.com/docs/models/sora-2): Flagship video generation with synced audio.
*   [Sora 2 Pro](https://platform.openai.com/docs/models/sora-2-pro): Most advanced synced-audio video generation.

#### Image Generation

*   [GPT Image 1](https://platform.openai.com/docs/models/gpt-image-1): State-of-the-art image generation model.
*   [gpt-image-1-mini](https://platform.openai.com/docs/models/gpt-image-1-mini): A cost-efficient version of GPT Image 1.

#### Deep Research Models

*   [o3-deep-research](https://platform.openai.com/docs/models/o3-deep-research): Our most powerful deep research model.
*   [o4-mini-deep-research](https://platform.openai.com/docs/models/o4-mini-deep-research): Faster, more affordable deep research model.
*   [o3-pro](https://platform.openai.com/docs/models/o3-pro): Version of o3 with more compute for better responses.
*   [o3](https://platform.openai.com/docs/models/o3): Reasoning model for complex tasks, succeeded by GPT-5.
*   [o3-mini](https://platform.openai.com/docs/models/o3-mini): A small model alternative to o3.
*   [o4-mini](https://platform.openai.com/docs/models/o4-mini): Fast, cost-efficient reasoning model, succeeded by GPT-5 mini.
*   [o1-pro](https://platform.openai.com/docs/models/o1-pro): Version of o1 with more compute for better responses.
*   [o1](https://platform.openai.com/docs/models/o1): Previous full o-series reasoning model.

#### Search Models

*   [GPT-4o mini Search Preview](https://platform.openai.com/docs/models/gpt-4o-mini-search-preview): Fast, affordable small model for web search.
*   [GPT-4o Search Preview](https://platform.openai.com/docs/models/gpt-4o-search-preview): GPT model for web search in Chat Completions.

#### Computer Use Models

*   [computer-use-preview](https://platform.openai.com/docs/models/computer-use-preview): Specialized model for computer use tool.

#### Moderation Models

*   [omni-moderation](https://platform.openai.com/docs/models/omni-moderation-latest): Identify potentially harmful content in text and images.

### Audio & Speech Models

#### Realtime Text & Audio

*   [gpt-realtime](https://platform.openai.com/docs/models/gpt-realtime): Model capable of realtime text and audio inputs and outputs.
*   [gpt-realtime-mini](https://platform.openai.com/docs/models/gpt-realtime-mini): A cost-efficient version of GPT Realtime.
*   [GPT-4o mini Realtime](https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview): Smaller realtime model for text and audio inputs and outputs.
*   [GPT-4o Realtime](https://platform.openai.com/docs/models/gpt-4o-realtime-preview): Model capable of realtime text and audio inputs and outputs.

#### Audio Inputs & Outputs (Chat Completions API)

*   [gpt-audio](https://platform.openai.com/docs/models/gpt-audio): For audio inputs and outputs with Chat Completions API.
*   [gpt-audio-mini](https://platform.openai.com/docs/models/gpt-audio-mini): A cost-efficient version of GPT Audio.
*   [GPT-4o Audio](https://platform.openai.com/docs/models/gpt-4o-audio-preview): GPT-4o models capable of audio inputs and outputs.
*   [GPT-4o mini Audio](https://platform.openai.com/docs/models/gpt-4o-mini-audio-preview): Smaller model capable of audio inputs and outputs.

#### Text-to-Speech (TTS)

*   [GPT-4o mini TTS](https://platform.openai.com/docs/models/gpt-4o-mini-tts): Text-to-speech model powered by GPT-4o mini.
*   [TTS-1](https://platform.openai.com/docs/models/tts-1): Text-to-speech model optimized for speed.
*   [TTS-1 HD](https://platform.openai.com/docs/models/tts-1-hd): Text-to-speech model optimized for quality.

#### Speech-to-Text (STT)

*   [GPT-4o Transcribe](https://platform.openai.com/docs/models/gpt-4o-transcribe): Speech-to-text model powered by GPT-4o.
*   [GPT-4o mini Transcribe](https://platform.openai.com/docs/models/gpt-4o-mini-transcribe): Speech-to-text model powered by GPT-4o mini.
*   [GPT-4o Transcribe Diarize](https://platform.openai.com/docs/models/gpt-4o-transcribe-diarize): Transcription model that identifies who's speaking when.
*   [Whisper](https://platform.openai.com/docs/models/whisper-1): General-purpose speech recognition model.

### Embedding Models

*   [text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large): Most capable embedding model.
*   [text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small): Small embedding model.
*   [text-embedding-ada-002](https://platform.openai.com/docs/models/text-embedding-ada-002): Older embedding model.

### ChatGPT Models (Not Recommended for API Use)

*   [GPT-5.2 Chat](https://platform.openai.com/docs/models/gpt-5.2-chat-latest): GPT-5.2 model used in ChatGPT.
*   [GPT-5.1 Chat](https://platform.openai.com/docs/models/gpt-5.1-chat-latest): GPT-5.1 model used in ChatGPT.
*   [GPT-5 Chat](https://platform.openai.com/docs/models/gpt-5-chat-latest): GPT-5 model used in ChatGPT.
*   [ChatGPT-4o](https://platform.openai.com/docs/models/chatgpt-4o-latest): GPT-4o model used in ChatGPT.

### Deprecated Models

These models are deprecated and may be replaced by newer alternatives.

*   [DALL·E 3](https://platform.openai.com/docs/models/dall-e-3): Previous generation image generation model.
*   [GPT-4.5 Preview](https://platform.openai.com/docs/models/gpt-4.5-preview): Deprecated large model.
*   [o1-mini](https://platform.openai.com/docs/models/o1-mini): A small model alternative to o1.
*   [o1 Preview](https://platform.openai.com/docs/models/o1-preview): Preview of our first o-series reasoning model.
*   [babbage-002](https://platform.openai.com/docs/models/babbage-002): Replacement for the GPT-3 ada and babbage base models.
*   [codex-mini-latest](https://platform.openai.com/docs/models/codex-mini-latest): Fast reasoning model optimized for the Codex CLI.
*   [DALL·E 2](https://platform.openai.com/docs/models/dall-e-2): Our first image generation model.
*   [davinci-002](https://platform.openai.com/docs/models/davinci-002): Replacement for the GPT-3 curie and davinci base models.
*   [GPT-4 Turbo Preview](https://platform.openai.com/docs/models/gpt-4-turbo-preview): An older fast GPT model.
*   [text-moderation](https://platform.openai.com/docs/models/text-moderation-latest): Previous generation text-only moderation model.
*   [text-moderation-stable](https://platform.openai.com/docs/models/text-moderation-stable): Previous generation text-only moderation model.

## Code Examples

Unknown from provided sources.

## Common Parameters

Unknown from provided sources.

## Limits

Unknown from provided sources.

---
https://platform.openai.com/docs/models
https://platform.openai.com/docs/models/gpt-5-2