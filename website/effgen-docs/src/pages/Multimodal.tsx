import React from 'react';
import { Image } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Multimodal() {
  return (
    <DocPage
      title="Multimodal Input"
      subtitle="effGen v0.2.8 makes image, audio, and video first-class input types across 6 cloud providers plus local MLX-VLM. A unified Message content schema, per-provider adapters, automatic preprocessing, and capability gating mean your code only ever speaks effGen — the adapter translates to each provider's format."
      icon={<Image size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Multimodal' },
      ]}
    >
      <InfoBox type="success" title="New in v0.2.8">
        <p>
          The <strong>Multimodal Input</strong> release adds a typed{' '}
          <code>ContentPart</code> schema (<code>TextPart</code>, <code>ImagePart</code>,{' '}
          <code>AudioPart</code>, <code>VideoPart</code>, <code>ToolCallPart</code>,{' '}
          <code>ToolResultPart</code>), provider adapters for image / audio / video, automatic
          preprocessing, a new <code>multimodal</code> preset, the{' '}
          <code>MultimodalDescribeTool</code>, a local <code>MLX-VLM</code> adapter for Apple
          Silicon, and 5 cookbook walkthroughs. No breaking API changes — the old{' '}
          <code>Message(role, "text")</code> constructor still works.
        </p>
      </InfoBox>

      <h2>Unified Message Schema</h2>
      <p>
        <code>Message.content</code> is always a typed <code>List[ContentPart]</code>. The
        constructor remains backwards-compatible: a plain string is auto-wrapped in a{' '}
        <code>TextPart</code>, and the <code>Message.text</code> property joins all text parts.
      </p>
      <CodeBlock
        code={`from effgen.core.messages import (
    Message, Role,
    TextPart, ImagePart, AudioPart, VideoPart,
    ToolCallPart, ToolResultPart,
    ContentPart,
)

# Structured construction
msg = Message(
    role=Role.USER,
    content=[
        ImagePart(image=b"...", mime="image/jpeg"),
        TextPart(text="What is in this image?"),
    ],
)

# Backwards-compatible — still works
msg = Message(role=Role.USER, content="What is in this image?")
print(msg.text)  # "What is in this image?"`}
        language="python"
        filename="message_schema.py"
      />
      <ApiTable
        headers={['ContentPart', 'Fields', 'Validation']}
        rows={[
          [<code>TextPart</code>, 'type="text", text: str', '—'],
          [<code>ImagePart</code>, 'type="image", image: bytes, mime: str, meta: dict', 'MIME ∈ {png, jpeg, gif, webp}'],
          [<code>AudioPart</code>, 'type="audio", audio: bytes, mime: str, duration_s: float | None', 'MIME ∈ {mp3, wav, flac, ogg, m4a}'],
          [<code>VideoPart</code>, 'type="video_frames", frames: List[bytes], fps: float, mime: str', 'frames non-empty'],
          [<code>ToolCallPart</code>, 'type="tool_call", id, name, arguments', '—'],
          [<code>ToolResultPart</code>, 'type="tool_result", id, content', '—'],
        ]}
      />
      <p>
        Invalid MIME types or empty video frames raise{' '}
        <code>InvalidMultimodalContent</code> at construction time.
      </p>

      <h2>Multimodal Helpers</h2>
      <p>
        <code>image_from</code>, <code>audio_from</code>, and <code>video_from</code> accept
        bytes, local paths, URLs, and (for images) <code>PIL.Image</code> /{' '}
        <code>np.ndarray</code>. MIME is sniffed automatically.
      </p>
      <CodeBlock
        code={`from effgen import image_from, audio_from, video_from

# image_from — bytes, path, URL, PIL.Image, np.ndarray
img = image_from("/tmp/photo.jpg")
img = image_from("https://example.com/image.png")
img = image_from(pil_image)          # PIL.Image.Image
img = image_from(numpy_array)        # np.ndarray HxWxC

# audio_from — bytes, path, URL
aud = audio_from("/tmp/recording.mp3")

# video_from — bytes, path, URL; samples keyframes at fps via ffmpeg
vid = video_from("/tmp/clip.mp4", fps=1)   # 1 frame/second, max 16 frames

# v0.3.1: a bare file path in inputs= is auto-wrapped by extension
agent.run("Describe this", inputs=["/tmp/photo.jpg"])   # → ImagePart`}
        language="python"
        filename="helpers.py"
      />

      <h2>Preprocessing Pipeline</h2>
      <p>
        Each modality has a per-provider preprocessor that applies the provider's constraints
        before the request is sent. Every transformation is recorded in{' '}
        <code>part.meta["preprocessing"]</code> for observability.
      </p>
      <ApiTable
        headers={['Module', 'Responsibility']}
        rows={[
          [<code>multimodal/image_pre.py</code>, 'prepare(part, provider, model) → ImagePart. Downscales to provider max pixel dims (e.g. 2048×2048 for Gemini) with PIL Lanczos; converts PNG ↔ JPEG; enforces byte limits.'],
          [<code>multimodal/audio_pre.py</code>, 'Downsamples to 16 kHz mono when required (OpenAI Whisper); chunks clips longer than the provider max duration into sequential calls and concatenates results (pydub).'],
          [<code>multimodal/video_pre.py</code>, 'VideoSource(path_or_url).sample_frames(fps, max_frames) → List[ImagePart]; .extract_audio() → AudioPart | None. Requires ffmpeg.'],
        ]}
      />
      <CodeBlock
        code={`from effgen.multimodal.video_pre import VideoSource

vs = VideoSource("/tmp/clip.mp4")
frames = vs.sample_frames(fps=1, max_frames=16)   # → List[ImagePart]
audio  = vs.extract_audio()                        # → AudioPart | None`}
        language="python"
        filename="video_pre.py"
      />
      <InfoBox type="warning" title="ffmpeg required for video">
        <p>
          Video frame-sampling and audio extraction require <code>ffmpeg</code> on{' '}
          <code>PATH</code>. When absent, effGen raises{' '}
          <code>MissingSystemDependency("ffmpeg", install_hint=...)</code> with per-OS
          install instructions — never a silent failure.
        </p>
      </InfoBox>

      <h2>Capability Gating</h2>
      <p>
        Every adapter checks <code>Capability.vision</code> /{' '}
        <code>Capability.audio_input</code> / <code>Capability.video_input</code> before
        sending. No adapter silently downcasts an image to{' '}
        <code>"[image not supported]"</code> — the error is immediate and explicit.
      </p>
      <CodeBlock
        code={`from effgen.errors import CapabilityNotSupportedError
from effgen.models.capabilities import Capability

try:
    result = model.generate(messages_with_image)
except CapabilityNotSupportedError as e:
    print(e.capability)   # Capability.vision
    print(e.provider)     # "cerebras"`}
        language="python"
        filename="capability_gating.py"
      />

      <h2>Provider Support Matrix</h2>
      <ApiTable
        headers={['Provider', 'Image', 'Audio', 'Video (native)', 'Video (frames)']}
        rows={[
          ['Gemini 2.x/3.x', '✅', '✅', '✅', '✅'],
          ['OpenAI gpt-4o family', '✅', '✅ (Whisper)', '❌', '✅'],
          ['Groq (Llama 4 / 3.2-vision)', '✅', '❌', '❌', '✅'],
          ['Anthropic (code-only, no live key)', '✅', '❌', '❌', '❌'],
          ['Together (vision models)', '✅', '❌', '❌', '✅'],
          ['HuggingFace Inference (BLIP/LLaVA)', '✅', '✅ (ASR)', '❌', '✅'],
          ['Cerebras', '❌', '❌', '❌', '❌'],
          ['MLX-VLM (Apple Silicon only)', '✅', '❌', '❌', '✅'],
        ]}
      />
      <p>
        Providers without native video receive a <code>VideoPart</code> as a sequence of
        sampled <code>ImagePart</code>s (plus an optional <code>AudioPart</code> from the
        audio track). The local <code>MLX-VLM</code> adapter (
        <code>effgen/models/mlx_vlm_engine.py</code>) raises{' '}
        <code>MissingSystemDependency</code> on non-Apple-Silicon hosts.
      </p>

      <h2>The <code>multimodal</code> Preset</h2>
      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)

# Tools wired in: ImageInfoTool, ImageCaptionTool, OCRTool, AudioTranscribeTool,
#                 PDFTool, WeatherTool, MultimodalDescribeTool
result = agent.run("Describe /tmp/chart.png and read off the largest bar.")
print(result.output)`}
        language="python"
        filename="multimodal_preset.py"
      />
      <ApiTable
        headers={['Slot', 'Configuration']}
        rows={[
          ['Primary model', 'Gemini Flash-Lite (vision + audio + video)'],
          ['Fallback', 'OpenAI gpt-4o-mini (vision), HF BLIP (vision-only)'],
          ['Tools', 'ImageInfoTool, ImageCaptionTool, OCRTool, AudioTranscribeTool, PDFTool, WeatherTool, MultimodalDescribeTool'],
        ]}
      />
      <p>
        <code>MultimodalDescribeTool</code> inspects the input part type and automatically
        invokes the right tool (<code>ImageCaption</code>, <code>OCR</code>, or{' '}
        <code>AudioTranscribe</code>) — no manual routing needed.
      </p>

      <h2>Cookbook</h2>
      <p>
        Five end-to-end walkthroughs ship in <code>docs/cookbook/</code>:
      </p>
      <ApiTable
        headers={['Walkthrough', 'What it shows']}
        rows={[
          ['Image Q&A', 'Ask questions about an image with Gemini and OpenAI.'],
          ['Audio transcribe + reason', 'Transcribe an audio clip then run sentiment / content analysis.'],
          ['Video summarize', 'Sample keyframes from a video and produce a narrative summary.'],
          ['OCR + LLM', 'Extract text from an image then apply the contract_summarize_v1 prompt template.'],
          ['Chart reading', 'Read a bar chart from an image and answer comparison questions.'],
        ]}
      />

      <QuickLinks
        links={[
          { icon: 'M', title: 'Models', description: 'Local engines, cloud providers, and capability flags', path: '/models' },
          { icon: 'T', title: 'Tools', description: 'ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF', path: '/tools' },
          { icon: 'P', title: 'Prompts', description: 'Prompt Library templates used in the OCR cookbook', path: '/prompts' },
          { icon: 'R', title: 'Release Notes', description: 'v0.2.8 multimodal release details', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
