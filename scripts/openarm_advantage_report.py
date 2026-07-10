"""Reusable KAI0-style HTML renderer for model-predicted OpenArm Stage reports."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import shutil
from typing import Any


@dataclasses.dataclass(frozen=True)
class StageReportConfig:
    title: str
    subtitle: str
    score_source: str
    progress_title: str = "Start-anchored progress"
    advantage_title: str = "Direct 50-frame advantage"
    stage_names: tuple[str, ...] = ("Flattening", "Folding")
    neutral_epsilon: float = 0.01


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _write_text_atomic(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def _validate_episode_payload(payload: dict[str, Any]) -> None:
    episode_index = int(payload["episode_index"])
    length = int(payload["length"])
    if length <= 0:
        raise ValueError(f"Episode {episode_index} has invalid length {length}")
    for key in ("progress", "advantage", "stage_id"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"Episode {episode_index} {key} length does not match {length}")
    for key in ("progress", "advantage"):
        if not all(math.isfinite(float(value)) for value in payload[key]):
            raise ValueError(f"Episode {episode_index} {key} contains non-finite values")
    videos = payload.get("videos")
    if not isinstance(videos, dict) or not any(videos.get(key) for key in ("base", "left_wrist", "right_wrist")):
        raise ValueError(f"Episode {episode_index} has no report video")


def write_stage_report(
    output_dir: pathlib.Path,
    payloads: list[dict[str, Any]],
    summary: dict[str, Any],
    config: StageReportConfig,
) -> pathlib.Path:
    """Write a self-contained report index plus one JSON payload per episode."""

    if not payloads:
        raise ValueError("A Stage report requires at least one episode")
    output_dir = output_dir.resolve()
    payloads = sorted(payloads, key=lambda payload: int(payload["episode_index"]))
    for payload in payloads:
        _validate_episode_payload(payload)

    data_dir = output_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    episode_index_rows = []
    for payload in payloads:
        episode_index = int(payload["episode_index"])
        _write_text_atomic(
            data_dir / f"episode_{episode_index:06d}.json",
            _json_for_script(payload),
        )
        episode_index_rows.append(
            {
                key: payload.get(key)
                for key in (
                    "episode_index",
                    "length",
                    "duration_s",
                    "quality",
                    "eligible_for_k_data",
                    "source_dataset",
                    "source_episode_index",
                    "advantage_mean",
                    "advantage_min",
                    "advantage_max",
                    "negative_fraction",
                    "score_source",
                )
            }
        )

    replacements = {
        "__EPISODES_JSON__": _json_for_script(episode_index_rows),
        "__SUMMARY_JSON__": _json_for_script(summary),
        "__FIRST_EPISODE_JSON__": _json_for_script(payloads[0]),
        "__REPORT_CONFIG_JSON__": _json_for_script(dataclasses.asdict(config)),
        "__REPORT_TITLE__": config.title,
        "__REPORT_SUBTITLE__": config.subtitle,
    }
    page = HTML_TEMPLATE
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    unresolved = [placeholder for placeholder in replacements if placeholder in page]
    if unresolved:
        raise ValueError(f"Unresolved report placeholders: {unresolved}")

    index_path = output_dir / "index.html"
    _write_text_atomic(index_path, page)
    return index_path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__REPORT_TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090a0b;
      --surface: #111214;
      --surface-raised: #17181b;
      --line: #303238;
      --line-soft: #22242a;
      --text: #f4f4f5;
      --muted: #9da1aa;
      --blue: #4b8df8;
      --blue-soft: rgba(75, 141, 248, 0.22);
      --positive: #59b77c;
      --negative: #f07171;
      --neutral: #a5a8b0;
      --stage: #f1b84b;
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, select, input { font: inherit; letter-spacing: 0; }
    button, select { color: var(--text); }
    .topbar { border-bottom: 1px solid var(--line-soft); background: #0d0e10; }
    .topbar-inner, main { width: min(1240px, calc(100% - 32px)); margin: 0 auto; }
    .topbar-inner {
      min-height: 92px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 14px 0;
    }
    .eyebrow { color: var(--blue); font-size: 11px; font-weight: 750; text-transform: uppercase; }
    h1 { margin: 3px 0 0; font-size: 23px; line-height: 1.2; }
    .subtitle { margin-top: 5px; color: var(--muted); max-width: 680px; font-size: 13px; }
    .episode-nav { display: flex; align-items: center; gap: 7px; flex: 0 0 auto; }
    .icon-button, select, .segment-button {
      height: 36px;
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 5px;
    }
    .icon-button {
      width: 38px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      cursor: pointer;
      font-size: 18px;
    }
    .icon-button:hover, select:hover, .segment-button:hover { border-color: #555963; background: var(--surface-raised); }
    .icon-button:disabled { cursor: default; opacity: 0.35; }
    select { min-width: 180px; padding: 0 34px 0 11px; }
    main { padding: 22px 0 40px; }
    .run-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
    }
    .run-stat { min-width: 0; padding: 11px 14px; border-right: 1px solid var(--line-soft); }
    .run-stat:last-child { border-right: 0; }
    .stat-label { color: var(--muted); font-size: 10px; text-transform: uppercase; }
    .stat-value { margin-top: 2px; font-size: 14px; font-weight: 700; overflow-wrap: anywhere; }
    .viewer-head {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 10px;
    }
    .viewer-title { font-size: 15px; font-weight: 750; }
    .viewer-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .segmented {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0d0e10;
    }
    .segment-button {
      height: 30px;
      border-color: transparent;
      padding: 0 11px;
      color: var(--muted);
      cursor: pointer;
      white-space: nowrap;
    }
    .segment-button[aria-pressed="true"] { color: var(--text); border-color: var(--line); background: var(--surface-raised); }
    .segment-button:disabled { opacity: 0.3; cursor: default; }
    .stage-frame {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border: 5px solid var(--neutral);
      border-radius: 5px;
      background: #050506;
      transition: border-color 180ms ease, box-shadow 180ms ease;
    }
    .stage-frame[data-signal="positive"] { border-color: var(--positive); box-shadow: 0 0 0 1px rgba(89,183,124,.12), 0 16px 50px rgba(0,0,0,.42); }
    .stage-frame[data-signal="negative"] { border-color: var(--negative); box-shadow: 0 0 0 1px rgba(240,113,113,.12), 0 16px 50px rgba(0,0,0,.42); }
    .stage-frame[data-signal="neutral"] { border-color: var(--neutral); box-shadow: 0 16px 50px rgba(0,0,0,.42); }
    video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #050506; }
    .video-scrim { position: absolute; inset: 0; background: rgba(0,0,0,.36); pointer-events: none; }
    #overlayCanvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
    .signal-tag, .frame-tag, .stage-tag {
      position: absolute;
      z-index: 2;
      border: 1px solid rgba(255,255,255,.16);
      border-radius: 4px;
      background: rgba(9,10,11,.78);
      backdrop-filter: blur(8px);
      color: #fff;
      pointer-events: none;
    }
    .signal-tag { top: 13px; left: 13px; padding: 6px 9px; font-size: 11px; font-weight: 800; }
    .frame-tag { top: 13px; right: 13px; padding: 6px 9px; font: 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .stage-tag { left: 13px; bottom: 13px; padding: 6px 9px; color: #f6d896; font-size: 11px; }
    .transport {
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 10px 0 8px;
    }
    .transport .icon-button { height: 34px; width: 36px; }
    input[type="range"] { width: 100%; accent-color: var(--blue); cursor: pointer; }
    .time-readout { color: var(--muted); font: 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; min-width: 98px; text-align: right; }
    .mode-band {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 0 14px;
    }
    .legend { color: var(--muted); display: flex; align-items: center; gap: 14px; font-size: 11px; flex-wrap: wrap; }
    .legend-key { display: inline-flex; align-items: center; gap: 6px; }
    .legend-line { width: 17px; height: 3px; border-radius: 2px; background: var(--blue); }
    .legend-line.positive { background: var(--positive); }
    .legend-line.negative { background: var(--negative); }
    .legend-line.reference { background: var(--stage); }
    .metric-strip {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }
    .metric { min-width: 0; padding: 12px 13px; border-right: 1px solid var(--line-soft); }
    .metric:last-child { border-right: 0; }
    .metric-value { margin-top: 3px; font-size: 16px; font-weight: 750; overflow-wrap: anywhere; }
    .metric-value.positive { color: var(--positive); }
    .metric-value.negative { color: var(--negative); }
    .source-band {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 11px;
      padding: 11px 2px 0;
      flex-wrap: wrap;
    }
    @media (max-width: 820px) {
      .topbar-inner { align-items: flex-start; flex-direction: column; gap: 12px; }
      .episode-nav { width: 100%; }
      select { flex: 1; min-width: 0; }
      .run-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .run-stat:nth-child(2) { border-right: 0; }
      .run-stat:nth-child(-n+2) { border-bottom: 1px solid var(--line-soft); }
      .viewer-head, .mode-band { align-items: flex-start; flex-direction: column; }
      .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2n) { border-right: 0; }
      .metric:nth-child(-n+4) { border-bottom: 1px solid var(--line-soft); }
      .camera-control { width: 100%; }
      .camera-control .segment-button { flex: 1; min-width: 0; padding: 0 6px; }
    }
    @media (max-width: 520px) {
      .topbar-inner, main { width: min(100% - 20px, 1240px); }
      h1 { font-size: 20px; }
      .stage-frame { border-width: 3px; }
      .signal-tag, .frame-tag { top: 8px; }
      .signal-tag { left: 8px; }
      .frame-tag { right: 8px; }
      .stage-tag { left: 8px; bottom: 8px; }
      .transport { grid-template-columns: 36px minmax(0, 1fr); }
      .time-readout { grid-column: 1 / -1; min-width: 0; text-align: right; }
      .mode-control { width: 100%; }
      .mode-control .segment-button { flex: 1; padding: 0 7px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <div class="eyebrow" id="scoreSource">Stage Advantage</div>
        <h1>__REPORT_TITLE__</h1>
        <div class="subtitle">__REPORT_SUBTITLE__</div>
      </div>
      <div class="episode-nav">
        <button class="icon-button" id="prev" title="Previous episode" aria-label="Previous episode">&#8592;</button>
        <select id="episodeSelect" aria-label="Episode"></select>
        <button class="icon-button" id="next" title="Next episode" aria-label="Next episode">&#8594;</button>
      </div>
    </div>
  </header>

  <main>
    <section class="run-strip" aria-label="Report summary">
      <div class="run-stat"><div class="stat-label">Episodes</div><div class="stat-value" id="runEpisodes">-</div></div>
      <div class="run-stat"><div class="stat-label">Frames</div><div class="stat-value" id="runFrames">-</div></div>
      <div class="run-stat"><div class="stat-label">Negative frames</div><div class="stat-value" id="runNegative">-</div></div>
      <div class="run-stat"><div class="stat-label">Generated</div><div class="stat-value" id="runGenerated">-</div></div>
    </section>

    <section aria-label="Stage trajectory viewer">
      <div class="viewer-head">
        <div>
          <div class="viewer-title" id="plotTitle">-</div>
          <div class="viewer-subtitle" id="episodeLabel">-</div>
        </div>
        <div class="segmented camera-control" aria-label="Camera">
          <button class="segment-button" data-camera="base" aria-pressed="true">Base</button>
          <button class="segment-button" data-camera="left_wrist" aria-pressed="false">Left wrist</button>
          <button class="segment-button" data-camera="right_wrist" aria-pressed="false">Right wrist</button>
        </div>
      </div>

      <div class="stage-frame" id="stageFrame" data-signal="neutral">
        <video id="mainVideo" muted playsinline preload="metadata"></video>
        <div class="video-scrim"></div>
        <canvas id="overlayCanvas"></canvas>
        <div class="signal-tag" id="signalTag">Neutral</div>
        <div class="frame-tag" id="frameTag">0 / 0</div>
        <div class="stage-tag" id="stageTag">-</div>
      </div>

      <div class="transport">
        <button class="icon-button" id="playToggle" title="Play or pause" aria-label="Play">&#9654;</button>
        <input id="timeline" type="range" min="0" max="1" value="0" step="1" aria-label="Frame timeline">
        <div class="time-readout" id="timeReadout">00:00 / 00:00</div>
      </div>

      <div class="mode-band">
        <div class="segmented mode-control" aria-label="Curve">
          <button class="segment-button" data-mode="progress" aria-pressed="true">Progress</button>
          <button class="segment-button" data-mode="advantage" aria-pressed="false">50-frame advantage</button>
        </div>
        <div class="legend">
          <span class="legend-key"><span class="legend-line"></span>Progress</span>
          <span class="legend-key" id="referenceLegend" hidden><span class="legend-line reference"></span>Human reference</span>
          <span class="legend-key"><span class="legend-line positive"></span>Forward</span>
          <span class="legend-key"><span class="legend-line negative"></span>Regression</span>
        </div>
      </div>

      <div class="metric-strip">
        <div class="metric"><div class="stat-label">Progress</div><div class="metric-value" id="mProgress">-</div></div>
        <div class="metric"><div class="stat-label">Advantage</div><div class="metric-value" id="mAdvantage">-</div></div>
        <div class="metric"><div class="stat-label">Stage</div><div class="metric-value" id="mStage">-</div></div>
        <div class="metric"><div class="stat-label">Frame</div><div class="metric-value" id="mFrame">-</div></div>
        <div class="metric"><div class="stat-label">Mean advantage</div><div class="metric-value" id="mMean">-</div></div>
        <div class="metric"><div class="stat-label">Negative share</div><div class="metric-value" id="mNegative">-</div></div>
      </div>
      <div class="source-band"><span id="sourceText">-</span><span id="qualityText">-</span></div>
    </section>
  </main>

  <script>
    const EPISODES = __EPISODES_JSON__;
    const SUMMARY = __SUMMARY_JSON__;
    const FIRST_EPISODE = __FIRST_EPISODE_JSON__;
    const CONFIG = __REPORT_CONFIG_JSON__;
    const select = document.getElementById('episodeSelect');
    const video = document.getElementById('mainVideo');
    const canvas = document.getElementById('overlayCanvas');
    const timeline = document.getElementById('timeline');
    const frameElement = document.getElementById('stageFrame');
    let current = null;
    let currentIndex = 0;
    let camera = 'base';
    let mode = 'progress';

    document.getElementById('scoreSource').textContent = CONFIG.score_source;
    document.getElementById('runEpisodes').textContent = String(SUMMARY.completed_episodes ?? EPISODES.length);
    document.getElementById('runFrames').textContent = Number(SUMMARY.completed_frames ?? 0).toLocaleString();
    document.getElementById('runNegative').textContent = formatPercent(SUMMARY.negative_frame_fraction);
    document.getElementById('runGenerated').textContent = formatGenerated(SUMMARY.generated_at);

    for (const episode of EPISODES) {
      const option = document.createElement('option');
      option.value = episode.episode_index;
      option.textContent = `Episode ${String(episode.episode_index).padStart(3, '0')}`;
      select.appendChild(option);
    }

    function formatPercent(value) {
      return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '-';
    }

    function formatGenerated(value) {
      if (!value) return '-';
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
    }

    function formatTime(seconds) {
      const safe = Math.max(0, Number(seconds) || 0);
      const minutes = Math.floor(safe / 60);
      const remainder = Math.floor(safe % 60);
      return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
    }

    function episodeFile(index) {
      return `data/episode_${String(index).padStart(6, '0')}.json`;
    }

    function frameFromVideo() {
      if (!current) return 0;
      return Math.max(0, Math.min(current.length - 1, Math.round(video.currentTime * current.fps)));
    }

    function signalAt(frame) {
      const value = Number(current.advantage[frame]);
      if (value > CONFIG.neutral_epsilon) return 'positive';
      if (value < -CONFIG.neutral_epsilon) return 'negative';
      return 'neutral';
    }

    function stageNameAt(frame) {
      const stageId = Math.max(0, Math.min(CONFIG.stage_names.length - 1, Number(current.stage_id[frame]) || 0));
      return CONFIG.stage_names[stageId] ?? `Stage ${stageId + 1}`;
    }

    function loadCamera(nextCamera, frame = 0) {
      if (!current || !current.videos[nextCamera]) return;
      const wasPlaying = !video.paused;
      camera = nextCamera;
      frameElement.style.aspectRatio = String(current.video_aspect_ratios?.[camera] || (16 / 9));
      document.querySelectorAll('[data-camera]').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.camera === camera));
      });
      video.pause();
      video.src = current.videos[camera];
      video.load();
      video.addEventListener('loadedmetadata', () => {
        video.currentTime = Math.min(frame / current.fps, Math.max(0, video.duration - 0.001));
        if (wasPlaying) video.play().catch(() => {});
        updateFrame(frame);
      }, {once: true});
    }

    async function loadEpisode(index) {
      currentIndex = Math.max(0, Math.min(EPISODES.length - 1, index));
      const meta = EPISODES[currentIndex];
      select.value = meta.episode_index;
      current = meta.episode_index === FIRST_EPISODE.episode_index
        ? FIRST_EPISODE
        : await fetch(episodeFile(meta.episode_index)).then(response => response.json());
      timeline.max = String(Math.max(0, current.length - 1));
      timeline.value = '0';
      document.querySelectorAll('[data-camera]').forEach(button => {
        button.disabled = !current.videos[button.dataset.camera];
      });
      camera = current.videos.base ? 'base' : Object.keys(current.videos)[0];
      document.getElementById('episodeLabel').textContent =
        `Episode ${String(current.episode_index).padStart(3, '0')} / ${current.length.toLocaleString()} frames / ${current.fps} fps`;
      document.getElementById('sourceText').textContent =
        `${current.source_dataset || 'OpenArm'} / source episode ${current.source_episode_index ?? current.episode_index}`;
      document.getElementById('qualityText').textContent =
        `${current.score_source || CONFIG.score_source} / ${current.quality || 'predicted'}`;
      document.getElementById('referenceLegend').hidden = !Array.isArray(current.reference_progress);
      document.getElementById('mMean').textContent = Number(current.advantage_mean).toFixed(4);
      const negativeFraction = Number.isFinite(Number(current.negative_fraction))
        ? Number(current.negative_fraction)
        : current.advantage.filter(value => value < 0).length / current.length;
      document.getElementById('mNegative').textContent = formatPercent(negativeFraction);
      document.getElementById('prev').disabled = currentIndex === 0;
      document.getElementById('next').disabled = currentIndex === EPISODES.length - 1;
      loadCamera(camera, 0);
      updateFrame(0);
    }

    function updateFrame(frame) {
      if (!current) return;
      const safeFrame = Math.max(0, Math.min(current.length - 1, Math.round(frame)));
      const progress = Number(current.progress[safeFrame]);
      const advantage = Number(current.advantage[safeFrame]);
      const signal = signalAt(safeFrame);
      const stageName = stageNameAt(safeFrame);
      timeline.value = String(safeFrame);
      frameElement.dataset.signal = signal;
      document.getElementById('signalTag').textContent = signal === 'positive' ? 'Positive' : signal === 'negative' ? 'Negative' : 'Neutral';
      document.getElementById('frameTag').textContent = `${safeFrame} / ${current.length - 1}`;
      document.getElementById('stageTag').textContent = stageName;
      document.getElementById('mProgress').textContent = progress.toFixed(4);
      const advantageMetric = document.getElementById('mAdvantage');
      advantageMetric.textContent = advantage.toFixed(4);
      advantageMetric.className = `metric-value ${signal === 'neutral' ? '' : signal}`;
      document.getElementById('mStage').textContent = stageName;
      document.getElementById('mFrame').textContent = safeFrame.toLocaleString();
      document.getElementById('timeReadout').textContent =
        `${formatTime(safeFrame / current.fps)} / ${formatTime((current.length - 1) / current.fps)}`;
      drawOverlay(safeFrame);
    }

    function prepareCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const pixelWidth = Math.max(1, Math.round(rect.width * ratio));
      const pixelHeight = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      const context = canvas.getContext('2d');
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {context, width: rect.width, height: rect.height};
    }

    function drawOverlay(frame) {
      if (!current) return;
      const {context: ctx, width, height} = prepareCanvas();
      const compact = width < 620;
      const pad = compact
        ? {left: 34, right: 18, top: 42, bottom: 34}
        : {left: 50, right: 28, top: 54, bottom: 42};
      const plotWidth = Math.max(1, width - pad.left - pad.right);
      const plotHeight = Math.max(1, height - pad.top - pad.bottom);
      const values = mode === 'progress' ? current.progress : current.advantage;
      const sortedAbsAdvantage = current.advantage
        .map(value => Math.abs(Number(value)))
        .filter(Number.isFinite)
        .sort((left, right) => left - right);
      const robustIndex = Math.floor(Math.max(0, sortedAbsAdvantage.length - 1) * 0.98);
      const maxAbs = Math.max(0.05, sortedAbsAdvantage[robustIndex] || 0);
      const minValue = mode === 'progress' ? -0.04 : -maxAbs;
      const maxValue = mode === 'progress' ? 1.04 : maxAbs;
      const x = index => pad.left + (index / Math.max(1, current.length - 1)) * plotWidth;
      const y = value => {
        const clipped = Math.max(minValue, Math.min(maxValue, Number(value)));
        return pad.top + ((maxValue - clipped) / (maxValue - minValue)) * plotHeight;
      };
      ctx.clearRect(0, 0, width, height);

      ctx.lineWidth = 1;
      ctx.font = compact ? '9px ui-monospace, monospace' : '11px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(255,255,255,.62)';
      ctx.strokeStyle = 'rgba(255,255,255,.16)';
      const gridValues = mode === 'progress' ? [0, 0.5, 1] : [-maxAbs, 0, maxAbs];
      for (const value of gridValues) {
        const gridY = y(value);
        ctx.beginPath();
        ctx.moveTo(pad.left, gridY);
        ctx.lineTo(width - pad.right, gridY);
        ctx.stroke();
        ctx.fillText(value.toFixed(mode === 'progress' ? 1 : 2), 4, gridY + 3);
      }

      if (mode === 'progress' && current.flatten_done_frame !== null) {
        const boundaryX = x(Number(current.flatten_done_frame));
        ctx.save();
        ctx.strokeStyle = 'rgba(241,184,75,.75)';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(boundaryX, pad.top);
        ctx.lineTo(boundaryX, height - pad.bottom);
        ctx.stroke();
        ctx.restore();
      }

      if (mode === 'progress' && Array.isArray(current.reference_progress)) {
        ctx.save();
        ctx.strokeStyle = 'rgba(241,184,75,.86)';
        ctx.lineWidth = compact ? 1.4 : 2;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        current.reference_progress.forEach((value, index) => {
          if (index === 0) ctx.moveTo(x(index), y(value)); else ctx.lineTo(x(index), y(value));
        });
        ctx.stroke();
        ctx.restore();
      }

      if (mode === 'progress') {
        const fill = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
        fill.addColorStop(0, 'rgba(75,141,248,.24)');
        fill.addColorStop(1, 'rgba(75,141,248,.02)');
        ctx.beginPath();
        ctx.moveTo(x(0), y(values[0]));
        for (let index = 1; index < values.length; index += 1) ctx.lineTo(x(index), y(values[index]));
        ctx.lineTo(x(values.length - 1), height - pad.bottom);
        ctx.lineTo(x(0), height - pad.bottom);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }

      function strokeRange(start, end, alpha, widthValue) {
        if (end <= start) return;
        ctx.lineWidth = widthValue;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        for (let index = Math.max(1, start + 1); index <= end; index += 1) {
          const color = mode === 'progress'
            ? `rgba(75,141,248,${alpha})`
            : Number(values[index]) >= 0
              ? `rgba(89,183,124,${alpha})`
              : `rgba(240,113,113,${alpha})`;
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(x(index - 1), y(values[index - 1]));
          ctx.lineTo(x(index), y(values[index]));
          ctx.stroke();
        }
      }
      strokeRange(0, values.length - 1, 0.34, compact ? 1.3 : 1.7);
      ctx.shadowBlur = compact ? 5 : 9;
      ctx.shadowColor = mode === 'progress' ? 'rgba(75,141,248,.65)' : 'rgba(255,255,255,.24)';
      strokeRange(0, frame, 1, compact ? 2.2 : 3);
      ctx.shadowBlur = 0;

      const playheadX = x(frame);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,.92)';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(playheadX, pad.top);
      ctx.lineTo(playheadX, height - pad.bottom);
      ctx.stroke();
      ctx.restore();

      const pointY = y(values[frame]);
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(playheadX, pointY, compact ? 3 : 4, 0, Math.PI * 2);
      ctx.fill();
      const valueText = Number(values[frame]).toFixed(3);
      ctx.font = compact ? '10px ui-monospace, monospace' : '11px ui-monospace, monospace';
      const labelWidth = ctx.measureText(valueText).width + 14;
      const labelX = Math.max(pad.left, Math.min(width - pad.right - labelWidth, playheadX - labelWidth / 2));
      const labelY = Math.max(pad.top, pointY - 28);
      ctx.fillStyle = 'rgba(9,10,11,.88)';
      ctx.fillRect(labelX, labelY, labelWidth, 20);
      ctx.strokeStyle = 'rgba(255,255,255,.32)';
      ctx.strokeRect(labelX, labelY, labelWidth, 20);
      ctx.fillStyle = '#fff';
      ctx.fillText(valueText, labelX + 7, labelY + 14);

      ctx.fillStyle = 'rgba(255,255,255,.62)';
      ctx.font = compact ? '9px ui-monospace, monospace' : '10px ui-monospace, monospace';
      ctx.fillText('0', pad.left, height - 10);
      const endLabel = String(current.length - 1);
      ctx.fillText(endLabel, width - pad.right - ctx.measureText(endLabel).width, height - 10);
    }

    function seekToFrame(frame) {
      if (!current) return;
      const safeFrame = Math.max(0, Math.min(current.length - 1, Math.round(frame)));
      video.currentTime = safeFrame / current.fps;
      updateFrame(safeFrame);
    }

    video.addEventListener('timeupdate', () => updateFrame(frameFromVideo()));
    video.addEventListener('play', () => {
      document.getElementById('playToggle').innerHTML = '&#10074;&#10074;';
      document.getElementById('playToggle').setAttribute('aria-label', 'Pause');
    });
    video.addEventListener('pause', () => {
      document.getElementById('playToggle').innerHTML = '&#9654;';
      document.getElementById('playToggle').setAttribute('aria-label', 'Play');
    });
    document.getElementById('playToggle').addEventListener('click', () => {
      if (video.paused) video.play().catch(() => {}); else video.pause();
    });
    timeline.addEventListener('input', () => seekToFrame(Number(timeline.value)));
    canvas.addEventListener('click', event => {
      if (!current) return;
      const rect = canvas.getBoundingClientRect();
      const left = rect.width < 620 ? 34 : 50;
      const right = rect.width < 620 ? 18 : 28;
      const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / (rect.width - left - right)));
      seekToFrame(fraction * (current.length - 1));
    });
    document.querySelectorAll('[data-camera]').forEach(button => {
      button.addEventListener('click', () => loadCamera(button.dataset.camera, frameFromVideo()));
    });
    document.querySelectorAll('[data-mode]').forEach(button => {
      button.addEventListener('click', () => {
        mode = button.dataset.mode;
        document.querySelectorAll('[data-mode]').forEach(candidate => {
          candidate.setAttribute('aria-pressed', String(candidate.dataset.mode === mode));
        });
        document.getElementById('plotTitle').textContent =
          mode === 'progress' ? CONFIG.progress_title : CONFIG.advantage_title;
        drawOverlay(frameFromVideo());
      });
    });
    select.addEventListener('change', () => {
      loadEpisode(EPISODES.findIndex(item => item.episode_index === Number(select.value)));
    });
    document.getElementById('prev').addEventListener('click', () => loadEpisode(currentIndex - 1));
    document.getElementById('next').addEventListener('click', () => loadEpisode(currentIndex + 1));
    window.addEventListener('resize', () => drawOverlay(frameFromVideo()));

    document.getElementById('plotTitle').textContent = CONFIG.progress_title;
    loadEpisode(0);
  </script>
</body>
</html>
"""
