"""Generate an interactive KAI0-style HTML review for an OpenArm Site-GT dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tqdm

VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(
        info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    )


def _episode_payload(dataset: pathlib.Path, info: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    episode_index = int(row["episode_index"])
    frame = pd.read_parquet(
        dataset / _format_data_path(info, episode_index),
        columns=["stage_progress_gt", "stage_id", "advantage_gt"],
    )
    progress = frame["stage_progress_gt"].to_numpy(dtype=np.float32)
    advantage = frame["advantage_gt"].to_numpy(dtype=np.float32)
    stage_ids = frame["stage_id"].to_numpy(dtype=np.int64)
    videos = {
        key.removeprefix("observation.images."): "../" + _format_video_path(info, episode_index, key).as_posix()
        for key in VIDEO_KEYS
    }
    return {
        "episode_index": episode_index,
        "length": len(frame),
        "fps": int(info.get("fps", 30)),
        "duration_s": (len(frame) - 1) / float(info.get("fps", 30)),
        "quality": row.get("annotation_quality", "success"),
        "eligible_for_k_data": bool(row.get("eligible_for_k_data", True)),
        "flatten_done_frame": int(row["flatten_done_frame"]),
        "source_dataset": row.get("source_dataset"),
        "source_episode_index": row.get("source_episode_index"),
        "progress": np.round(progress, 6).tolist(),
        "stage_id": stage_ids.tolist(),
        "advantage": np.round(advantage, 6).tolist(),
        "advantage_min": float(advantage.min()),
        "advantage_mean": float(advantage.mean()),
        "advantage_max": float(advantage.max()),
        "videos": videos,
    }


def build_html_report(dataset: pathlib.Path, output_dir: pathlib.Path, *, overwrite: bool) -> pathlib.Path:
    dataset = dataset.resolve()
    output_dir = output_dir.resolve()
    info = _load_json(dataset / "meta/info.json")
    rows = _load_jsonl(dataset / "meta/episodes.jsonl")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)

    episode_index_rows = []
    first_episode_payload = None
    for row in tqdm.tqdm(rows, desc="Writing report data"):
        payload = _episode_payload(dataset, info, row)
        if first_episode_payload is None:
            first_episode_payload = payload
        episode_index = int(payload["episode_index"])
        (data_dir / f"episode_{episode_index:06d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        episode_index_rows.append(
            {
                key: payload[key]
                for key in (
                    "episode_index",
                    "length",
                    "duration_s",
                    "quality",
                    "eligible_for_k_data",
                    "flatten_done_frame",
                    "source_dataset",
                    "source_episode_index",
                    "advantage_mean",
                    "advantage_max",
                )
            }
        )

    build_report_path = dataset / "site_gt_build_report.json"
    build_report = _load_json(build_report_path) if build_report_path.exists() else {}
    page = HTML_TEMPLATE.replace("__EPISODES_JSON__", json.dumps(episode_index_rows, ensure_ascii=False))
    page = page.replace("__SUMMARY_JSON__", json.dumps(build_report, ensure_ascii=False))
    page = page.replace("__FIRST_EPISODE_JSON__", json.dumps(first_episode_payload, ensure_ascii=False))
    index_path = output_dir / "index.html"
    index_path.write_text(page)
    return index_path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Site-GT Stage Advantage Review</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1c232b;
      --muted: #65717d;
      --line: #d5dbe0;
      --panel: #ffffff;
      --canvas: #f7f8f6;
      --stage-a: #dcebf3;
      --stage-b: #dfeadb;
      --positive: #247a52;
      --negative: #b44337;
      --accent: #315f7c;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: #eef0ed; font: 14px/1.45 Inter, Arial, sans-serif; letter-spacing: 0; }
    header { background: var(--panel); border-bottom: 1px solid var(--line); }
    .header-inner, main { width: min(1500px, calc(100% - 32px)); margin: 0 auto; }
    .header-inner { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }
    h1 { margin: 0; font-size: 22px; font-weight: 700; }
    .subtitle { color: var(--muted); margin-top: 2px; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    button, select { height: 36px; border: 1px solid #adb7bf; background: #fff; color: var(--ink); border-radius: 4px; padding: 0 10px; font: inherit; }
    button { cursor: pointer; min-width: 38px; }
    button:hover { border-color: var(--accent); background: #f2f6f8; }
    select { min-width: 180px; }
    main { padding: 18px 0 28px; }
    .status-band { display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); border: 1px solid var(--line); background: var(--panel); }
    .metric { padding: 10px 12px; border-right: 1px solid var(--line); min-width: 0; }
    .metric:last-child { border-right: 0; }
    .metric-label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .metric-value { margin-top: 2px; font-size: 16px; font-weight: 650; overflow-wrap: anywhere; }
    .quality-success { color: var(--positive); }
    .quality-failure { color: var(--negative); }
    .video-grid { margin-top: 14px; display: grid; grid-template-columns: 1.3fr 1fr 1fr; gap: 10px; }
    .video-view { align-self: start; border: 1px solid var(--line); background: var(--panel); min-width: 0; }
    .video-title { height: 34px; display: flex; align-items: center; padding: 0 10px; border-bottom: 1px solid var(--line); font-size: 12px; font-weight: 650; }
    video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #111; object-fit: contain; }
    .plot-band { margin-top: 14px; border: 1px solid var(--line); background: var(--panel); }
    .plot-header { min-height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 12px; border-bottom: 1px solid var(--line); }
    .plot-title { font-weight: 700; }
    .legend { display: flex; align-items: center; gap: 14px; color: var(--muted); font-size: 12px; flex-wrap: wrap; }
    .key { display: inline-flex; align-items: center; gap: 5px; }
    .swatch { width: 15px; height: 8px; border: 1px solid rgba(0,0,0,.12); }
    .line-key { width: 16px; height: 2px; background: var(--ink); }
    .plot { position: relative; width: 100%; height: 260px; background: var(--canvas); }
    .plot.advantage { height: 210px; border-top: 1px solid var(--line); }
    canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
    .source-note { padding: 9px 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
    @media (max-width: 900px) {
      .header-inner { align-items: flex-start; flex-direction: column; padding: 12px 0; }
      .status-band { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2n) { border-right: 0; }
      .video-grid { grid-template-columns: 1fr; }
      .plot { height: 220px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>Site-GT Stage Advantage Review</h1>
        <div class="subtitle">KAI0-style cumulative progress and frame-wise advantage</div>
      </div>
      <div class="toolbar">
        <button id="prev" title="Previous episode">Prev</button>
        <select id="episodeSelect" aria-label="Episode"></select>
        <button id="next" title="Next episode">Next</button>
      </div>
    </div>
  </header>
  <main>
    <section class="status-band">
      <div class="metric"><div class="metric-label">Episode</div><div class="metric-value" id="mEpisode">-</div></div>
      <div class="metric"><div class="metric-label">Frame</div><div class="metric-value" id="mFrame">-</div></div>
      <div class="metric"><div class="metric-label">Stage</div><div class="metric-value" id="mStage">-</div></div>
      <div class="metric"><div class="metric-label">Progress</div><div class="metric-value" id="mProgress">-</div></div>
      <div class="metric"><div class="metric-label">Advantage</div><div class="metric-value" id="mAdvantage">-</div></div>
      <div class="metric"><div class="metric-label">Quality</div><div class="metric-value" id="mQuality">-</div></div>
    </section>

    <section class="video-grid">
      <div class="video-view"><div class="video-title">Base</div><video id="baseVideo" controls muted preload="auto"></video></div>
      <div class="video-view"><div class="video-title">Left wrist</div><video id="leftVideo" muted preload="auto"></video></div>
      <div class="video-view"><div class="video-title">Right wrist</div><video id="rightVideo" muted preload="auto"></video></div>
    </section>

    <section class="plot-band">
      <div class="plot-header">
        <div class="plot-title">Cumulative progress based on Site-GT</div>
        <div class="legend">
          <span class="key"><span class="swatch" style="background:var(--stage-a)"></span>Flattening</span>
          <span class="key"><span class="swatch" style="background:var(--stage-b)"></span>Folding</span>
          <span class="key"><span class="line-key"></span>Progress</span>
        </div>
      </div>
      <div class="plot"><canvas id="progressPlot"></canvas></div>
      <div class="plot advantage"><canvas id="advantagePlot"></canvas></div>
      <div class="source-note">
        <span id="sourceText">-</span>
        <span id="summaryText">-</span>
      </div>
    </section>
  </main>
  <script>
    const EPISODES = __EPISODES_JSON__;
    const SUMMARY = __SUMMARY_JSON__;
    const FIRST_EPISODE = __FIRST_EPISODE_JSON__;
    const select = document.getElementById('episodeSelect');
    const baseVideo = document.getElementById('baseVideo');
    const leftVideo = document.getElementById('leftVideo');
    const rightVideo = document.getElementById('rightVideo');
    const allVideos = [baseVideo, leftVideo, rightVideo];
    let current = null;
    let currentIndex = 0;
    let syncing = false;

    for (const episode of EPISODES) {
      const option = document.createElement('option');
      option.value = episode.episode_index;
      const qualityMark = episode.quality === 'failure' ? ' [failure]' : '';
      option.textContent = `Episode ${String(episode.episode_index).padStart(3, '0')}${qualityMark}`;
      select.appendChild(option);
    }

    function episodeFile(index) {
      return `data/episode_${String(index).padStart(6, '0')}.json`;
    }

    async function loadEpisode(index) {
      currentIndex = Math.max(0, Math.min(EPISODES.length - 1, index));
      const meta = EPISODES[currentIndex];
      select.value = meta.episode_index;
      current = meta.episode_index === FIRST_EPISODE.episode_index
        ? FIRST_EPISODE
        : await fetch(episodeFile(meta.episode_index)).then(response => response.json());
      const paths = [current.videos.base, current.videos.left_wrist, current.videos.right_wrist];
      allVideos.forEach((video, i) => {
        video.pause();
        video.src = paths[i];
        video.addEventListener('loadeddata', () => { if (video.currentTime === 0) video.currentTime = 0.001; }, {once: true});
        video.load();
      });
      updateFrame(0);
      drawAll(0);
      document.getElementById('mEpisode').textContent = String(current.episode_index).padStart(3, '0');
      const quality = document.getElementById('mQuality');
      quality.textContent = current.quality;
      quality.className = `metric-value quality-${current.quality}`;
      document.getElementById('sourceText').textContent =
        `${current.source_dataset || 'Site'} / source episode ${current.source_episode_index ?? current.episode_index}`;
      document.getElementById('summaryText').textContent =
        `Mean advantage ${current.advantage_mean.toFixed(4)} | Max ${current.advantage_max.toFixed(4)} | K-Data ${current.eligible_for_k_data ? 'eligible' : 'excluded'}`;
    }

    function frameFromVideo() {
      if (!current) return 0;
      return Math.max(0, Math.min(current.length - 1, Math.round(baseVideo.currentTime * current.fps)));
    }

    function updateFrame(frame) {
      if (!current) return;
      const safeFrame = Math.max(0, Math.min(current.length - 1, frame));
      document.getElementById('mFrame').textContent = `${safeFrame} / ${current.length - 1}`;
      document.getElementById('mStage').textContent = current.stage_id[safeFrame] === 0 ? 'Flattening' : 'Folding';
      document.getElementById('mProgress').textContent = current.progress[safeFrame].toFixed(4);
      document.getElementById('mAdvantage').textContent = current.advantage[safeFrame].toFixed(4);
    }

    function prepareCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }

    function drawProgress(frame) {
      const canvas = document.getElementById('progressPlot');
      const {ctx, width, height} = prepareCanvas(canvas);
      const pad = {left: 54, right: 18, top: 24, bottom: 32};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const x = value => pad.left + (value / Math.max(1, current.length - 1)) * plotW;
      const y = value => pad.top + (1 - value) * plotH;
      ctx.clearRect(0, 0, width, height);
      const hasBoundary = current.flatten_done_frame !== null;
      const boundaryX = hasBoundary ? x(current.flatten_done_frame) : width - pad.right;
      ctx.fillStyle = '#dcebf3'; ctx.fillRect(pad.left, pad.top, boundaryX - pad.left, plotH);
      if (hasBoundary) { ctx.fillStyle = '#dfeadb'; ctx.fillRect(boundaryX, pad.top, width - pad.right - boundaryX, plotH); }
      ctx.strokeStyle = '#c7ced3'; ctx.lineWidth = 1;
      ctx.fillStyle = '#65717d'; ctx.font = '11px Arial';
      for (let i = 0; i <= 4; i++) {
        const value = i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y(value)); ctx.lineTo(width - pad.right, y(value)); ctx.stroke();
        ctx.fillText(value.toFixed(2), 12, y(value) + 4);
      }
      ctx.strokeStyle = '#1c232b'; ctx.lineWidth = 1.6; ctx.beginPath();
      current.progress.forEach((value, i) => { const px=x(i), py=y(value); i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py); });
      ctx.stroke();
      drawPlayhead(ctx, x(frame), pad.top, plotH);
      ctx.fillStyle = '#315f7c'; ctx.font = '12px Arial';
      ctx.fillText('Flattening', pad.left + 8, pad.top + 17);
      if (hasBoundary) { ctx.fillStyle = '#247a52'; ctx.fillText('Folding', boundaryX + 8, pad.top + 17); }
      else { ctx.fillStyle = '#b44337'; ctx.fillText('No 0.5 crossing', width - pad.right - 108, pad.top + 17); }
      drawXAxis(ctx, pad, width, height, current.length);
    }

    function drawAdvantage(frame) {
      const canvas = document.getElementById('advantagePlot');
      const {ctx, width, height} = prepareCanvas(canvas);
      const pad = {left: 54, right: 18, top: 22, bottom: 32};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const x = value => pad.left + (value / Math.max(1, current.length - 1)) * plotW;
      const minValue = Math.min(-0.02, current.advantage_min);
      const maxValue = Math.max(0.05, current.advantage_max);
      const y = value => pad.top + (maxValue - value) / (maxValue - minValue) * plotH;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#f7f8f6'; ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#9ca6ad'; ctx.beginPath(); ctx.moveTo(pad.left, y(0)); ctx.lineTo(width-pad.right, y(0)); ctx.stroke();
      ctx.lineWidth = 1.2;
      for (let i = 1; i < current.advantage.length; i++) {
        const positive = current.advantage[i] >= 0;
        ctx.strokeStyle = positive ? '#247a52' : '#b44337';
        ctx.beginPath(); ctx.moveTo(x(i-1), y(current.advantage[i-1])); ctx.lineTo(x(i), y(current.advantage[i])); ctx.stroke();
      }
      ctx.fillStyle = '#65717d'; ctx.font = '11px Arial';
      ctx.fillText(maxValue.toFixed(3), 10, pad.top + 4);
      ctx.fillText('0.000', 10, y(0) + 4);
      ctx.fillText(minValue.toFixed(3), 10, pad.top + plotH);
      ctx.fillStyle = '#1c232b'; ctx.font = '12px Arial'; ctx.fillText('50-frame advantage', pad.left + 8, pad.top + 15);
      drawPlayhead(ctx, x(frame), pad.top, plotH);
      drawXAxis(ctx, pad, width, height, current.length);
    }

    function drawPlayhead(ctx, x, top, height) {
      ctx.strokeStyle = '#b44337'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
    }

    function drawXAxis(ctx, pad, width, height, length) {
      ctx.fillStyle = '#65717d'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
      for (let i=0; i<=4; i++) { const frame=Math.round((length-1)*i/4); const x=pad.left+(width-pad.left-pad.right)*i/4; ctx.fillText(String(frame),x,height-10); }
      ctx.textAlign = 'start';
    }

    function drawAll(frame) { if (current) { drawProgress(frame); drawAdvantage(frame); } }

    function seekFromCanvas(event) {
      if (!current) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const left = 54, right = 18;
      const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / (rect.width - left - right)));
      const frame = Math.round(fraction * (current.length - 1));
      baseVideo.currentTime = frame / current.fps;
      allVideos.slice(1).forEach(video => { video.currentTime = baseVideo.currentTime; });
      updateFrame(frame); drawAll(frame);
    }

    baseVideo.addEventListener('play', () => allVideos.slice(1).forEach(video => video.play().catch(() => {})));
    baseVideo.addEventListener('pause', () => allVideos.slice(1).forEach(video => video.pause()));
    baseVideo.addEventListener('seeking', () => {
      if (syncing) return; syncing = true;
      allVideos.slice(1).forEach(video => { if (Math.abs(video.currentTime-baseVideo.currentTime)>.06) video.currentTime=baseVideo.currentTime; });
      syncing = false;
    });
    baseVideo.addEventListener('timeupdate', () => { const frame=frameFromVideo(); updateFrame(frame); drawAll(frame); });
    document.getElementById('progressPlot').addEventListener('click', seekFromCanvas);
    document.getElementById('advantagePlot').addEventListener('click', seekFromCanvas);
    select.addEventListener('change', () => loadEpisode(EPISODES.findIndex(item => item.episode_index === Number(select.value))));
    document.getElementById('prev').addEventListener('click', () => loadEpisode(currentIndex - 1));
    document.getElementById('next').addEventListener('click', () => loadEpisode(currentIndex + 1));
    window.addEventListener('resize', () => drawAll(frameFromVideo()));

    const eligible = EPISODES.filter(item => item.eligible_for_k_data).length;
    document.getElementById('summaryText').textContent = `${EPISODES.length} episodes | ${eligible} K-Data eligible`;
    loadEpisode(0);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset / "site_gt_report"
    print(build_html_report(args.dataset, output_dir, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
