"""Browser-based OpenArm Stage Advantage annotation tool.

Run this script on a LeRobot v2.1 dataset and open the printed URL. The page
shows the three OpenArm camera videos for one episode and saves Task-A
flattening/folding boundary annotations to a sidecar JSONL file.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
import http.server
import json
import mimetypes
import pathlib
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse

VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenArm Stage Annotator</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #1f2933; }
    header { position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d8dee8; padding: 10px 16px; }
    main { display: grid; grid-template-columns: 290px 1fr; min-height: calc(100vh - 56px); }
    aside { background: #ffffff; border-right: 1px solid #d8dee8; padding: 12px; overflow: auto; }
    section { padding: 12px; overflow: auto; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .stack { display: grid; gap: 8px; }
    label { font-size: 12px; color: #52606d; }
    input, select, textarea, button { font: inherit; }
    input, select, textarea { border: 1px solid #cbd2d9; border-radius: 6px; padding: 6px 8px; background: #ffffff; }
    input[type="number"] { width: 92px; }
    textarea { width: 100%; min-height: 70px; resize: vertical; }
    button { border: 1px solid #a8b3c2; background: #ffffff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button.primary { background: #0b6bcb; color: white; border-color: #0b6bcb; }
    button.warn { background: #fff7ed; border-color: #fdba74; }
    .video-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 10px; }
    .video-card { background: #ffffff; border: 1px solid #d8dee8; border-radius: 8px; padding: 8px; }
    .video-card h3 { margin: 0 0 6px; font-size: 13px; }
    video { width: 100%; aspect-ratio: 16 / 9; background: #111827; display: block; }
    .panel { background: #ffffff; border: 1px solid #d8dee8; border-radius: 8px; padding: 12px; margin-top: 10px; }
    .episodes { max-height: calc(100vh - 190px); overflow: auto; border: 1px solid #d8dee8; border-radius: 8px; }
    .episode { display: flex; justify-content: space-between; padding: 7px 8px; border-bottom: 1px solid #eef2f7; cursor: pointer; }
    .episode.active { background: #e8f3ff; }
    .episode.done { color: #0f7b45; }
    .status { font-size: 12px; color: #52606d; }
    .error { color: #b42318; }
    .ok { color: #0f7b45; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid #d8dee8; }
      .video-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header class="row">
  <strong>OpenArm Stage Annotator</strong>
  <span id="dataset" class="status"></span>
  <span id="saveState" class="status"></span>
</header>
<main>
  <aside class="stack">
    <div class="row">
      <label>Episode</label>
      <input id="episodeJump" type="number" min="0" />
      <button onclick="jumpEpisode()">跳转</button>
    </div>
    <div class="episodes" id="episodes"></div>
  </aside>
  <section>
    <div class="video-grid" id="videos"></div>
    <div class="panel stack">
      <div class="row">
        <button onclick="syncSeek(-1)">-1s</button>
        <button onclick="togglePlay()">播放/暂停</button>
        <button onclick="syncSeek(1)">+1s</button>
        <span class="status">当前帧 <strong id="currentFrame">0</strong></span>
      </div>
      <div class="row">
        <button onclick="mark('episode_start')">S 开始</button>
        <button onclick="mark('flatten_done')">F 展开完成</button>
        <button onclick="mark('fold_start')">G 开始折叠</button>
        <button onclick="mark('episode_end')">E 结束</button>
      </div>
      <div class="row">
        <label>start</label><input id="episode_start" type="number" min="0" />
        <label>flatten_done</label><input id="flatten_done" type="number" min="0" />
        <label>fold_start</label><input id="fold_start" type="number" min="0" />
        <label>end</label><input id="episode_end" type="number" min="0" />
        <label>quality</label>
        <select id="quality">
          <option value="success">success</option>
          <option value="failure">failure</option>
          <option value="partial">partial</option>
        </select>
      </div>
      <label>notes</label>
      <textarea id="notes"></textarea>
      <div class="row">
        <button class="primary" onclick="saveAnnotation()">保存</button>
        <button class="warn" onclick="clearForm()">清空当前表单</button>
        <span id="validation" class="status"></span>
      </div>
    </div>
  </section>
</main>
<script>
let state = { episodes: [], annotations: {}, selected: null, fps: 30, dataset: "" };
const videoKeys = ["observation.images.base", "observation.images.left_wrist", "observation.images.right_wrist"];

async function loadState() {
  const response = await fetch("/api/episodes");
  state = await response.json();
  document.getElementById("dataset").textContent = state.dataset + " | fps=" + state.fps;
  renderEpisodeList();
  if (state.episodes.length) selectEpisode(state.episodes[0].episode_index);
}

function renderEpisodeList() {
  const box = document.getElementById("episodes");
  box.innerHTML = "";
  for (const ep of state.episodes) {
    const div = document.createElement("div");
    const done = state.annotations[String(ep.episode_index)];
    div.className = "episode" + (done ? " done" : "") + (state.selected === ep.episode_index ? " active" : "");
    div.onclick = () => selectEpisode(ep.episode_index);
    div.innerHTML = `<span>#${ep.episode_index}</span><span>${ep.length}f ${done ? "✓" : ""}</span>`;
    box.appendChild(div);
  }
}

function selectEpisode(episodeIndex) {
  state.selected = Number(episodeIndex);
  document.getElementById("episodeJump").value = state.selected;
  renderEpisodeList();
  renderVideos();
  loadAnnotation();
}

function renderVideos() {
  const box = document.getElementById("videos");
  box.innerHTML = "";
  for (const key of videoKeys) {
    const card = document.createElement("div");
    card.className = "video-card";
    const safeKey = encodeURIComponent(key);
    card.innerHTML = `<h3>${key}</h3><video controls muted src="/video/${state.selected}/${safeKey}"></video>`;
    box.appendChild(card);
  }
  for (const video of videos()) {
    video.addEventListener("timeupdate", updateCurrentFrame);
    video.addEventListener("seeked", () => syncAll(video.currentTime));
  }
}

function videos() { return Array.from(document.querySelectorAll("video")); }
function primaryVideo() { return videos()[0]; }
function currentFrame() {
  const v = primaryVideo();
  return Math.max(0, Math.round((v ? v.currentTime : 0) * state.fps));
}
function updateCurrentFrame() { document.getElementById("currentFrame").textContent = currentFrame(); }
function syncAll(time) {
  for (const video of videos()) {
    if (Math.abs(video.currentTime - time) > 0.08) video.currentTime = time;
  }
}
function syncSeek(delta) {
  const v = primaryVideo();
  if (!v) return;
  const next = Math.max(0, v.currentTime + delta);
  for (const video of videos()) video.currentTime = next;
}
function togglePlay() {
  const shouldPlay = videos().some(v => v.paused);
  for (const video of videos()) shouldPlay ? video.play() : video.pause();
}
function mark(name) {
  document.getElementById(name).value = currentFrame();
  validateForm();
}
function clearForm() {
  for (const id of ["episode_start", "flatten_done", "fold_start", "episode_end", "notes"]) {
    document.getElementById(id).value = "";
  }
  document.getElementById("quality").value = "success";
  validateForm();
}
function loadAnnotation() {
  clearForm();
  const ann = state.annotations[String(state.selected)];
  if (!ann) return;
  for (const event of ann.events || []) {
    if (document.getElementById(event.name)) document.getElementById(event.name).value = event.frame;
  }
  document.getElementById("quality").value = ann.quality || "success";
  document.getElementById("notes").value = ann.notes || "";
  validateForm();
}
function value(id) {
  const raw = document.getElementById(id).value;
  return raw === "" ? null : Number(raw);
}
function validateForm() {
  const start = value("episode_start");
  const flat = value("flatten_done");
  const fold = value("fold_start");
  const end = value("episode_end");
  const label = document.getElementById("validation");
  if ([start, flat, fold, end].some(v => v === null)) {
    label.textContent = "未完成";
    label.className = "status";
    return false;
  }
  if (!(start <= flat && flat < fold && fold <= end)) {
    label.textContent = "边界顺序错误";
    label.className = "status error";
    return false;
  }
  label.textContent = "边界有效";
  label.className = "status ok";
  return true;
}
async function saveAnnotation() {
  if (!validateForm()) return;
  const payload = {
    schema_version: "openarm_stage_v1",
    episode_index: state.selected,
    fps: state.fps,
    task: "Fold the T-shirt properly",
    quality: document.getElementById("quality").value,
    events: [
      {name: "episode_start", frame: value("episode_start")},
      {name: "flatten_done", frame: value("flatten_done")},
      {name: "fold_start", frame: value("fold_start")},
      {name: "episode_end", frame: value("episode_end")}
    ],
    stage_boundaries: [
      {stage_id: 0, name: "flattening", start_frame: value("episode_start"), end_frame: value("fold_start") - 1},
      {stage_id: 1, name: "folding", start_frame: value("fold_start"), end_frame: value("episode_end")}
    ],
    notes: document.getElementById("notes").value
  };
  const response = await fetch("/api/annotation", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const result = await response.json();
  document.getElementById("saveState").textContent = result.ok ? "已保存 #" + state.selected : "保存失败";
  await loadState();
  selectEpisode(payload.episode_index);
}
function jumpEpisode() {
  const idx = Number(document.getElementById("episodeJump").value);
  if (state.episodes.some(ep => ep.episode_index === idx)) selectEpisode(idx);
}
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
  if (event.key === " ") { event.preventDefault(); togglePlay(); }
  if (event.key === "ArrowLeft") syncSeek(-1);
  if (event.key === "ArrowRight") syncSeek(1);
  if (event.key.toLowerCase() === "s") mark("episode_start");
  if (event.key.toLowerCase() === "f") mark("flatten_done");
  if (event.key.toLowerCase() === "g") mark("fold_start");
  if (event.key.toLowerCase() === "e") mark("episode_end");
});
for (const id of ["episode_start", "flatten_done", "fold_start", "episode_end"]) {
  document.addEventListener("input", (event) => { if (event.target.id === id) validateForm(); });
}
loadState();
</script>
</body>
</html>
"""


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(
        info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    )


class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class AnnotatorServer:
    def __init__(self, dataset: pathlib.Path, annotations: pathlib.Path, episode_limit: int | None) -> None:
        self.dataset = dataset.resolve()
        self.annotations = annotations.resolve()
        self.info = _load_json(self.dataset / "meta/info.json")
        episodes = _load_jsonl(self.dataset / "meta/episodes.jsonl")
        self.episodes = episodes[:episode_limit] if episode_limit else episodes

    def load_annotations(self) -> dict[int, dict[str, Any]]:
        return {int(row["episode_index"]): row for row in _load_jsonl(self.annotations)}

    def save_annotation(self, annotation: dict[str, Any]) -> None:
        episode_index = int(annotation["episode_index"])
        rows_by_episode = self.load_annotations()
        rows_by_episode[episode_index] = annotation
        _write_jsonl(self.annotations, [rows_by_episode[key] for key in sorted(rows_by_episode)])

    def state_payload(self) -> dict[str, Any]:
        return {
            "dataset": str(self.dataset),
            "fps": int(self.info.get("fps", 30)),
            "episodes": self.episodes,
            "annotations": {str(key): value for key, value in self.load_annotations().items()},
        }

    def video_path(self, episode_index: int, video_key: str) -> pathlib.Path:
        if video_key not in VIDEO_KEYS:
            raise FileNotFoundError(video_key)
        return self.dataset / _format_video_path(self.info, episode_index, video_key)


def make_handler(app: AnnotatorServer) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/episodes":
                self._send_json(app.state_payload())
                return
            if parsed.path.startswith("/video/"):
                self._serve_video(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            app.save_annotation(payload)
            self._send_json({"ok": True})

        def _serve_video(self, path: str) -> None:
            parts = path.split("/", 3)
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                episode_index = int(parts[2])
                key = pathlib.PurePosixPath(parts[3]).as_posix()
                video_path = app.video_path(episode_index, unquote(key))
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not video_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(video_path)

        def _serve_file(self, path: pathlib.Path) -> None:
            size = path.stat().st_size
            start = 0
            end = size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                status = HTTPStatus.PARTIAL_CONTENT
                range_spec = range_header.split("=", 1)[1]
                start_text, _, end_text = range_spec.partition("-")
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else end
                end = min(end, size - 1)
            content_length = end - start + 1
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument(
        "--annotations",
        type=pathlib.Path,
        help="Defaults to <dataset>/annotations/openarm_stage_v1.jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episode-limit", type=int)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    annotations = args.annotations or args.dataset / "annotations/openarm_stage_v1.jsonl"
    app = AnnotatorServer(args.dataset, annotations, args.episode_limit)
    handler = make_handler(app)
    httpd = _ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Annotation file: {annotations}")
    print(f"Open http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
