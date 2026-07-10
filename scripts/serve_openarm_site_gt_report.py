"""Serve a Site-GT report with HTTP byte-range support for responsive video seeking."""

from __future__ import annotations

import argparse
import http.server
import os
import pathlib
import shutil
from typing import BinaryIO


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, directory: str, **kwargs) -> None:
        self._byte_range: tuple[int, int] | None = None
        super().__init__(*args, directory=directory, **kwargs)

    def send_head(self) -> BinaryIO | None:
        self._byte_range = None
        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        path = pathlib.Path(self.translate_path(self.path))
        if not path.is_file() or not range_header.startswith("bytes="):
            return super().send_head()

        size = path.stat().st_size
        range_spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        start_text, separator, end_text = range_spec.partition("-")
        if not separator:
            self.send_error(http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix_length = int(end_text)
            start = max(0, size - suffix_length)
            end = size - 1
        if start < 0 or start >= size or end < start:
            self.send_response(http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        end = min(end, size - 1)

        file = path.open("rb")
        self.send_response(http.HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        file.seek(start)
        self._byte_range = (start, end)
        return file

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        if self._byte_range is None:
            shutil.copyfileobj(source, outputfile)
            return
        start, end = self._byte_range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--index-path", default="site_gt_report/index.html")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    index_path = dataset / args.index_path
    if not index_path.exists():
        raise FileNotFoundError(f"Missing report index: {index_path}")

    handler = lambda *handler_args, **handler_kwargs: RangeRequestHandler(  # noqa: E731
        *handler_args,
        directory=os.fspath(dataset),
        **handler_kwargs,
    )
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {dataset} on http://{args.host}:{args.port}/{args.index_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
