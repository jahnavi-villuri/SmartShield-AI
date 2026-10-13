"""Read part of a big remote file with HTTP range requests (no full download).

Used to read only a few small columns of very large parquet files (PhreshPhish ships 77 files of
~500 MB each, mostly HTML pages of phishing sites; we only want URL, label, brand and date).
Parquet stores each column separately, so reading three columns fetches only those byte ranges.

Safety: only bytes of the requested columns are fetched; nothing is executed or rendered.
"""

import io
from collections import OrderedDict

import requests

BLOCK = 1 << 20  # fetch in 1 MB aligned blocks to limit the number of requests
MAX_CACHED_BLOCKS = 24
HEADERS = {"User-Agent": "curl/8.4.0"}


class RangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP range requests."""

    def __init__(self, url: str, session: requests.Session | None = None):
        self.session = session or requests.Session()
        resolved = self.session.get(
            url, headers={**HEADERS, "Range": "bytes=0-0"}, stream=True, timeout=120
        )
        resolved.close()
        resolved.raise_for_status()
        self.url = resolved.url  # final address after redirects (a signed storage URL)
        self.size = int(resolved.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.requests_made = 0
        self.bytes_fetched = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _block(self, index: int) -> bytes:
        if index in self._blocks:
            self._blocks.move_to_end(index)
            return self._blocks[index]
        start = index * BLOCK
        end = min(start + BLOCK, self.size) - 1
        response = self.session.get(
            self.url, headers={**HEADERS, "Range": f"bytes={start}-{end}"}, timeout=180
        )
        response.raise_for_status()
        self.requests_made += 1
        self.bytes_fetched += len(response.content)
        self._blocks[index] = response.content
        if len(self._blocks) > MAX_CACHED_BLOCKS:
            self._blocks.popitem(last=False)
        return response.content

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            index, offset = divmod(self.pos, BLOCK)
            chunk = self._block(index)[offset : offset + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
