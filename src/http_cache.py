"""A tiny on-disk HTTP cache with retry/backoff.

Why this exists
---------------
Collecting ~3,500 days of GDELT data takes roughly an hour of wall-clock time.
Without a cache, every notebook re-run, every bug fix, and every team member
re-hits a *free public service* with the same requests. That is both slow for
us and rude to GDELT. It also makes the dataset non-reproducible: GDELT
back-fills its index, so the same query issued a week apart returns slightly
different results.

The cache therefore does double duty:
  1. speed,
  2. **reproducibility** — once a day is cached, the corpus is frozen. The
     cache directory is the actual scientific artefact; ship it with the repo
     (or a checksum manifest of it) so results can be reproduced exactly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)

# Identify ourselves honestly. Anonymous scraping of a public research service
# is bad practice; a contactable UA lets the operator reach us if we misbehave.
USER_AGENT = (
    "UGM-NLP-GeoUSD/1.0 (academic coursework; contact: nlp-team@mail.ugm.ac.id)"
)

# GDELT's 429 body states: "Please limit requests to one every 5 seconds."
# Backing off by less than that on a 429 is not a retry, it is a second
# offence: it keeps the IP inside the penalty window, so the next attempt is
# refused too and the run collapses into a retry storm. Every retry off a rate
# limit therefore waits at least this long, independently of `backoff_factor`,
# which is tuned for transient network errors and would otherwise start the
# ladder at 1.2s.
RATE_LIMIT_FLOOR_SECONDS = 8.0
RATE_LIMIT_STATUSES = (429, 503)

# Ceiling on a single backoff wait. With backoff_factor=3 the ladder is
# 8, 24, 72, 216, 648s — so one unlucky request could stall a run for a quarter
# of an hour while the progress bar sits still. If four escalating waits have
# not cleared the limit, the IP is in an extended penalty window and waiting
# longer inside one request will not fix it: better to fail that day, let the
# run continue, and pick it up on the next pass (failures are never cached).
MAX_BACKOFF_SECONDS = 120.0


def _key(url: str, params: Optional[Dict[str, Any]]) -> str:
    blob = url + "|" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class CachedSession:
    """requests.Session + disk cache + exponential backoff.

    Parameters
    ----------
    cache_dir : directory for cached payloads (one file per request).
    sleep : polite delay inserted *only* on a cache miss.
    max_retries / backoff : exponential backoff on 429/5xx and network errors.
    """

    def __init__(
        self,
        cache_dir: Path,
        sleep: float = 1.2,
        max_retries: int = 4,
        backoff: float = 2.0,
        timeout: int = 60,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        # `throttled` counts 429/503 responses, not requests. It is diagnostic:
        # a healthy run should sit near zero. A large number means sleep_seconds
        # is too low for the current limit, or another process is sharing the IP.
        self.stats = {"hit": 0, "miss": 0, "fail": 0, "throttled": 0}

    # ------------------------------------------------------------------
    def get_text(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        namespace: str = "misc",
        force: bool = False,
    ) -> Optional[str]:
        """GET and return the body as text, or ``None`` if it permanently failed.

        A permanent failure is cached as an empty sentinel so that a rerun does
        not retry a request we already know is hopeless (e.g. a GDELT day with
        genuinely zero matching articles returns an empty body).
        """
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        fp = ns_dir / f"{_key(url, params)}.txt"

        if fp.exists() and not force:
            self.stats["hit"] += 1
            txt = fp.read_text(encoding="utf-8")
            return txt if txt != "" else None

        delay = self.sleep
        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            rate_limited = False
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    self.stats["miss"] += 1
                    self._write_atomic(fp, resp.text)
                    time.sleep(self.sleep)
                    return resp.text
                if resp.status_code in (404, 400):
                    # Deterministic "no data" — cache the negative result.
                    self._write_atomic(fp, "")
                    self.stats["miss"] += 1
                    return None
                if resp.status_code in RATE_LIMIT_STATUSES:
                    # Not our fault and not permanent: we simply asked too fast.
                    # Never cache it as "no data" — that would silently freeze a
                    # throttled day into the corpus as an empty one.
                    rate_limited = True
                    self.stats["throttled"] += 1
                    delay = max(delay, self._retry_after(resp), RATE_LIMIT_FLOOR_SECONDS)
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:  # network / TLS / timeout
                last_err = type(exc).__name__
                # A server that is actively throttling us also drops connections,
                # so a ConnectionError mid-run is usually the rate limit wearing
                # a different hat. Treat it with the same floor rather than
                # hammering back in 1.2s.
                if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                    rate_limited = True
                    delay = max(delay, RATE_LIMIT_FLOOR_SECONDS)

            if attempt < self.max_retries:
                log.warning(
                    "retry %d/%d for %s (%s) in %.1fs%s",
                    attempt, self.max_retries, url, last_err, delay,
                    " [rate-limited]" if rate_limited else "",
                )
                time.sleep(delay)
                delay = min(delay * self.backoff, MAX_BACKOFF_SECONDS)

        self.stats["fail"] += 1
        log.error("giving up on %s params=%s (%s)", url, params, last_err)
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _retry_after(resp: "requests.Response") -> float:
        """Seconds requested by a ``Retry-After`` header, or 0 if absent/odd.

        GDELT does not currently send one, but honouring it costs nothing and
        is the correct thing to do if it ever starts.
        """
        raw = resp.headers.get("Retry-After")
        if not raw:
            return 0.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0  # HTTP-date form; not worth parsing for this use

    @staticmethod
    def _write_atomic(fp: Path, text: str) -> None:
        """Write via a temp file + replace, so a half-written cache entry is
        never visible.

        Without this, an interrupted run (Ctrl-C, or two processes racing on the
        same key) can leave a truncated JSON body on disk. It would then be
        served as a cache *hit* forever, silently poisoning the corpus with a
        day that looks fetched but is not.
        """
        tmp = fp.with_suffix(fp.suffix + f".tmp{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, fp)

    # ------------------------------------------------------------------
    def get_json(self, url: str, params=None, *, namespace="misc", force=False):
        txt = self.get_text(url, params, namespace=namespace, force=force)
        if not txt:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # GDELT occasionally returns an HTML error page with status 200.
            # Treat that as a miss rather than crashing a 3,500-iteration loop.
            log.warning("non-JSON body for %s params=%s", url, params)
            return None

    def report(self) -> str:
        # Only outcomes of a *request* go in the denominator. `throttled` counts
        # retry attempts, several of which can belong to one eventually-successful
        # request, so including it would understate the hit rate.
        t = self.stats["hit"] + self.stats["miss"] + self.stats["fail"] or 1
        msg = (
            f"cache hits={self.stats['hit']} ({self.stats['hit']/t:.0%}) "
            f"misses={self.stats['miss']} failures={self.stats['fail']}"
        )
        if self.stats["throttled"]:
            msg += f" throttled={self.stats['throttled']}"
        if self.stats["fail"]:
            msg += (
                f"  <-- {self.stats['fail']} requests exhausted every retry and were "
                "DROPPED; those days are missing from this run. Re-run to retry them "
                "(failures are deliberately not cached)."
            )
        return msg


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------
class ConcurrentRunError(RuntimeError):
    """Raised when a second fetching run starts while one is already going."""


@contextmanager
def single_fetch_lock(cache_dir: Path, *, enabled: bool = True):
    """Refuse to start a second network run against the same cache.

    Why this exists
    ---------------
    GDELT's rate limit is enforced **per IP, not per process**. Two
    ``build_dataset`` runs each sleeping a polite 6s therefore present as one
    client issuing a request every 3s, which trips the limit and throttles
    *both*. The failure is quiet and expensive: both runs keep going, every
    call is retried, and throughput collapses (measured: 88s per successful
    fetch instead of ~16s) without either process reporting anything worse
    than a warning.

    They also race on cache writes, since the same query maps to the same key.

    The lock is held by the OS for the lifetime of the process, so killing a
    run — Ctrl-C, a crash, a closed terminal — releases it immediately. There
    is no stale lockfile to clean up by hand.
    """
    if not enabled:
        yield
        return

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".fetch.lock"
    # Mode "a+" would position at EOF, and msvcrt locks a byte range relative to
    # the current position — so two processes would lock *different* bytes and
    # neither would notice the other. Open "r+" and seek(0) so both contend for
    # byte 0. The PID banner is written from byte 1 on, leaving byte 0 as a pure
    # lock token that nobody writes through.
    lock_path.touch(exist_ok=True)
    fh = open(lock_path, "r+", encoding="utf-8")
    try:
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Reading the banner can itself be refused while the holder has the
            # region locked; the diagnosis matters more than the PID.
            try:
                fh.seek(1)
                holder = fh.read().strip() or "another process"
            except OSError:
                holder = "another process"
            raise ConcurrentRunError(
                f"another build_dataset run is already fetching ({holder}).\n"
                "GDELT rate-limits per IP, so a second run would throttle both and "
                "roughly halve the throughput of each. Wait for it to finish, or stop "
                "it first — the cache is shared, so no progress is lost either way.\n"
                f"Lock: {lock_path}"
            ) from None

        fh.seek(1)
        fh.truncate(1)
        fh.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}")
        fh.flush()
        yield
    finally:
        try:
            if os.name == "nt":
                fh.seek(0)
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fh.close()


__all__ = [
    "CachedSession", "USER_AGENT", "single_fetch_lock", "ConcurrentRunError",
    "RATE_LIMIT_FLOOR_SECONDS",
]
