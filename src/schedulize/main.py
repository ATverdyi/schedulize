"""
workwindow.py — tiny library for running a task only within configured working hours

Features
- Per‑weekday intervals like "09:00-12:30" (multiple per day).
- Optional overnight spans (e.g. "22:00-02:00").
- Time zone aware via zoneinfo key (Europe/Kyiv by default).
- Holidays/blackout dates (skip whole days).
- Compute: is_open(dt), next_open(dt), next_close(dt), next_run_after(dt, every=timedelta,...)
- Async runner for shell commands or callables. Pauses outside windows, resumes when open.
- Async watcher that toggles state (True inside window, False outside).
- No third‑party deps; Python 3.11+.

Usage (quick):
    schedule = {
        "timezone": "Europe/Kyiv",
        "week": {
            "mon": ["09:00-13:00", "14:00-18:00"],
            "tue": ["09:00-18:00"],
            "wed": [],  # day off
            "thu": ["09:00-18:00"],
            "fri": ["09:00-17:00"],
            "sat": [],
            "sun": [],
        },
        "blackout_dates": ["2025-12-25"],
    }

    ws = WorkSchedule.from_config(schedule)

    # Option 1: periodic jobs
    runner = CommandScheduler(ws, every=timedelta(minutes=15))
    await runner.run_shell(["/usr/bin/python3", "script.py"])

    # Option 2: variable switching
    async def on_state_change(opened: bool):
        global is_open
        is_open = opened
        print("should_execute ->", is_open)

    watcher = WindowWatcher(ws, on_change=on_state_change)
    asyncio.create_task(watcher.run())

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable, Awaitable, Iterable, List, Optional, Tuple
import asyncio
import math
import shlex
import sys
from zoneinfo import ZoneInfo

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

@dataclass(frozen=True)
class Interval:
    start: time  # inclusive
    end: time    # exclusive

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        # overnight like 22:00-02:00
        return t >= self.start or t < self.end

    def is_overnight(self) -> bool:
        return self.start > self.end

    def __str__(self) -> str:
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(hour=int(hh), minute=int(mm))


def parse_interval(spec: str) -> Interval:
    try:
        a, b = spec.split("-")
        return Interval(_parse_hhmm(a.strip()), _parse_hhmm(b.strip()))
    except Exception as e:
        raise ValueError(f"Invalid interval '{spec}': {e}")


def _normalize_day_list(day: str, items: Iterable[str]) -> List[Interval]:
    seen: List[Interval] = []
    for raw in items:
        iv = parse_interval(raw)
        if iv.start == iv.end:
            continue
        seen.append(iv)
    non_overnight = [iv for iv in seen if not iv.is_overnight()]
    non_overnight.sort(key=lambda i: (i.start, i.end))
    merged: List[Interval] = []
    for iv in non_overnight:
        if not merged:
            merged.append(iv)
        else:
            last = merged[-1]
            if iv.start <= last.end:
                merged[-1] = Interval(last.start, max(last.end, iv.end))
            else:
                merged.append(iv)
    overnight = [iv for iv in seen if iv.is_overnight()]
    return merged + overnight


@dataclass
class WorkSchedule:
    tz: ZoneInfo
    week: dict
    blackout: set[date]

    @staticmethod
    def from_config(cfg: dict) -> "WorkSchedule":
        tzname = cfg.get("timezone") or "Europe/Kyiv"
        week_cfg = cfg.get("week") or {}
        week: dict[str, List[Interval]] = {k: [] for k in WEEKDAYS}
        for k in WEEKDAYS:
            raw = week_cfg.get(k, [])
            week[k] = _normalize_day_list(k, raw)
        blackout_items = cfg.get("blackout_dates", [])
        bset = set()
        for s in blackout_items:
            y, m, d = s.split("-")
            bset.add(date(int(y), int(m), int(d)))
        return WorkSchedule(tz=ZoneInfo(tzname), week=week, blackout=bset)

    def _weekday_key(self, dt: datetime) -> str:
        return WEEKDAYS[dt.weekday()]

    def is_blackout(self, d: date) -> bool:
        return d in self.blackout

    def intervals_for(self, d: date) -> List[Interval]:
        return list(self.week[WEEKDAYS[d.weekday()]])

    def is_open(self, dt: datetime) -> bool:
        dt = dt.astimezone(self.tz)
        if self.is_blackout(dt.date()):
            return False
        t = dt.timetz().replace(tzinfo=None)
        for iv in self.intervals_for(dt.date()):
            if iv.contains(t):
                return True
        y = (dt - timedelta(days=1)).date()
        for iv in self.intervals_for(y):
            if iv.is_overnight() and t < iv.end and not self.is_blackout(y):
                return True
        return False

    def next_open(self, dt: datetime) -> Optional[datetime]:
        dt = dt.astimezone(self.tz)
        probe = dt
        for _ in range(366):
            d = probe.date()
            if not self.is_blackout(d):
                for iv in sorted(self.intervals_for(d), key=lambda i: (i.start, i.end)):
                    if iv.is_overnight():
                        cand = datetime.combine(d, iv.start, self.tz)
                        if cand >= dt:
                            return cand
                    else:
                        cand = datetime.combine(d, iv.start, self.tz)
                        if cand >= dt:
                            return cand
            probe = datetime.combine(d + timedelta(days=1), time(0, 0), self.tz)
        return None

    def next_close(self, dt: datetime) -> Optional[datetime]:
        dt = dt.astimezone(self.tz)
        t = dt.timetz().replace(tzinfo=None)
        today = dt.date()
        for iv in sorted(self.intervals_for(today), key=lambda i: (i.start, i.end)):
            if iv.contains(t):
                if iv.is_overnight():
                    return datetime.combine(today + timedelta(days=1), iv.end, self.tz)
                else:
                    return datetime.combine(today, iv.end, self.tz)
        y = today - timedelta(days=1)
        for iv in self.intervals_for(y):
            if iv.is_overnight() and not self.is_blackout(y):
                if t < iv.end:
                    return datetime.combine(today, iv.end, self.tz)
        return None

    def next_run_after(
        self,
        dt: datetime,
        every: timedelta,
        *,
        align_to: Optional[time] = time(0, 0),
        jitter: Optional[timedelta] = None,
    ) -> Optional[datetime]:
        import random
        dt = dt.astimezone(self.tz)
        if not self.is_open(dt):
            nxt = self.next_open(dt)
            if nxt is None:
                return None
            base = nxt
        else:
            base = dt
        if align_to is not None:
            grid_anchor = datetime.combine(base.date(), align_to, self.tz)
            if base < grid_anchor:
                k = 0
            else:
                k = math.ceil((base - grid_anchor) / every)
            base = grid_anchor + k * every
        close = self.next_close(base)
        if close is not None and base >= close:
            nxt = self.next_open(close)
            if nxt is None:
                return None
            base = nxt
            if align_to is not None:
                grid_anchor = datetime.combine(base.date(), align_to, self.tz)
                k = math.ceil((base - grid_anchor) / every)
                base = grid_anchor + k * every
        if jitter:
            base = base + timedelta(seconds=random.uniform(0, jitter.total_seconds()))
        return base


class CommandScheduler:
    def __init__(
        self,
        schedule: WorkSchedule,
        *,
        every: timedelta,
        overlap: str = "skip",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        assert overlap in {"skip", "queue", "cancel"}
        self.schedule = schedule
        self.every = every
        self.overlap = overlap
        self.on_log = on_log or (lambda s: print(s, file=sys.stderr))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._inflight: Optional[asyncio.Task] = None

    async def _sleep_until(self, when: datetime):
        now = datetime.now(self.schedule.tz)
        delay = (when - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)

    async def run_shell(self, cmd: Iterable[str] | str, *, env: Optional[dict] = None):
        import subprocess, shlex
        if isinstance(cmd, str):
            argv = shlex.split(cmd)
        else:
            argv = list(cmd)
        async def _runner():
            p = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await p.communicate()
            return p.returncode, out, err
        await self._run_loop(_runner)

    async def run_callable(self, fn: Callable[[], Awaitable[None]]):
        async def _runner():
            await fn()
            return 0, b"", b""
        await self._run_loop(_runner)

    async def _run_loop(self, job_factory: Callable[[], Awaitable[Tuple[int, bytes, bytes]]]):
        self._running = True
        log = self.on_log
        try:
            while self._running:
                now = datetime.now(self.schedule.tz)
                nxt = self.schedule.next_run_after(now, self.every)
                if nxt is None:
                    log("No further runs found; exiting.")
                    return
                if not self.schedule.is_open(now):
                    open_at = self.schedule.next_open(now)
                    log(f"Closed. Sleeping until open at {open_at}.")
                    await self._sleep_until(nxt)
                else:
                    await self._sleep_until(nxt)
                if self._inflight and not self._inflight.done():
                    if self.overlap == "skip":
                        log("Previous run still in progress; skipping this tick.")
                        continue
                    elif self.overlap == "cancel":
                        log("Cancelling previous run...")
                        self._inflight.cancel()
                        try:
                            await self._inflight
                        except Exception:
                            pass
                if not self.schedule.is_open(datetime.now(self.schedule.tz)):
                    continue
                self._inflight = asyncio.create_task(job_factory())
                try:
                    rc, out, err = await self._inflight
                    if rc != 0:
                        log(f"Job failed rc={rc}: {err.decode(errors='ignore')[:400]}")
                    else:
                        if out:
                            log(out.decode(errors='ignore').strip())
                except asyncio.CancelledError:
                    log("Job cancelled.")
        finally:
            self._running = False

    def stop(self):
        self._running = False
        if self._inflight and not self._inflight.done():
            self._inflight.cancel()


class WindowWatcher:
    """Async watcher: run callback if in/out working window."""

    def __init__(self, schedule: WorkSchedule, *, on_change: Callable[[bool], Awaitable[None]]):
        self.schedule = schedule
        self.on_change = on_change
        self._running = False

    async def run(self):
        self._running = True
        prev_state: Optional[bool] = None
        while self._running:
            now = datetime.now(self.schedule.tz)
            state = self.schedule.is_open(now)
            if state != prev_state:
                await self.on_change(state)
                prev_state = state
            if state:
                nxt = self.schedule.next_close(now)
            else:
                nxt = self.schedule.next_open(now)
            if nxt is None:
                break
            delay = (nxt - now).total_seconds()
            await asyncio.sleep(max(0.0, delay) + 0.2)

    def stop(self):
        self._running = False
