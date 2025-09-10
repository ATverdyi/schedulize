# WorkSchedule

📅 Async-friendly time window scheduler for Python 3.11+.
Designed for background jobs, automation scripts, and IoT/edge cases where you need
precise control over *when* tasks are allowed to run.

---

## ✨ Features

- ⏱ **Per-weekday intervals** like `"09:00-12:30"` (multiple per day).
- 🌙 **Overnight spans** (e.g. `"22:00-02:00"`).
- 🌍 **Time zone aware** via `zoneinfo` key (`Europe/Kyiv` by default).
- 🚫 **Holidays / blackout dates** (skip whole days).
- 🧮 Compute helpers:
  - `is_open(dt)`
  - `next_open(dt)`
  - `next_close(dt)`
  - `next_run_after(dt, every=timedelta, ...)`
- ⚡ **Async runner** for shell commands or callables.
  Pauses outside windows, resumes when open.
- 👀 **Async watcher** that toggles state (`True` inside window, `False` outside).
- 🛠 **No third-party deps**, just Python stdlib.
- ✅ Requires **Python 3.11+**.

---

## 🚀 Quick Usage

```python
import asyncio
from datetime import timedelta
from workschedule import WorkSchedule, CommandScheduler, WindowWatcher

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
asyncio.create_task(runner.run_shell(["/usr/bin/python3", "script.py"]))

# Option 2: variable switching
async def on_state_change(opened: bool):
    global is_open
    is_open = opened
    print("should_execute ->", is_open)

watcher = WindowWatcher(ws, on_change=on_state_change)
asyncio.create_task(watcher.run())
```
