"""Per-chunk CPU memory monitoring using psutil and tracemalloc."""

import os
import tracemalloc
import psutil


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_rss = None
        self._active = False

    def start_session(self):
        if self._active:
            self.stop_session()
        tracemalloc.start()
        self.baseline_rss = self.process.memory_info().rss / (1024 * 1024)
        self._active = True

    def measure(self):
        rss = self.process.memory_info().rss / (1024 * 1024)
        if self._active:
            current, peak = tracemalloc.get_traced_memory()
        else:
            current, peak = 0, 0
        baseline = self.baseline_rss if self.baseline_rss is not None else rss
        return {
            "rss_mb": round(rss, 2),
            "rss_delta_mb": round(rss - baseline, 2),
            "tracemalloc_current_mb": round(current / (1024 * 1024), 4),
            "tracemalloc_peak_mb": round(peak / (1024 * 1024), 4),
        }

    def stop_session(self):
        if self._active:
            tracemalloc.stop()
            self._active = False

    def reset_peak(self):
        if self._active:
            tracemalloc.reset_peak()
