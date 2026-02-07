import time
from functools import wraps
from typing import Callable, Any
import asyncio

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

async def async_timing_decorator(func: Callable) -> Callable:
    """Async decorator to measure execution time of functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class PerformanceMonitor:
    """A class to monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, unit: str = "seconds"):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "unit": unit, "timestamp": time.time()})
    
    def get_average_metric(self, name: str) -> float:
        """Get average value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        values = [m["value"] for m in self.metrics[name]]
        return sum(values) / len(values)
    
    def print_report(self):
        """Print a performance report"""
        print("\n📊 PERFORMANCE REPORT")
        print("=" * 50)
        for name, records in self.metrics.items():
            avg_value = self.get_average_metric(name)
            count = len(records)
            print(f"{name}: {avg_value:.2f}s avg over {count} calls")

# Global performance monitor instance
perf_monitor = PerformanceMonitor()