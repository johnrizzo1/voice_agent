# Performance Optimization Guide

## Overview

This guide provides comprehensive strategies for optimizing the performance of the multi-agent voice system. Whether you're dealing with slow response times, high memory usage, or audio latency issues, this guide will help you identify bottlenecks and implement effective solutions.

## 🎯 Performance Goals

### Target Performance Metrics

| Metric                   | Target        | Acceptable   | Notes                                        |
| ------------------------ | ------------- | ------------ | -------------------------------------------- |
| **Voice Response Time**  | < 2 seconds   | < 5 seconds  | Total time from speech end to response start |
| **Agent Routing Time**   | < 0.5 seconds | < 1 second   | Time to select appropriate agent             |
| **Memory Usage**         | < 4GB         | < 8GB        | Total system memory consumption              |
| **CPU Usage**            | < 50%         | < 80%        | Average CPU utilization during operation     |
| **Audio Latency**        | < 100ms       | < 300ms      | Delay between audio input and processing     |
| **Multi-Agent Workflow** | < 10 seconds  | < 20 seconds | Complex multi-step task completion           |

## 🚀 System-Level Optimizations

### Hardware Recommendations

#### Minimum Requirements

- **CPU**: 4-core processor (Intel i5 / AMD Ryzen 5 equivalent)
- **RAM**: 8GB (12GB recommended with multiple agents)
- **Storage**: 20GB free space for models and cache
- **Network**: Stable internet for model downloads

#### Optimal Configuration

- **CPU**: 8+ cores with good single-thread performance
- **RAM**: 16GB+ for smooth multi-agent operations
- **GPU**: CUDA-compatible GPU for accelerated inference
- **Storage**: SSD for faster model loading and file operations
- **Network**: High-speed connection for web search tools

### Operating System Optimizations

#### Linux (Recommended)

```bash
# Increase shared memory for audio processing
echo 'kernel.shmmax = 268435456' >> /etc/sysctl.conf

# Optimize for real-time audio
echo '@audio - rtprio 99' >> /etc/security/limits.conf
echo '@audio - memlock unlimited' >> /etc/security/limits.conf

# Disable swap for consistent performance (if sufficient RAM)
sudo swapoff -a

# Use performance governor
sudo cpupower frequency-set -g performance
```

#### macOS

```bash
# Increase audio buffer size if needed
sudo sysctl -w kern.maxvnodes=263168

# Optimize memory management
sudo sysctl -w vm.swappiness=10
```

#### Windows

```powershell
# Set high performance power plan
powercfg /setactive SCHEME_MIN

# Increase priority for audio processing
# Use Task Manager to set voice_agent process to "High" priority
```

## 🧠 Model Optimization

### Language Model Selection

#### Speed-Optimized Models

```yaml
llm:
  model: "llama3.2:1b" # Fastest, basic quality
  # Alternative: "phi3:mini" (3.8B parameters, good balance)

  # Optimize inference parameters
  temperature: 0.0 # Deterministic generation (faster)
  max_tokens: 256 # Shorter responses
  num_ctx: 2048 # Smaller context window
  keep_alive: "10m" # Keep model loaded longer
```

#### Quality-Optimized Models

```yaml
llm:
  model: "llama3.2:3b" # Good balance of speed/quality
  # Alternative: "mistral:7b" for highest quality

  # Quality parameters
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9
  num_ctx: 4096
```

#### Memory-Optimized Models

```yaml
llm:
  model: "llama3.2:1b" # Smallest footprint

  # Memory optimization
  num_ctx: 1024 # Minimal context
  mlock: false # Don't lock model in memory
  numa: true # Use NUMA optimization
```

### Speech-to-Text Optimization

#### Speed Priority

```yaml
stt:
  model: "whisper-tiny" # Fastest model (~39MB)
  compute_type: "int8" # 8-bit quantization
  beam_size: 1 # Minimal beam search
  best_of: 1 # Single candidate
  device: "cpu" # CPU often faster for tiny models
```

#### Accuracy Priority

```yaml
stt:
  model: "whisper-small" # Good accuracy/speed balance
  compute_type: "float16" # Higher precision
  beam_size: 5 # Better search
  best_of: 5 # Multiple candidates
  device: "cuda" # GPU acceleration if available
```

#### Memory Priority

```yaml
stt:
  model: "whisper-tiny" # Smallest model
  compute_type: "int8" # Minimal memory usage
  device: "cpu" # Lower memory than GPU
```

### Text-to-Speech Optimization

#### Speed Priority

```yaml
tts:
  engine: "espeak" # Fastest TTS engine
  voice: "default"
  speed: 1.2 # Slightly faster speech

  # Minimal processing
  post_tts_cooldown: 0.1 # Shorter cooldown
```

#### Quality Priority

```yaml
tts:
  engine: "bark" # Highest quality neural TTS
  bark_voice_preset: "v2/en_speaker_1" # Consistent voice

  # Quality settings
  post_tts_cooldown: 0.5 # Allow processing time
```

#### Resource Priority

```yaml
tts:
  engine: "pyttsx3" # Uses system TTS (low resource)
  voice: "default"
  speed: 1.0
```

## 🤖 Multi-Agent Optimizations

### Agent Configuration Tuning

#### Speed-Optimized Agent Settings

```yaml
multi_agent:
  # Reduce concurrent agents for speed
  max_concurrent_agents: 3

  # Faster routing
  routing_strategy: "rules_only" # Skip embedding computation
  confidence_threshold: 0.6 # Lower threshold for faster decisions

  # Streamlined workflows
  workflow_orchestration:
    max_concurrent_workflows: 2
    default_timeout: 60 # Shorter timeouts
    enable_parallel_execution: true # Utilize parallelism

  # Minimal communication overhead
  inter_agent_communication:
    message_queue_size: 500
    collaboration_timeout: 60

  # Simple delegation
  enhanced_delegation:
    patterns: ["capability_based"] # Single pattern for speed
```

#### Memory-Optimized Agent Settings

```yaml
multi_agent:
  # Limit concurrent agents
  max_concurrent_agents: 2
  context_window_size: 1000 # Smaller context windows

  # Context management
  context_preservation:
    context_compression: true # Compress contexts
    max_context_age: 1800 # Shorter context lifetime

  # Agent timeouts
  agents:
    general_agent:
      timeout_seconds: 15.0 # Shorter timeouts
      max_concurrent_tasks: 2 # Fewer concurrent tasks
```

### Routing Optimization

#### Rule-Based Routing (Fastest)

```yaml
multi_agent:
  routing_strategy: "rules_only"

  # Optimized routing rules
  routing_rules:
    - name: "math_quick"
      target_agent: "utility_agent"
      patterns: ["calculate", "math", "compute"]
      priority: 1
      confidence: 0.9
```

#### Hybrid Routing (Balanced)

```yaml
multi_agent:
  routing_strategy: "hybrid"
  confidence_threshold: 0.7

  # Fast embedding model
  embedding_model: "nomic-embed-text" # Smaller, faster embeddings
```

#### Embedding-Only Routing (Most Accurate)

```yaml
multi_agent:
  routing_strategy: "embeddings_only"
  embedding_model: "all-MiniLM-L6-v2" # Balance of speed/accuracy
```

## 🔊 Audio Performance Optimization

### Audio Buffer Optimization

#### Low Latency Configuration

```yaml
audio:
  chunk_size: 512 # Smaller chunks for lower latency
  sample_rate: 16000 # Standard rate for speech

  # Aggressive VAD for quick response
  vad_aggressiveness: 3
  min_speech_frames: 6 # Quick speech detection
  max_silence_frames: 10 # Quick silence detection
```

#### High Quality Configuration

```yaml
audio:
  chunk_size: 1024 # Larger chunks for quality
  sample_rate: 22050 # Higher quality audio

  # Conservative VAD for accuracy
  vad_aggressiveness: 2
  min_speech_frames: 8
  max_silence_frames: 15
```

### Audio Device Optimization

#### Device Selection

```python
# Find best audio devices
import pyaudio

def find_best_audio_devices():
    p = pyaudio.PyAudio()

    # Find devices with lowest latency
    best_input = None
    best_output = None
    min_input_latency = float('inf')
    min_output_latency = float('inf')

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # Input device
        if info['maxInputChannels'] > 0:
            if info['defaultLowInputLatency'] < min_input_latency:
                min_input_latency = info['defaultLowInputLatency']
                best_input = i

        # Output device
        if info['maxOutputChannels'] > 0:
            if info['defaultLowOutputLatency'] < min_output_latency:
                min_output_latency = info['defaultLowOutputLatency']
                best_output = i

    p.terminate()
    return best_input, best_output

# Use in configuration
input_device, output_device = find_best_audio_devices()
```

```yaml
audio:
  input_device: 1 # Use best input device ID
  output_device: 2 # Use best output device ID
```

## 💾 Memory Management

### Memory Usage Monitoring

```python
import psutil
import logging

def monitor_memory():
    """Monitor system memory usage."""
    memory = psutil.virtual_memory()

    logging.info(f"Memory Usage: {memory.percent}%")
    logging.info(f"Available: {memory.available / 1024**3:.1f}GB")

    if memory.percent > 85:
        logging.warning("High memory usage detected!")

    return memory.percent

# Set up memory monitoring
import threading
import time

def memory_monitor_thread():
    while True:
        monitor_memory()
        time.sleep(30)  # Check every 30 seconds

# Start monitoring
monitor_thread = threading.Thread(target=memory_monitor_thread, daemon=True)
monitor_thread.start()
```

### Memory Optimization Strategies

#### Model Loading Optimization

```yaml
llm:
  # Load model only when needed
  keep_alive: "2m" # Shorter keep-alive

  # Memory mapping optimization
  mmap: true # Memory map model files
  mlock: false # Don't lock in memory
  numa: true # NUMA-aware allocation
```

#### Context Management

```yaml
multi_agent:
  # Context size limits
  context_window_size: 2000 # Smaller contexts

  context_preservation:
    context_compression: true # Compress stored contexts
    max_context_age: 1800 # 30 minutes max age
    context_cleanup_interval: 300 # Clean up every 5 minutes
```

#### Garbage Collection Tuning

```python
import gc

# Optimize garbage collection
gc.set_threshold(700, 10, 10)  # More frequent collection

# Manual cleanup for long-running processes
def cleanup_memory():
    """Force garbage collection and cleanup."""
    gc.collect()

# Call periodically
import threading

def periodic_cleanup():
    while True:
        time.sleep(300)  # Every 5 minutes
        cleanup_memory()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()
```

## 🔄 Workflow Optimization

### Efficient Workflow Design

#### Parallel Execution

```yaml
multi_agent:
  workflow_orchestration:
    enable_parallel_execution: true
    max_concurrent_workflows: 3

    # Optimize task scheduling
    task_dependency_timeout: 30 # Quick dependency resolution
```

#### Pipeline Optimization

```yaml
multi_agent:
  workflow_orchestration:
    enable_pipeline_execution: true

    # Stream data between pipeline stages
    pipeline_buffer_size: 1000
    enable_streaming: true
```

### Agent Load Balancing

```python
# Custom load balancing strategy
class LoadBalancedRouter:
    def __init__(self):
        self.agent_loads = {}

    def get_least_loaded_agent(self, capable_agents):
        """Select agent with lowest current load."""
        if not capable_agents:
            return None

        # Track agent loads
        min_load = float('inf')
        selected_agent = None

        for agent_id in capable_agents:
            load = self.agent_loads.get(agent_id, 0)
            if load < min_load:
                min_load = load
                selected_agent = agent_id

        return selected_agent

    def update_load(self, agent_id, delta):
        """Update agent load."""
        self.agent_loads[agent_id] = self.agent_loads.get(agent_id, 0) + delta
```

## 📊 Performance Monitoring

### Real-Time Metrics

```python
import time
import threading
from collections import defaultdict, deque

class PerformanceMonitor:
    """Real-time performance monitoring."""

    def __init__(self):
        self.metrics = defaultdict(deque)
        self.start_times = {}

    def start_timer(self, operation):
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end_timer(self, operation):
        """End timing an operation."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)

            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation].popleft()

            del self.start_times[operation]
            return duration
        return 0

    def get_average(self, operation):
        """Get average time for operation."""
        if not self.metrics[operation]:
            return 0
        return sum(self.metrics[operation]) / len(self.metrics[operation])

    def get_stats(self):
        """Get performance statistics."""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        return stats

# Global performance monitor
perf_monitor = PerformanceMonitor()

# Usage in voice agent
async def process_with_monitoring(query):
    perf_monitor.start_timer('total_processing')

    # Agent routing
    perf_monitor.start_timer('agent_routing')
    agent = await route_to_agent(query)
    perf_monitor.end_timer('agent_routing')

    # Agent processing
    perf_monitor.start_timer('agent_processing')
    response = await agent.process(query)
    perf_monitor.end_timer('agent_processing')

    perf_monitor.end_timer('total_processing')

    return response
```

### Performance Alerts

```python
class PerformanceAlerter:
    """Alert on performance issues."""

    def __init__(self, monitor):
        self.monitor = monitor
        self.thresholds = {
            'total_processing': 5.0,      # 5 seconds max
            'agent_routing': 1.0,         # 1 second max
            'agent_processing': 3.0,      # 3 seconds max
        }

    def check_performance(self):
        """Check for performance issues."""
        stats = self.monitor.get_stats()
        alerts = []

        for operation, threshold in self.thresholds.items():
            if operation in stats:
                avg_time = stats[operation]['average']
                if avg_time > threshold:
                    alerts.append(f"{operation}: {avg_time:.2f}s (threshold: {threshold}s)")

        return alerts

    def start_monitoring(self):
        """Start continuous performance monitoring."""
        def monitor_loop():
            while True:
                time.sleep(30)  # Check every 30 seconds
                alerts = self.check_performance()
                if alerts:
                    logging.warning(f"Performance alerts: {alerts}")

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

# Initialize performance alerting
alerter = PerformanceAlerter(perf_monitor)
alerter.start_monitoring()
```

## 🎛️ Configuration Profiles

### Speed Profile

```yaml
# speed_optimized.yaml
audio:
  chunk_size: 512
  vad_aggressiveness: 3

stt:
  model: "whisper-tiny"
  compute_type: "int8"

tts:
  engine: "espeak"

llm:
  model: "llama3.2:1b"
  temperature: 0.0
  max_tokens: 256
  keep_alive: "5m"

multi_agent:
  max_concurrent_agents: 2
  routing_strategy: "rules_only"
  confidence_threshold: 0.6
```

### Quality Profile

```yaml
# quality_optimized.yaml
audio:
  chunk_size: 1024
  sample_rate: 22050

stt:
  model: "whisper-small"
  compute_type: "float16"
  beam_size: 5

tts:
  engine: "bark"
  bark_voice_preset: "v2/en_speaker_1"

llm:
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 512

multi_agent:
  max_concurrent_agents: 5
  routing_strategy: "hybrid"
```

### Balanced Profile

```yaml
# balanced.yaml
audio:
  chunk_size: 1024
  vad_aggressiveness: 3

stt:
  model: "whisper-base"
  compute_type: "float16"

tts:
  engine: "bark"

llm:
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 384

multi_agent:
  max_concurrent_agents: 3
  routing_strategy: "hybrid"
  confidence_threshold: 0.7
```

## 🛠️ Troubleshooting Performance Issues

### Common Performance Problems

#### Slow Response Times

**Symptoms**: Long delays between query and response

**Diagnosis**:

```python
# Add timing to identify bottlenecks
import time

async def diagnose_slow_responses():
    start = time.time()

    # STT timing
    stt_start = time.time()
    text = await stt_service.transcribe(audio)
    stt_time = time.time() - stt_start

    # Routing timing
    route_start = time.time()
    agent = await router.route(text)
    route_time = time.time() - route_start

    # Processing timing
    proc_start = time.time()
    response = await agent.process(text)
    proc_time = time.time() - proc_start

    # TTS timing
    tts_start = time.time()
    audio = await tts_service.synthesize(response)
    tts_time = time.time() - tts_start

    total_time = time.time() - start

    print(f"Total: {total_time:.2f}s")
    print(f"STT: {stt_time:.2f}s")
    print(f"Routing: {route_time:.2f}s")
    print(f"Processing: {proc_time:.2f}s")
    print(f"TTS: {tts_time:.2f}s")
```

**Solutions**:

1. Use faster models (whisper-tiny, smaller LLM)
2. Optimize audio settings (smaller chunks)
3. Use rule-based routing
4. Enable GPU acceleration
5. Increase model keep_alive time

#### High Memory Usage

**Symptoms**: System running out of memory, swapping

**Diagnosis**:

```python
def diagnose_memory():
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()

    print(f"RSS: {memory_info.rss / 1024**2:.1f}MB")
    print(f"VMS: {memory_info.vms / 1024**2:.1f}MB")

    # System memory
    sys_memory = psutil.virtual_memory()
    print(f"System: {sys_memory.percent}% used")
```

**Solutions**:

1. Use smaller models
2. Reduce context window sizes
3. Enable context compression
4. Limit concurrent agents
5. Add memory cleanup routines

#### Audio Latency Issues

**Symptoms**: Delays in audio processing, echoes

**Solutions**:

```yaml
# Low latency audio config
audio:
  chunk_size: 512 # Smaller chunks
  sample_rate: 16000 # Standard rate
  vad_aggressiveness: 3 # Quick detection
  min_speech_frames: 4 # Very responsive
  speech_detection_cooldown: 1.0 # Short cooldown

tts:
  post_tts_cooldown: 0.2 # Minimal delay
```

### Performance Testing Script

```python
#!/usr/bin/env python3
"""Performance testing script for multi-agent voice system."""

import asyncio
import time
import statistics
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator

async def performance_test():
    """Run comprehensive performance tests."""

    # Test queries
    test_queries = [
        "What is 25 times 7?",                    # Simple calculation
        "What's the weather in London?",          # Web API call
        "List files in current directory",       # File operation
        "Calculate 15% tip on $45.50",          # Complex calculation
        "Get weather for Tokyo and save it",     # Multi-agent workflow
    ]

    orchestrator = VoiceAgentOrchestrator()
    await orchestrator.initialize()

    results = {}

    for query in test_queries:
        print(f"\nTesting: {query}")
        times = []

        # Run each query 5 times
        for i in range(5):
            start_time = time.time()
            try:
                response = await orchestrator.process_text(query)
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"  Run {i+1}: {duration:.2f}s")
            except Exception as e:
                print(f"  Run {i+1}: ERROR - {e}")

        if times:
            results[query] = {
                'average': statistics.mean(times),
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times)
            }

    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)

    for query, stats in results.items():
        print(f"\nQuery: {query}")
        print(f"  Average: {stats['average']:.2f}s")
        print(f"  Range: {stats['min']:.2f}s - {stats['max']:.2f}s")
        print(f"  Median: {stats['median']:.2f}s")

    await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(performance_test())
```

This performance optimization guide provides comprehensive strategies for tuning every aspect of the multi-agent voice system. Use these techniques to achieve optimal performance for your specific use case and hardware configuration.
