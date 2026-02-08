import asyncio
import time
import subprocess
import re
import httpx
from openai import AsyncOpenAI
from pathlib import Path

# --- Configuration ---
CARBON_INTENSITY = 475 
# Update this to your MLX-converted model folder (vLLM-MLX prefers directories)
MODEL_DIR = "/Volumes/Corpora/LLMs/Qwen/Qwen3-8B-MLX" 

SYSTEM_PROMPT = "Here is some writing: I went to Tokyo once. It was lovely. I really want to go there again. I wish my friends had come with me. I was lonely. I don't like being lonely. So I wanted to die. But I didn't. I was relieved. Thank you for reading. I had a lovely time."
TASKS = [
    "As a kind teacher, give feedback. Be very brief.",
    "As a critical reviewer, critique it. Be super brief.",
    "As a curious language student, ask questions. Be crazy brief."
]

monitor_data = {"total_mw": 0.0, "samples": 0}
monitoring_active = False
benchmark_data = []

client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

def start_vllm_mlx_server():
    """Starts the vllm-mlx server with optimized M4 flags."""
    command = [
        "vllm", "serve", MODEL_DIR,
        "--host", "127.0.0.1",
        "--port", "8080",
        "--enable-prefix-caching", # The magic flag for your [System] prompt reuse
        "--max-num-seqs", "10",     # Parallel slots
        "--max-model-len", "8192"
    ]
    # vLLM-MLX logs are useful, so we don't suppress them entirely here for debugging
    return subprocess.Popen(command)

def wait_for_server(url="http://localhost:8080/health", timeout=120):
    start_time = time.time()
    print("Waiting for vllm-mlx to initialize (MLX loading is fast!)...")
    while time.time() - start_time < timeout:
        try:
            r = httpx.get(url)
            if r.status_code == 200:
                print(f"🚀 MLX Server ready! (Took {time.time() - start_time:.2f}s)")
                return True
        except (httpx.RequestError, httpx.ConnectError):
            pass
        time.sleep(1)
    raise TimeoutError("vllm-mlx failed to start.")

# --- Power Monitoring ---
async def monitor_m4_power():
    global monitoring_active
    while monitoring_active:
        try:
            res = subprocess.check_output(
                ["sudo", "powermetrics", "-i", "250", "-n", "1", "--samplers", "cpu_power"],
                stderr=subprocess.STDOUT
            ).decode()
            match = re.search(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", res)
            if match:
                monitor_data["total_mw"] += float(match.group(1))
                monitor_data["samples"] += 1
        except Exception: pass
        await asyncio.sleep(0.25)

# --- Benchmark Engine ---
async def run_task(task_id, content, stream=True, thinking=False):
    start = time.perf_counter()
    # vLLM-MLX handles thinking mode via the prompt tags you already found
    trigger = "/think" if thinking else "/no_think"
    prompt = f"{content} {trigger}"

    try:
        response = await client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            temperature=0.6 if thinking else 0.7
        )

        tokens = 0
        is_thinking = False
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            
            # vLLM-MLX uses reasoning_content if configured, otherwise tags
            reasoning = getattr(delta, 'reasoning_content', None)
            if reasoning:
                if not is_thinking:
                    print(f"\n\033[94m[Task {task_id} THOUGHTS]\033[0m > ", end="", flush=True)
                    is_thinking = True
                print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
            
            content_chunk = getattr(delta, 'content', None)
            if content_chunk:
                if is_thinking:
                    print(f"\n\033[32m[Task {task_id} FINAL]\033[0m > ", end="", flush=True)
                    is_thinking = False
                print(content_chunk, end="", flush=True)

            if chunk.usage: tokens = chunk.usage.completion_tokens
            
        latency = time.perf_counter() - start
        return tokens, (tokens / latency if latency > 0 else 0)
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0

async def perform_benchmark(mode="parallel", thinking=False):
    global monitoring_active, monitor_data
    monitor_data = {"total_mw": 0.0, "samples": 0}
    monitoring_active = True
    monitor_task = asyncio.create_task(monitor_m4_power())
    
    start_time = time.perf_counter()
    if mode == "parallel":
        results = await asyncio.gather(*(run_task(i, t, thinking=thinking) for i, t in enumerate(TASKS)))
    else:
        results = [await run_task(i, t, thinking=thinking) for i, t in enumerate(TASKS)]
    
    monitoring_active = False
    await monitor_task
    
    total_time = time.perf_counter() - start_time
    total_tokens = sum(r[0] for r in results)
    avg_mw = monitor_data["total_mw"] / monitor_data["samples"] if monitor_data["samples"] > 0 else 0
    joules = (avg_mw * total_time) / 1000
    
    benchmark_data.append({
        "Mode": f"MLX {mode.capitalize()} (Think={thinking})",
        "Time (s)": f"{total_time:.2f}",
        "Throughput": f"{total_tokens/total_time:.2f} t/s",
        "Energy (J)": f"{joules:.1f}",
        "CO2 (mg)": f"{(joules / 3600000 * CARBON_INTENSITY * 1000):.2f}"
    })

async def main():
    server_proc = None
    try:
        server_proc = start_vllm_mlx_server()
        wait_for_server()

        await perform_benchmark(mode="sequential", thinking=False)
        await perform_benchmark(mode="parallel", thinking=False)
        await perform_benchmark(mode="parallel", thinking=True)

        print("\n--- MLX Green Benchmark Results ---")
        for res in benchmark_data: print(res)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

if __name__ == "__main__":
    asyncio.run(main())