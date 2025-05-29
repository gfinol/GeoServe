import argparse
import asyncio
import json
import os
import time
from asyncio import Future
from typing import Tuple, Dict

import numpy as np
import ray
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle
from vllm import AsyncEngineArgs

from geoserve.async_llm_engine_support import AsyncLLMEngineDeployment
from geoserve.geospatial_deployment_benchmark import GeospatialDeploymentBenchamrk


async def uniform_throughput(handle: DeploymentHandle, queue: asyncio.Queue, geotiff_file: str, num_req: int, rps: int):
    """
    Send requests to the queue at a uniform rate of rps (requests per second).

    Args:
        handle: Deployment handle for the geospatial deployment.
        queue: Queue to send requests to.
        geotiff_file: Path or URL to the GeoTIFF file.
        num_req: Number of requests to send.
        rps: Requests per second.
    """
    for req_id in range(num_req):
        start_time = time.time_ns()
        queue_element = (req_id, start_time, handle.encode.remote(geotiff_file))
        await queue.put(queue_element)
        await asyncio.sleep(1 / rps)


async def benchmark(num_req: int, handle: DeploymentHandle, data_size: int, rps: int, geotiff_file: str, results_dir: str, args: argparse.Namespace):
    """
    Benchmark the geospatial deployment by sending requests and measuring latency.

    Args:
        num_req: Number of inferences to run.
        handle: Deployment handle for the geospatial deployment.
        data_size: Size of data to be passed (in bytes).
        rps: Requests per second.
        geotiff_file: Path or URL to the GeoTIFF file.
        results_dir: Directory to save results.
        args: Configuration used for the benchmark.
    """
    queue = asyncio.Queue()
    results_task = asyncio.create_task(process_outputs(queue, args.num_req))

    print("Starting benchmarking...")
    await uniform_throughput(handle, queue, geotiff_file, num_req, rps)

    results = await results_task
    print("Benchmarking completed.")
    save_results(args, results, results_dir)
    summary_results(results)


async def process_outputs(queue: asyncio.Queue[Tuple[int, float, Future[Tuple[torch.tensor, Dict]]]], num_req: int):
    """
    Process the outputs from the queue.
    Args:
        queue: Queue containing the results of the requests.
        num_req: Number of prompts to process.
    """
    results = []
    for _ in range(num_req):
        # Wait for the next output
        req_id, send_timestamp, task = await queue.get()
        pred, timestamps = await task
        receive_timestamp = time.time_ns()
        queue.task_done()
        result = {"send_time_ns": send_timestamp,
                  "receive_time_ns": receive_timestamp,
                  "latency_ns": receive_timestamp - send_timestamp,
                  "id": req_id}
        result |= timestamps
        results.append(result)

    return results

def save_results(args, results, results_dir):
    """
    Save the results to a file in the specified directory.

    Args:
        args: Configuration used for the benchmark.
        results: List of results to save.
        results_dir: Directory to save the results file.
    """
    # Create a directory for results if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    config = vars(args)

    to_save = {"config": config, "results": results}
    timestamp = time.time_ns()
    results_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(to_save, f, indent=4)

    print(f"Results saved to {results_file}")


def summary_results(results):
    """
    Summarize the results of the benchmark.

    Args:
        results: List of results to summarize.
    """
    latencies = np.array([result["latency_ns"] for result in results])
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    std_latency = np.std(latencies)
    print("-" * 50)
    print(f"Mean Latency: {mean_latency / 1e6:.2f} ms")
    print(f"Standard Deviation Latency: {std_latency / 1e6:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency / 1e6:.2f} ms")
    print(f"99th Percentile Latency: {p99_latency / 1e6:.2f} ms")
    print(f"Total Requests: {len(results)}")
    print("-" * 50)

def print_config(args):
    """
    Print the configuration used for the benchmark.

    Args:
        args: Configuration used for the benchmark.
    """
    print("-" * 50)
    print("Configuration:")
    print("-" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 50)

async def main(arg: argparse.Namespace):
    """
    Main function to run the benchmarking script.
    """

    print_config(arg)

    # Initialize Ray
    ray.init(address=arg.ray_address, runtime_env={"pip": arg.ray_pip_requirements, "env_vars": {"HF_HOME": "/data/gfinol/huggingface"}})
    # ray.init(address=args.ray_address, runtime_env={"image_uri": "icr.io/drl-nextgen/vllm/ad-orchestrator-gpu-0.8.1-vllm", "env_vars": {"HF_HOME": "/data/gfinol/huggingface"}})

    # Create a Ray Serve deployment
    engine_args = AsyncEngineArgs(model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM", skip_tokenizer_init=True, dtype="float32")
    vllm_engine = AsyncLLMEngineDeployment.bind(engine_args=engine_args)
    geospatial_deployment = GeospatialDeploymentBenchamrk.bind(vllm_deployment=vllm_engine, args=arg)

    try:
        handle = serve.run(geospatial_deployment, name=arg.ray_deployment_name)

        # Run the benchmark
        await benchmark(arg.num_req, handle, arg.data_size, arg.rps, arg.geotiff_file, arg.results_dir, arg)

    finally:
        # Clean up
        serve.delete(arg.ray_deployment_name)
        print("Cleaned up resources.")

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--ray-address", type=str, help="Ray cluster address", default="ray://localhost:10001")
    parser.add_argument("--num-req", type=int, help="Number of inferences to run", default=10)
    parser.add_argument("--data-size", type=int, help="Size of data to be passed (in bytes)", default=1 * 1024 * 1024)
    parser.add_argument("--sleep-time", type=float, help="Sleep time in preprocessing (seconds)", default=0.0)
    parser.add_argument("--rps", type=int, help="Requests per second", default=1)
    parser.add_argument("--geotiff-file", type=str, help="Path or URL to the GeoTIFF file", required=True)
    parser.add_argument("--results-dir", type=str, help="Directory to save results", default="results")
    parser.add_argument("--ray-pip-requirements", type=str, help="Path to the Ray Pip requirements file", default="requirements.txt")
    parser.add_argument("--ray-deployment-name", type=str, help="Name of the Ray deployment", default="geoserve-benchmark")
    parser.add_argument("--extra-data-size", type=int, help="Extra data size to be passed (in bytes)", default=0)

    # Arguments that modify the behavior of the preprocessor
    parser.add_argument("--sleep_distribution", type=str, help="Sleep time distribution", choices=["fixed", "uniform"], default="fixed")

    # Fixed sleep time
    parser.add_argument("--sleep_fixed", type=float, help="Fixed sleep time in preprocessing (seconds), requires sleep_distribution=fixed", default=0.1)

    # Uniform sleep time
    parser.add_argument("--sleep_min", type=float, help="Minimum sleep time in preprocessing (seconds), requires sleep_distribution=uniform", default=0.05)
    parser.add_argument("--sleep_max", type=float, help="Maximum sleep time in preprocessing (seconds), requires sleep_distribution=uniform", default=0.1)



    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
