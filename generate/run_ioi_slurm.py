#!/usr/bin/env python3
from math import ceil, gcd
import os
import argparse
import subprocess
from pathlib import Path
from transformers import AutoConfig
import logging

logger = logging.getLogger(__name__)

DEFAULT_TP = 8
MAX_CTX_LENGTH = None

MODEL_CONFIGS = {}

LOGS_DIR = "/fsx/hynek_kydlicek/logs/ioi-eval"
SLURM_SCRIPT_DIR = "/fsx/hynek_kydlicek/slurm/ioi-eval/output"
UV_ENV = "/fsx/hynek_kydlicek/projects/ioi-leaderboard/ioi-eval"


def get_concurrency(model_name: str, concurrency: int) -> int:
    """Get concurrency from model config."""
    return MODEL_CONFIGS.get(model_name, {}).get("concurrency", concurrency)


def get_tp(model_name: str, revision: str) -> int:
    default_tp = MODEL_CONFIGS.get(model_name, {}).get("tp", DEFAULT_TP)
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)

        # Check num_attention_heads and num_key_value_heads, and ensure that both are divisable by tp
        if hasattr(config, 'num_attention_heads'):
            if config.num_attention_heads % default_tp != 0:
                # Adjust tp to be the highest number that divides both num_attention_heads
                new_tp = gcd(config.num_attention_heads, default_tp)
                print(f"Adjusted tp for {model_name} from {default_tp} to {new_tp}")
                return new_tp
        return default_tp
    except Exception as e:
        print(f"Could not get tp from config for {model_name}: {e}")
        return default_tp

def get_context_length(model_name: str, revision: str) -> int:
    """Get maximum context length from model config."""
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        # Check various possible context length attributes
        context_length = (
            getattr(config, 'max_position_embeddings', None) or
            getattr(config, 'sliding_window', None) or
            getattr(config, 'max_sequence_length', None) or
            getattr(config, 'max_seq_len', None) or
            4096  # Default fallback
        )

        # Some models (like Qwen) might have sliding_window disabled
        if hasattr(config, 'use_sliding_window') and not config.use_sliding_window:
            # If sliding window is disabled, use max_position_embeddings instead
            context_length = getattr(config, 'max_position_embeddings', context_length)
            

        # cap to 64k
        if MAX_CTX_LENGTH is not None:
            context_length = min(context_length, MAX_CTX_LENGTH)
        return context_length
    except Exception as e:
        logger.warning(f"Could not get context length from config for {model_name}: {e}")
        return 4096  # Default fallback

def parse_args():
    parser = argparse.ArgumentParser(description="Run IOI evaluation on a model using Slurm")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to evaluate (predefined model name)")
    parser.add_argument("--eval_args", type=str, required=True,
                        help="Arguments to pass to the evaluation script")
    parser.add_argument("--time", type=str, default="7-00:00:00",
                        help="Job time limit (default: 7 days)")
    parser.add_argument("--partition", type=str, default="hopper-prod",
                        help="Slurm partition")
    parser.add_argument("--qos", type=str, default="normal",
                        help="Slurm QOS")
    parser.add_argument("--startup_delay", type=int, default=3600,
                        help="Delay in seconds before starting the server")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate script but don't submit job")

    parser.add_argument("--revision", type=str, default=None, help="Revision to use for the model")
    parser.add_argument("--concurrency", type=int, default=100,
                        help="Number of concurrent requests to the server")
    
    parser.add_argument("--uv_env", type=str, default=None, help="Path to the uv env")
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--slurm_dir", type=str, default=None)
    
    return parser.parse_args()

def create_slurm_script(args, logs_dir):
    # Override with custom values if provided
    concurrency = get_concurrency(args.model, args.concurrency)
    tp = get_tp(args.model, args.revision)
    context_length = get_context_length(args.model, args.revision)
    
    # Create a sanitized model name for the job name
    job_name = f"ioi-eval-{args.model.replace('/', '-')}"

    log_dir = logs_dir / job_name
    log_dir.mkdir(parents=True, exist_ok=True)

    n_nodes = ceil(tp / 8)
    tasks = n_nodes

    revision_arg = f"--revision {args.revision}" if args.revision else ""
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes={n_nodes}
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output={log_dir}/%j-%x.out
#SBATCH --error={log_dir}/%j-%x.out
#SBATCH --time={args.time}
#SBATCH --ntasks-per-node=1

set -exuo pipefail

SERVER_PORT=39877
DIST_PORT=45000

# random sleep (0-100) to prevent ddosing server
sleep $((RANDOM % 100 + 1))

# Environment configuration
export OUTLINES_CACHE_DIR=/scratch/serve_r1/ocache/
export TRITON_HOME=/scratch/serve_r1/triton/
export GLOO_SOCKET_IFNAME="enp71s0"
export NCCL_SOCKET_IFNAME="enp71s0"

# Evaluation script path
EVAL_SCRIPT_PATH="/fsx/hynek_kydlicek/projects/ioi/generate/evaluate.py"

module load cuda/12.4
source ~/.bashrc

# Activate uv
source {args.uv_env or UV_ENV}/bin/activate

FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
FIRST_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$FIRST_NODE" hostname --ip-address)

# Launch servers synchronously across all nodes
srun --nodes={n_nodes} --ntasks={tasks} --ntasks-per-node=1 \\
    bash -c "python -m sglang.launch_server \\
        --model-path '{args.model}' \\
        --tp {tp} \\
        --dist-init-addr '$FIRST_NODE_IP:$DIST_PORT' \\
        {revision_arg} \\
        --nnodes {n_nodes} \\
        --node-rank \\$SLURM_PROCID \\
        --port '$SERVER_PORT' \\
        --host 0.0.0.0 \\
        --trust-remote-code \\
        --max-running-requests {concurrency} \\
        --context-length {context_length}" &

# Wait for server with timeout
TIMEOUT={args.startup_delay}  # 1h, but model loading should take ~30min
START_TIME=$(date +%s)
echo "Waiting for SGLang server (http://$FIRST_NODE_IP:$SERVER_PORT)..."

while true; do
    if curl -s -o /dev/null -w "%{{http_code}}" "http://$FIRST_NODE_IP:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "Server is ready at http://$FIRST_NODE_IP:$SERVER_PORT"
        break
    fi

    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "Error: Server failed to start within $TIMEOUT seconds"
        exit 1
    fi

    echo "Still waiting... ($(($CURRENT_TIME - $START_TIME)) seconds elapsed)"
    sleep 60
done

echo "Checking available models..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/completions" \\
    -H "Content-Type: application/json" \\
    -d '{{
        "model": "default",
        "prompt": "hi, how are you?",
        "max_tokens": 2048,
        "temperature": 0.6
    }}'

python "$EVAL_SCRIPT_PATH" \\
    --model_id "sglang/{args.model}" \\
    {revision_arg} \\
    --api_base "http://localhost:$SERVER_PORT/v1" \\
    --concurrency {concurrency} \\
    {args.eval_args}

# Kill the server and exit
pkill -f "python -m sglang.launch_server"
exit 0
"""
    
    return slurm_script, job_name

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.slurm_dir or SLURM_SCRIPT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(args.logs_dir or LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the Slurm script
    slurm_script, job_name = create_slurm_script(args, logs_dir)
    
    # Create a timestamp for the filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the script to a file
    script_path = output_dir / f"{job_name}_{timestamp}.slurm"
    with open(script_path, "w") as f:
        f.write(slurm_script)
    
    logger.info(f"Slurm script saved to: {script_path}")
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Submit the job if not a dry run
    if not args.dry_run:
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Job submitted: {result.stdout.strip()} find logs at {LOGS_DIR}/{job_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"Error output: {e.stderr}")
    else:
        print("Dry run - job not submitted")

if __name__ == "__main__":
    main() 