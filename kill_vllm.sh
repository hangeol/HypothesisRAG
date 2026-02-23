#!/bin/bash
# Cleanup script to kill all vLLM, TRL, and training processes
# and free GPU memory completely.

echo "=== Stopping all vLLM/TRL/Training processes ==="

# Kill by process name
pkill -9 -f "vllm" 2>/dev/null
pkill -9 -f "trl" 2>/dev/null
pkill -9 -f "train_rewriter" 2>/dev/null
pkill -9 -f "accelerate" 2>/dev/null

sleep 2

# Kill any remaining GPU processes
echo "Killing remaining GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null && echo "  Killed PID $pid"
    fi
done

sleep 2

# Free stale port bindings (weight sync port)
for port in 51216 8080 8081; do
    pids=$(lsof -i :$port -t 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Freeing port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null
    fi
done

sleep 1

# Verify cleanup
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv 2>/dev/null
echo ""
echo "=== Port Status ==="
for port in 8080 8081 51216; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "  Port $port: STILL IN USE"
    else
        echo "  Port $port: FREE"
    fi
done
echo ""
echo "=== Cleanup Complete ==="
