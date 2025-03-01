import subprocess
import json
import time
import os
import argparse
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Monitor dynolog process")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=str, default="0",
                       help="Total monitoring duration (e.g. 600s, 10m, 2h). 0 means infinite.")
    parser.add_argument("--log_dir", type=str, default="log",
                       help="Log directory (default: log)")
    return parser.parse_args()

def parse_duration(duration_str):
    unit_multiplier = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400
    }
    
    if duration_str[-1].isdigit():
        return int(duration_str)
    
    unit = duration_str[-1].lower()
    if unit not in unit_multiplier:
        raise ValueError(f"Invalid duration unit: {unit}")
    
    value = int(duration_str[:-1])
    return value * unit_multiplier[unit]

def get_log_file_name(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"dynolog_stats_{current_time}.json")

def get_dynolog_process_info():
    try:
        pid = subprocess.check_output(["pgrep", "dynolog"]).decode().strip()
    except subprocess.CalledProcessError:
        print("dynolog process not found!")
        return None

    try:
        ps_output = subprocess.check_output([
            "ps", "-o", "pcpu,pmem,rss", "-p", pid
        ])
        ps_output = ps_output.decode().splitlines()[1].split()
    except subprocess.CalledProcessError:
        print("Failed to get process info!")
        return None

    return {
        "timestamp": time.time(),
        "cpu_percent": ps_output[0],
        "mem_percent": ps_output[1],
        "rss": ps_output[2]
    }

def append_to_log(data, log_file):
    with open(log_file, "a") as f:
        json.dump(data, f)
        f.write('\n')

def main():
    args = parse_arguments()
    
    try:
        total_duration = parse_duration(args.duration)
    except ValueError as e:
        print(f"Invalid duration format: {e}")
        return

    log_file = get_log_file_name(args.log_dir)
    interval = args.interval
    next_time = time.time()
    end_time = time.time() + total_duration if total_duration > 0 else float('inf')

    print(f"Start monitoring dynolog (Interval: {interval}s, Duration: {args.duration})")
    print(f"Log file: {log_file}")

    try:
        while time.time() < end_time:
            now = time.time()
            sleep_time = next_time - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            data = get_dynolog_process_info()
            if data is not None:
                append_to_log(data, log_file)
            
            next_time += interval
            
            if total_duration > 0 and time.time() >= end_time:
                break
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("Monitoring completed")

if __name__ == "__main__":
    main()