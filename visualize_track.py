import json
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(log_path):
    timestamps = []
    cpu_percents = []
    mem_percents = []
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = datetime.fromtimestamp(entry['timestamp'])
                timestamps.append(ts)
                cpu_percents.append(float(entry['cpu_percent']))
                mem_percents.append(float(entry['mem_percent']))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid entry: {e}")
                continue
    
    return timestamps, cpu_percents, mem_percents

def generate_output_path(log_path):
    base_dir = os.path.dirname(log_path)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    return os.path.join(base_dir, f"{base_name}.png")

def plot_metrics(timestamps, cpu_data, mem_data, output_path):
    if not timestamps:
        print("No valid data to plot!")
        return
    
    # Calculate statistics
    cpu_min, cpu_max = min(cpu_data), max(cpu_data)
    mem_min, mem_max = min(mem_data), max(mem_data)
    time_deltas = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
    
    plt.figure(figsize=(12, 6))
    
    # Plot CPU with min/max annotation
    plt.plot(time_deltas, cpu_data, 
             label=f'CPU: {cpu_min:.1f}% - {cpu_max:.1f}%',
             color='tab:blue', marker='o', markersize=4, 
             linestyle='-', linewidth=1.5)
    
    # Plot Memory with min/max annotation
    plt.plot(time_deltas, mem_data, 
             label=f'Memory: {mem_min:.1f}% - {mem_max:.1f}%',
             color='tab:orange', marker='s', markersize=4,
             linestyle='--', linewidth=1.5)
    
    # Configure plot
    plt.title('Resource Usage Statistics\nMin-Max Range Display')
    plt.xlabel(f'Time Since {timestamps[0].strftime("%Y-%m-%d %H:%M:%S")}')
    plt.ylabel('Usage Percentage (%)')
    plt.grid(True, alpha=0.3)
    
    # Format time axis
    max_seconds = time_deltas[-1]
    ax = plt.gca()
    if max_seconds > 3600:
        ax.xaxis.set_major_formatter(
            lambda x, _: f"{int(x//3600):02d}:{int(x%3600//60):02d}")
        plt.xlabel('Elapsed Time (HH:MM)')
    else:
        ax.xaxis.set_major_formatter(
            lambda x, _: f"{int(x//60)}:{int(x%60):02d}")
        plt.xlabel('Elapsed Time (MM:SS)')
    
    # Enhance legend
    legend = plt.legend(
        title="Metric Ranges",
        loc='center right',
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize resource metrics with min/max values')
    parser.add_argument('log_path', type=str, help='Path to the JSON log file')
    args = parser.parse_args()
    
    try:
        output_path = generate_output_path(args.log_path)
        ts, cpu, mem = parse_log_file(args.log_path)
        plot_metrics(ts, cpu, mem, output_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.log_path}")
    except Exception as e:
        print(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()