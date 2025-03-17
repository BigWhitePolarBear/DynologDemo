import argparse
import re
import subprocess

# Parse arguments
parser = argparse.ArgumentParser(description='GPU node trace script')
parser.add_argument('--nodelist', required=True, help='node list, support format: mcnode22, mcnode[22-23], mcnode[22,24]')
parser.add_argument('--job-id', required=True, help='job id to trace')
parser.add_argument('--output-dir', default='trace_out/trace.json', help='output directory')
parser.add_argument('--duration-ms', type=int, default=500, help='duration in ms')
parser.add_argument('--iterations', type=int, default=0, help='number of iterations, this will override duration-ms')
args = parser.parse_args()

# Node list parsing
def parse_nodelist(nodelist_str):
    # Handle range format mcnode[22-24]
    if match := re.match(r'^(\w+)\[(\d+)-(\d+)\]', nodelist_str):
        prefix, start, end = match.groups()
        nodes = [f"{prefix}{i}" for i in range(int(start), int(end)+1)]
        return nodes
    
    # Handle comma separated format mcnode[22,24]
    if match := re.match(r'^(\w+)\[(\d+(?:,\d+)*)\]', nodelist_str):
        prefix, nums = match.groups()
        nodes = [f"{prefix}{i}" for i in map(int, nums.split(','))]
        return nodes
    
    # Handle single node format mcnode22
    if re.match(r'^(\w+)(\d+)$', nodelist_str):
        return [nodelist_str]
    
    raise ValueError(f"Invalid node format: {nodelist_str}")

# Command execution
nodes = parse_nodelist(args.nodelist)
for node in nodes:
    while True:
        cmd = [
            'dyno', '--hostname', node,
            'gputrace', '--job-id', args.job_id,
            '--log-file', args.output_dir,
            '--duration-ms', str(args.duration_ms),
        ]
        if args.iterations > 0:
            cmd.extend(['--iterations', str(args.iterations)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            if "No processes were matched" in result.stdout.splitlines()[-1]:
                print(f"[Failed] {node}: No processes were matched, please check Job ID")
            else:
                print(f"[Success] {node}: Trace file is generating")
                break  # Success, break the loop
        else:
            print(f"[Error] {node}: Command execution failed")