import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_runtime_data(base_dir):
    # Dictionary to store list of runtimes for each GPU count
    # e.g., {1: [750.088, ...], 2: [386.738, ...]}
    gpu_data = {}
    
    # Pattern to match directories like 'gpus_1', 'gpus_2', 'gpus_4', 'gpus_7'
    dir_pattern = os.path.join(base_dir, 'gpus_*')
    
    # Regex to find the runtime value in the text files
    time_regex = re.compile(r'Run complete\.\s+Time:\s+([\d.]+)\s+s')

    for gpu_dir in glob.glob(dir_pattern):
        # Extract the number of GPUs from the directory name
        dir_name = os.path.basename(gpu_dir)
        try:
            num_gpus = int(dir_name.split('_')[1])
        except (IndexError, ValueError):
            continue
            
        gpu_data[num_gpus] = []
        
        # Read all output_*.txt files in this directory
        file_pattern = os.path.join(gpu_dir, 'output_*.txt')
        for file_path in glob.glob(file_pattern):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    match = time_regex.search(content)
                    if match:
                        runtime = float(match.group(1))
                        gpu_data[num_gpus].append(runtime)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
    return gpu_data

def calculate_metrics(gpu_data):
    # Calculate means
    mean_runtimes = {gpus: np.mean(times) for gpus, times in gpu_data.items() if times}
    
    # Sort by number of GPUs to keep things in order
    sorted_gpus = sorted(mean_runtimes.keys())
    
    if 1 not in mean_runtimes:
        raise ValueError("Error: Could not find 1 GPU data to calculate the scaling factor.")
        
    base_runtime = mean_runtimes[1]
    
    # Calculate scaling factors: (Runtime of 1 GPU) / (Runtime of N GPUs)
    scaling_factors = {gpus: base_runtime / mean_runtimes[gpus] for gpus in sorted_gpus}
    
    return sorted_gpus, mean_runtimes, scaling_factors

def plot_and_report(sorted_gpus, mean_runtimes, scaling_factors):
    # Print the text report
    print(f"{'GPUs':<10}{'Mean Runtime (s)':<20}{'Scaling Factor':<15}")
    print("-" * 45)
    for gpus in sorted_gpus:
        print(f"{gpus:<10}{mean_runtimes[gpus]:<20.3f}{scaling_factors[gpus]:<15.2f}x")
    
    # Generate the bar chart
    runtimes = [mean_runtimes[gpus] for gpus in sorted_gpus]
    labels = [f"{gpus} GPU(s)" for gpus in sorted_gpus]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, runtimes, color='lightgreen', edgecolor='black', width=0.6)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(runtimes)*0.01,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
                 
    plt.title('Mean Runtime vs. Number of GPUs', fontsize=14, fontweight='bold')
    plt.xlabel('GPU Configuration', fontsize=12)
    plt.ylabel('Mean Runtime (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart image and show
    plt.savefig('gpu_scaling_chart.png', dpi=300)
    print("\n[INFO] Bar chart saved as 'gpu_scaling_chart.png'")
    plt.show()

if __name__ == "__main__":
    # Use current directory '.' if script is placed inside 'scheduler/' 
    # Otherwise replace with your absolute path: '/home/nqr159/data/scheduler-test/...'
    BASE_DIR = '.' 
    
    print("Parsing dataset...")
    raw_data = parse_runtime_data(BASE_DIR)
    
    if not raw_data:
        print("No data found. Ensure you are running the script in the correct folder.")
    else:
        gpus, means, scaling = calculate_metrics(raw_data)
        plot_and_report(gpus, means, scaling)
