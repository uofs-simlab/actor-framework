import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_file(filename, random_data, uniform_data):
    if not os.path.exists(filename):
        print(f"⚠️  File not found: {filename}")
        return
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines) - 5:
        line = lines[i]
        if ("Random Scaling" in line or "Uniform" in line) and \
           ("WITH scheduler" in line or "NO scheduler" in line) and \
           "actors=" in line:
            
            is_random = "Random Scaling" in line
            scheduler = "WITH scheduler" in line
            
            actors_m = re.search(r'actors=(\d+)', line)
            if not actors_m:
                i += 1
                continue
            actors = int(actors_m.group(1))
            
            N_val = None
            if not is_random:
                n_m = re.search(r'N=(\d+)', line)
                if not n_m:
                    i += 1
                    continue
                N_val = int(n_m.group(1))
            
            # === FIXED: take the SECOND block (real total runtime) ===
            # i+1 : spawn header
            # i+2 : spawn runtime
            # i+3 : total header (no "spawn")
            # i+4 : total runtime  ← this is what we want
            if ("SUPERVISOR TOTAL TIME" in lines[i + 3] and
                "spawn" not in lines[i + 3] and
                "Total runtime:" in lines[i + 4]):
                
                runtime_m = re.search(r'Total runtime:\s*([\d.]+)\s*s', lines[i + 4])
                if runtime_m:
                    runtime = float(runtime_m.group(1))
                    if is_random:
                        random_data[(scheduler, actors)].append(runtime)
                    else:
                        uniform_data[(N_val, scheduler, actors)].append(runtime)
            
            i += 5   # skip entire config block
            continue
        i += 1


# ====================== MAIN ======================
random_data = defaultdict(list)
uniform_data = defaultdict(list)

print("📂 Parsing 10 output files...")
for i in range(1, 11):
    fname = f"output{i}.txt"
    parse_file(fname, random_data, uniform_data)

print(f"✅ Parsed {len(random_data)} Random Scaling configs and {len(uniform_data)} Uniform configs.")

# --------------------- Random Scaling Graph (actors ≥ 30 000) ---------------------
if random_data:
    all_actors = sorted({act for (_, act) in random_data.keys() if act > 1})
    
    with_r, no_r = [], []
    for act in all_actors:
        key_w = (True, act)
        key_n = (False, act)
        avg_w = np.mean(random_data[key_w]) if key_w in random_data and random_data[key_w] else np.nan
        avg_n = np.mean(random_data[key_n]) if key_n in random_data and random_data[key_n] else np.nan
        with_r.append(avg_w)
        no_r.append(avg_n)
    
    plt.figure(figsize=(11, 7))
    plt.plot(all_actors, with_r, 'o-', linewidth=2.5, label='With Scheduler')
    plt.plot(all_actors, no_r, 's-', linewidth=2.5, label='No Scheduler')
    plt.xlabel('Number of Actors')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Heterogeneous Scaling On gpufarm5 — With Scheduler vs No Scheduler\n(actors ≥ 30 000, averaged over 10 runs)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('random_scaling_comparison.png', dpi=300)
    plt.close()
    print("📊 Saved: random_scaling_comparison.png")

# --------------------- Uniform Graphs ---------------------
if uniform_data:
    unique_ns = sorted({n for (n, _, _) in uniform_data.keys()})
    print(f"📈 Generating {len(unique_ns)} Uniform graphs...")
    for N in unique_ns:
        actors_for_n = sorted({act for (nn, _, act) in uniform_data if nn == N})
        with_r, no_r = [], []
        for act in actors_for_n:
            key_w = (N, True, act)
            key_n = (N, False, act)
            avg_w = np.mean(uniform_data[key_w]) if key_w in uniform_data and uniform_data[key_w] else np.nan
            avg_n = np.mean(uniform_data[key_n]) if key_n in uniform_data and uniform_data[key_n] else np.nan
            with_r.append(avg_w)
            no_r.append(avg_n)
        
        plt.figure(figsize=(12, 8))
        plt.plot(actors_for_n, with_r, 'o-', linewidth=2.5, label='With Scheduler')
        plt.plot(actors_for_n, no_r, 's-', linewidth=2.5, label='No Scheduler')
        plt.xlabel('Number of Actors')
        plt.ylabel('Average Runtime (seconds)')
        plt.title(f'Uniform Test On gpufarm5 — N = {N}\nWith Scheduler vs No Scheduler (10-run average)')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'uniform_N_{N}_comparison.png', dpi=300)
        plt.close()
        print(f"   → uniform_N_{N}_comparison.png")

# ====================== NEW: MEAN DIFFERENCE STATISTICS ======================
print("\n" + "="*80)
print("📊 MEAN DIFFERENCE (With Scheduler vs No Scheduler)")
print("="*80)

# --- Random Scaling ---
if random_data:
    print("\n🔹 Random Scaling (Heterogeneous, actors ≥ 30 000)")
    all_actors = sorted({act for (_, act) in random_data.keys() if act > 1})
    
    total_with, total_no = [], []
    
    print(f"{'Actors':>8} | {'With Scheduler':>15} | {'No Scheduler':>15} | {'Diff (s)':>12} | {'Diff (%)':>10}")
    print("-" * 78)
    
    for act in all_actors:
        key_w = (True, act)
        key_n = (False, act)
        avg_w = np.mean(random_data.get(key_w, [])) if random_data.get(key_w) else np.nan
        avg_n = np.mean(random_data.get(key_n, [])) if random_data.get(key_n) else np.nan
        diff_s = avg_w - avg_n
        diff_pct = (diff_s / avg_n * 100) if avg_n and not np.isnan(avg_n) else np.nan
        
        print(f"{act:8,} | {avg_w:15.3f} | {avg_n:15.3f} | {diff_s:12.3f} | {diff_pct:9.2f}%")
        
        total_with.extend(random_data.get(key_w, []))
        total_no.extend(random_data.get(key_n, []))
    
    # Overall mean for Random Scaling
    if total_with and total_no:
        mean_w = np.mean(total_with)
        mean_n = np.mean(total_no)
        overall_diff_s = mean_w - mean_n
        overall_diff_pct = (overall_diff_s / mean_n * 100) if mean_n else 0.0
        
        print("-" * 78)
        print(f"OVERALL (all {len(total_with)} runs):")
        print(f"  Mean With Scheduler : {mean_w:8.3f} s")
        print(f"  Mean No Scheduler   : {mean_n:8.3f} s")
        print(f"  Absolute difference : {overall_diff_s:8.3f} s")
        print(f"  Percentage difference : {overall_diff_pct:6.2f}%")

# --- Uniform (all N combined) ---
if uniform_data:
    print("\n🔹 Uniform Tests (ALL N combined)")
    total_with_u, total_no_u = [], []
    
    unique_ns = sorted({n for (n, _, _) in uniform_data.keys()})
    for N in unique_ns:
        actors_for_n = sorted({act for (nn, _, act) in uniform_data if nn == N})
        for act in actors_for_n:
            key_w = (N, True, act)
            key_n = (N, False, act)
            total_with_u.extend(uniform_data.get(key_w, []))
            total_no_u.extend(uniform_data.get(key_n, []))
    
    if total_with_u and total_no_u:
        mean_w_u = np.mean(total_with_u)
        mean_n_u = np.mean(total_no_u)
        overall_diff_s_u = mean_w_u - mean_n_u
        overall_diff_pct_u = (overall_diff_s_u / mean_n_u * 100) if mean_n_u else 0.0
        
        print(f"  Mean With Scheduler : {mean_w_u:8.3f} s")
        print(f"  Mean No Scheduler   : {mean_n_u:8.3f} s")
        print(f"  Absolute difference : {overall_diff_s_u:8.3f} s")
        print(f"  Percentage difference : {overall_diff_pct_u:6.2f}%")
    else:
        print("  (No uniform data found)")

print("\n🎉 Done! Graphs saved + mean differences printed above.")
