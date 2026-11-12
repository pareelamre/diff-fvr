import subprocess
import sys
import time
from pathlib import Path

scripts = [
    "experiments/exp1_complexity_scaling.py",
    "experiments/exp2_rotation_fidelity_360.py",
    "experiments/exp3_ablation_studies.py",
    "experiments/exp4_registration_monte_carlo.py",
    "experiments/exp5_energy_landscape.py"
]

def main():
    print("=" * 70)
    print("  LAUNCHING FULL A* PUBLICATION EXPERIMENTAL SUITE")
    print("=" * 70)
    t_start = time.time()

    base_dir = Path(__file__).resolve().parent
    py_exe = base_dir / ".venv" / "Scripts" / "python.exe"

    for idx, script in enumerate(scripts, 1):
        print(f"\n[{idx}/{len(scripts)}] Running: {script} ...")
        t0 = time.time()
        res = subprocess.run([str(py_exe), script], cwd=str(base_dir), capture_output=False)
        if res.returncode != 0:
            print(f"Error running {script}, exited with code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[{idx}/{len(scripts)}] Finished {script} in {time.time() - t0:.2f} s")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  ALL 5 EXPERIMENTS COMPLETED IN {total_time:.2f} s (~{total_time/60:.1f} min)")
    print("  Publication Figures & Tables Generated in: outputs/paper_figures/")
    print("=" * 70)

if __name__ == "__main__":
    main()
