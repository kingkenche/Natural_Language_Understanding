"""
run_all.py
─────────────────────────────────────────────────────────────────────────────
Master runner: executes all tasks in sequence.

    python run_all.py              # runs all tasks (1-6)
    python run_all.py --task 2     # runs only task 2  (Gensim training)
    python run_all.py --task 5     # runs only task 5  (scratch training)
    python run_all.py --task 6     # runs only task 6  (comparison)

Task overview:
    1  Preprocessing & Dataset Statistics   (02_preprocess.py)
    2  Gensim Model Training                (03_train_models.py)
    3  Semantic Analysis                    (04_semantic_analysis.py)
    4  Visualization – PCA / t-SNE          (05_visualize.py)
    5  Word2Vec From Scratch (NumPy only)   (03b_train_scratch.py)
    6  Scratch vs Gensim Comparison         (06_compare_models.py)
─────────────────────────────────────────────────────────────────────────────
"""
import sys, importlib, time
from pathlib import Path

# Ensure local modules are discoverable
sys.path.insert(0, str(Path(__file__).parent))

TASKS = {
    1: ("02_preprocess",        "Preprocessing & Dataset Statistics"),
    2: ("03_train_models",      "Gensim Model Training"),
    3: ("04_semantic_analysis", "Semantic Analysis (Gensim)"),
    4: ("05_visualize",         "Visualization – PCA / t-SNE"),
    5: ("03b_train_scratch",    "Word2Vec From Scratch (NumPy only)"),
    6: ("06_compare_models",    "Scratch vs Gensim Comparison"),
}


def run_task(task_num: int):
    module_name, title = TASKS[task_num]
    print(f"\n{'#'*62}")
    print(f"#  TASK {task_num}: {title}")
    print(f"{'#'*62}")
    t0  = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"  ✓  Task {task_num} completed in {elapsed:.1f}s")


def main():
    if "--task" in sys.argv:
        idx       = sys.argv.index("--task") + 1
        task_nums = [int(sys.argv[idx])]
    else:
        task_nums = list(TASKS.keys())

    t_start = time.time()
    for n in task_nums:
        run_task(n)

    print(f"\n{'='*62}")
    print(f"  All done in {time.time()-t_start:.1f}s")
    print(f"  Outputs → outputs/")
    print(f"  Models  → models/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
