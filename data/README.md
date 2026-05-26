# Data Layout

This directory stores dataset instructions and local generated data. Large files are ignored by git.

Expected local layout:

```text
data/
  math_shepherd/
    README.md
    raw/      # raw Math-Shepherd trajectories, ignored
    steps/    # collected hidden states and step jsonl, ignored
  processbench/
    README.md
    raw/      # ProcessBench files, ignored
  prmbench/
    README.md
```

SPRM uses trajectory text and terminal `final_reward`. Intermediate labels from source datasets are not the main SPRM supervision.
