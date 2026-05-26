# Math-Shepherd Data

Place raw Math-Shepherd-style jsonl files under `raw/`.

The data collection command should convert raw trajectories into:

```text
steps/math_shepherd_steps.jsonl
steps/*.hidden_states.f16.bin
```

The step jsonl should contain `final_reward` and `steps[].hidden_state_ref` or `steps[].hidden_state_refs`.

Example:

```bash
python -m sprm.data.collect_hidden_states \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --input-jsonl data/math_shepherd/raw/math_shepherd.jsonl \
  --output-jsonl data/math_shepherd/steps/math_shepherd_steps.jsonl \
  --hidden-state-dir data/math_shepherd/steps \
  --text-field label \
  --question-field input \
  --reward-from-suffix \
  --overwrite
```
