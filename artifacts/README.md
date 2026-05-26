# Artifacts

This directory is for local generated outputs and checkpoints. Contents are ignored by git except this README.

Expected layout:

```text
artifacts/
  stage1/
    full_model/
    trajectory_value_head.bin
  stage2/
    math_shepherd_steps-sprm-labels.jsonl
    cti_cache/
  stage3/
    sprm_prm_model/
      adapter_config.json
      adapter_model.safetensors
      dual_v_heads.safetensors
      tokenizer files...
  eval/
    processbench/
    prmbench/
```

Publish model weights through a model repository, not through the source repository.
