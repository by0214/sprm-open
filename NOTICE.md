# Notice

This project will include code adapted from the development repository for the SPRM paper.

The dual value-head wrapper should acknowledge its relationship to the TRL value-head wrapper design. The released implementation should document local changes clearly:

- dual scalar value heads;
- PEFT/LoRA adapter loading;
- `dual_v_heads.safetensors` save/load support;
- `<ST_END>` step-boundary scoring.

Datasets and benchmarks should be referenced rather than vendored:

- Math-Shepherd for outcome-labeled mathematical reasoning trajectories;
- ProcessBench for process error localization evaluation;
- PRMBench for PRM benchmark evaluation;
- Hugging Face Transformers, TRL, PEFT, datasets, and safetensors as dependencies.
