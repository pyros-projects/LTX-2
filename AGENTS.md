# AGENTS.md

This is the private working fork of `Lightricks/LTX-2`.

## Repo Identity

- Primary private project repo:
  - `/home/pyro/projects/private/LTX-2`
- GitHub fork:
  - `https://github.com/pyros-projects/LTX-2`
- `origin` should point to the fork.
- `upstream` should point to the official repo:
  - `https://github.com/Lightricks/LTX-2`

There is also an older local checkout at:

- `/home/pyro/repos/LTX-2`

Treat `/home/pyro/projects/private/LTX-2` as the main project repo unless the user explicitly says otherwise.

## Current Branch Context

The initial private-project integration snapshot was pushed on:

- branch: `feature/integrate-quantization-and-block-swap`
- commit: `d657670`
- commit message: `feat: integrate quantization experiments and block swap`

This snapshot intentionally carried over local project state from the older checkout, including:

- `.codex/`
- `.claude/`
- `openspec/`
- local/generated trainer configs
- local tests and design notes

`_data/` was intentionally excluded from git and added to `.gitignore` in this private repo.

## Quantization Notes To Remember

The three external quantization frameworks the user explicitly wanted to evaluate were:

- `TorchAO`
- `bitsandbytes` (`bnb`)
- `HQQ`

Do not “correct” this list to repo-native quantization paths unless the user is asking for currently implemented options. This list is specifically about framework candidates to evaluate.

Separate from that, the repo already contains and/or uses:

- `optimum-quanto` in the trainer
- native `ltx-core` FP8 paths such as `fp8_cast` and `fp8_scaled_mm`

## Block Swap Status

Single-GPU trainer-side transformer block swap was integrated in this private repo.

Key files:

- `packages/ltx-core/src/ltx_core/model/transformer/block_swap.py`
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-trainer/src/ltx_trainer/config.py`
- `packages/ltx-trainer/src/ltx_trainer/trainer.py`

Related archived OpenSpec change:

- `openspec/changes/archive/2026-03-23-add-ltx-block-swap/`

Main spec synced into:

- `openspec/specs/single-gpu-transformer-block-swap/spec.md`

## Useful Local Paths

Known local BF16 checkpoint path:

- `/home/pyro/models/comfy/diffusion_models/ltx-2.3-22b-dev.safetensors`

Known local Gemma path:

- `/home/pyro/models/gemma/gemma-3-12b-it-qat-q4_0-unquantized`

Ready-to-run smoke config for block swap:

- `packages/ltx-trainer/configs/generated/ltx2_image_lora_block_swap_smoke.local.yaml`

Notes about that config:

- image-only LoRA smoke setup
- validation disabled
- checkpoints disabled
- `blocks_to_swap: 8`
- user discussed switching it to `int8-quanto` and likely starting around `blocks_to_swap: 12` on a 24 GB 4090

Another generated local config exists for earlier quantization work:

- `packages/ltx-trainer/configs/generated/ltx2_image_lora_24gb_fp8.local.yaml`

## Practical Guidance For Future Work

- If the user talks about “the new project repo”, they likely mean:
  - `/home/pyro/projects/private/LTX-2`
- If they mention “the old repo” or “current repo we were in before”, they may mean:
  - `/home/pyro/repos/LTX-2`
- Before changing remotes or branches, verify which local checkout they mean.
- Be careful not to accidentally commit `_data/`.
- If preparing another machine, the user may want:
  - fork as `origin`
  - official repo as `upstream`

## Backup Note

During fork/transplant setup, a large working-tree backup tarball was created at:

- `/home/pyro/repos/_sandbox/ltx2-fork-prep-2026-03-24/LTX-2-working-tree-backup.tgz`

Do not rely on it being permanent without confirming it still exists.
