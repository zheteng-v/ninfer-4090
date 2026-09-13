# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [RTX 3090 Linux build](rtx-3090-linux.md) | Docker and native Ubuntu builds for the `sm_86` applications |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

The [model and weight decoupling charter](maintainer/model-weight-execution.md) starts from the
refactor's acceptance goal, maps responsibilities across conversion, artifacts, binding, execution,
resources, and product integration, and connects them with diagrams and end-to-end examples.
It preserves the agreed architectural constraints and guides further module design. Sharding and
custom chat templates are topics for that detailed design. This target charter does not claim
implemented container/runtime support or prescribe migration work packages.

The target [model contracts](maintainer/model-contracts.md) define common responsibilities for
fixed model code, a small set of instance parameters, and physical weight bindings. Extension
examples use Qwen4Exp and DeepSeek V4.1. The [Qwen3.5 contract](maintainer/qwen3_5-model-contracts.md)
separates fixed mathematics and derived values from persisted fields, with concrete Dense/MoE
parameters, optional components, and checked instances.
Both are target specification drafts; the current v2 container spec remains the delivered reference.

The temporary [refactor execution plan](maintainer/2026-09-13-model-weight-refactor-execution-plan.md)
defines the stages and shared execution requirements. The
[first-stage delivery record](maintainer/2026-09-13-model-weight-refactor-phase-1.md) records the
completed converter, v3 production, format renaming, offline upgrades, and validation. C++ v3 loading
and Engine integration remain for later stages, whose detailed plans follow their predecessors.
The references below describe the existing Engine and Op implementation.

Runtime and Op references:

- [Engine architecture, execution ownership, scheduling, and request lifecycles](maintainer/engine-architecture.md)
- [Resource scheduling, continuation/checkpoint, and Device/Host context-cache contracts](maintainer/resource-scheduling-and-context-cache.md)
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Operational logging channels, ownership, format, levels, and data policy](maintainer/logging.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)

`engine-architecture.md` is the sole top-level Engine architecture reference.
`resource-scheduling-and-context-cache.md` is its narrower authority for resource selection,
materialization, checkpoint ownership, and replica policy. The remaining files define physical
storage, model, artifact, Op, or measurement contracts rather than parallel architecture variants.

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.6-27B model semantics](maintainer/qwen3.6-27b-model.md)
- [Qwen3.6-27B artifact contracts, including NVFP4](maintainer/qwen3.6-27b-artifact.md)
- [Qwen3.8-27B DFlash2 mathematics and Engine state contract](maintainer/qwen3.8-27b-dflash2.md)
- [Qwen3.8-27B artifact contracts, including the NVFP4 target](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)
