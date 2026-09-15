# NInfer Jinja source base

This directory is the source base for NInfer's maintained Jinja implementation. It was imported
from [llama.cpp](https://github.com/ggml-org/llama.cpp), commit
[`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
under the MIT license; see [LICENSE](LICENSE).

## Imported sources

- `jinja/`: the lexer, parser, runtime, value and string implementations and their headers from
  upstream `common/jinja/`, including `utils.h`.
- `support/unicode.*`: the UTF-8 helpers from upstream `common/unicode.*`.

The local `ninfer_jinja` static target builds these sources without llama, ggml, Python, the
upstream build system, or configure-time downloads. It is excluded from the default build until
the NInfer template frontend consumes it; it can be built explicitly.

## NInfer changes

- Removed the upstream `common_json` import functions and their declaration. NInfer's private
  template adapter will construct runtime values from the project's JSON representation.
  The language runtime retains its own value types and JSON output functions.
- Applied the repository's C++ formatting.
- Added the local explicit source target. No upstream tests, capability detectors, chat adapters
  or build-system files were imported.

This source base is not yet used to render product prompts. Unicode handling, value/default
semantics, JSON options and concurrent time formatting need correction before frontend integration.

NInfer may reorganize or replace this implementation to establish consistent language semantics
and clear ownership. Upstream fixes are adopted selectively; retaining its internal API or file
organization is not a maintenance requirement. Keep the origin and license when changing files.
