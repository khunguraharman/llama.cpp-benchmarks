# Eagle3 draft memory estimates

For a dense attention cache with identical dimensions in every draft layer:

\[
M_{\mathrm{KV,draft}} = S C L_d H_{\mathrm{KV}}(d_K b_K + d_V b_V) \quad \text{bytes}.
\]

Here S is the number of separately allocated sequence caches, C is allocated
tokens per sequence (including implementation padding), L_d is draft attention
layers, H_KV is KV heads, d_K and d_V are head dimensions, and b_K and b_V are
bytes per element. For varying layers, sum the head/dimension expression over
layers. With a unified cache, use its total allocated token slots instead of S*C.
For quantized caches, use the implementation's block storage sizes (including
scales) rather than simply bit-width/8.

For FP16 K and V with the same head dimension d:

\[
M_{\mathrm{KV,draft}} = 4 S C L_d H_{\mathrm{KV}}d.
\]

The actual local Qwen3-4B-eagle3.gguf metadata gives L_d=1, H_KV=8,
d_K=d_V=128. With the launch script's C=8192, S=1 and FP16 defaults:

\[
M_{\mathrm{KV,draft}} = 8192\times1\times8\times128\times2\times2
=33,554,432\ \mathrm{bytes}=32\ \mathrm{MiB}.
\]

This is 4096 bytes (4 KiB) per allocated token. The three target extraction
layers [2,18,33] are feature sources, not three draft KV-cache layers. The draft
context covers the full sequence, not only n_predict speculative tokens. Draft
depth changes occupied temporary positions but does not multiply this fixed
8192-token allocation. No additional n_predict slots are assumed beyond C.

The target Qwen3-4B has a separate KV cache. Its 36 layers, 8 KV heads and
128-dimensional heads give 1152 MiB at the same FP16 8192-token capacity.
Target plus draft KV is therefore approximately 1184 MiB; the incremental draft
KV is 32 MiB. Target weights and target KV already exist in the n=0 baseline.

Draft weight estimate:

\[
M_{\mathrm{weights,draft}}\approx\sum_{t\in\mathrm{draft\ GGUF}}\mathrm{tensor\_bytes}(t)
=436,766,720\ \mathrm{bytes}=416.533203125\ \mathrm{MiB}.
\]

This sums actual mixed-type tensor storage, mostly BF16, rather than assuming
the draft uses the target's Q8_0 format. It includes the small d2t mapping and
F32 norms, excludes GGUF metadata, and does not count shared target embeddings
again. GPU allocation can differ slightly due to tensor placement and alignment.

Estimated incremental weights plus KV: 448.533203125 MiB. The measured n=0 to
n=1 peak-device-memory difference is approximately 2898-2922 MiB. The remaining
2449-2473 MiB is unattributed: changed compute/feature buffers, allocator effects,
or other device-wide memory usage may contribute. These peak differences do not
provide a direct measurement of individual runtime allocations.

The updated PNG/SVG has four panels, each with vertical brackets and horizontal
reference levels. Annotations use mean n=0 and n=1 peaks across OSL within each
panel; original measured curves retain separate OSL values. The companion CSV
reports each OSL independently. Weights and KV bands are estimated contributions
placed above the baseline, not directly measured absolute allocation addresses.

Regenerate only the memory plot with `python plot_eagle3_results.py --memory-only`.
The captured input dimensions, payload bytes and assumptions are in the root
`eagle3_memory_spec.json`; output provenance is in `eagle3_memory_methodology.json`.
Historical logs do not record all environment overrides; estimates assume the
current local model and launch defaults represent the benchmark.

Sources inspected on 2026-11-03:

- Local model: C:/Users/Harman/Documents/C++/llama.cpp/LLMs/Qwen3-4B-eagle3.gguf
- Local launch script: C:/Users/Harman/Documents/C++/llama.cpp/bench-eagle3-speed.bat
- Local source checkout: 147c3bf0734823f44ee76c3d595c4666d8719b75
- [Eagle3 model implementation](https://github.com/ggml-org/llama.cpp/blob/147c3bf0734823f44ee76c3d595c4666d8719b75/src/models/eagle3.cpp)
- [KV tensor allocation](https://github.com/ggml-org/llama.cpp/blob/147c3bf0734823f44ee76c3d595c4666d8719b75/src/llama-kv-cache.cpp)
- [Draft-context initialization](https://github.com/ggml-org/llama.cpp/blob/147c3bf0734823f44ee76c3d595c4666d8719b75/common/speculative.cpp)
