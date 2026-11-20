# W6NL routed experts: candidate implementation

`w6nl64a8h` is a six-bit scalar codebook for the routed experts' down projection. It is a candidate format, not a claim that six-bit down improves task quality. `data/recipes/qwen4exp-u4-w6nl-bf16ple.recipe` pairs it with U4 trellis gate/up and preserves the existing U5 trunk, protected BF16 experts, BF16 PLE/vision and draft recipe.

## Format

The canonical weight is an affine `u6` code plane with a 64-entry `f32` table, `fp8_e4m3` scales per row per 64 columns and one `f32` scale above them, with the `fwht128` transform. The first 32 table entries are positive E4M3-exact magnitudes and the second 32 their negatives. Bit 5 is sign; bits 0–4 select the magnitude. The fixed upper scale is `2^-13`; the quantizer folds the unnormalised Hadamard's `1/128` into the weight scale.

The container's six-bit codes are packed LSB-first. At load, `libr4d` arranges each row as the permuted low nibbles followed by two bit planes (the high magnitude bit and sign). Both arrangements occupy `6*K/8` code bytes per row. The kernel expands the two 16-entry halves of the magnitude table to E4M3, selects between them and merges sign before the existing FP8 WMMA. The loader checks the entire table, the fixed upper scale, transform and shape; same-width encodings with different semantics are refused.

A trellis gate reads rotated INT8 activations; W6NL down reads a separately quantized rotated E4M3 gated intermediate, like W5NL. The architecture must select that producer and the W6 kernel for the down projection. The BF16 protected experts retain their separate path.

## Codebook pilot

The candidate table is fitted, not derived from the evaluation prompts. `tools/fit_w6nl.py /path/to/original-bf16-checkpoint` reads only selected BF16 safetensor rows; it does not load or serve a model. It samples five experts per layer, 128 rows per expert, and FWHT-rotates weights in groups of 128 before normalising each group of 64 by its absolute maximum. Fit layers are 0, 8, 16, 24, 32, 40 and 47 (2,867,200 weights); separate check layers are 4, 12, 20, 28, 36 and 44 (2,457,600 weights). The script fits a 32-magnitude symmetric grid and chooses distinct E4M3-representable values.

On that separate weight sample, normalised scalar MSE with one globally optimized scale per table was approximately 0.000305 for W5NL and 0.000103 for this W6NL candidate. This is **not** a GPTQ, activation-weighted, KL or task result. It does not establish model quality or prove the candidate's table is optimal.

A separate 41,600-group sample had candidate pre-rounded group scales between 0.136 and 1.160 in E4M3 units under the fixed upper scale, with none subnormal or above the format's maximum. This does not cover every expert; check saturation and fallback reports during actual conversion.

## Checks and limits

The host tests cover W6NL GPTQ on a synthetic BF16-domain Gram, all 64 code values, exact table/scale checks, packing and reference math. The synthetic GPU tests exercise all 64 codes, sorted and token-order inputs, two/four classes and decode/prefill forms; they do not load a model. The U4/W5, U4/W6, U5/W5 and U5/W6 graph declarations are checked for both TP ranks.

A `rad-convert --plan-only` pass over the BF16 checkpoint declared 51,489 weights and estimated about 170.26 GiB for the candidate recipe. The plan used placeholder calibration paths and wrote no model: it validates neither the downloaded Grams nor conversion.

For a synthetic TP2-shaped down-projection latency comparison, run `r4d_selftest /path/to/libr4d.so --perfw6nl M [iters] [experts] [top_k]`. It compares W5NL and W6NL using the same routing and resident, synthetic weights, with automatic and pinned launch forms; byte-parity checks compare launch forms **within** each format, not W5 values to W6 values. Repeated synthetic kernel timing is not end-to-end model throughput.

These tests are not an end-to-end model validation. No full W6 model has been converted or scored; mixed-model inference, model throughput, placement, and speculative parity require separate validation. The main calibration directory must be reassembled and verified before conversion, with the separate draft Gram directory supplied as `MTP_CALIB`.
