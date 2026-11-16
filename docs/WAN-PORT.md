# Wan 2.2 TI2V-5B port

The first target is the single-transformer TI2V-5B model. Both text and
first-frame conditioning go through `POST /v1/videos`. The result is an
MP4 with a video track. Soundtrack generation is optional in the existing
video driver.

Use the [Diffusers checkpoint at b8fff731](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/tree/b8fff7315c768468a5333511427288870b2e9635)
for tensor names and configuration, and the [Wan implementation at 1ea34ff4](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff)
for inference behavior. Record the exact reference implementation and
options with each parity fixture. Compare inputs and intermediate tensors
before judging final MP4 frames, since container encoding and noise
generators can differ.

The builder and video driver now support video-only output. The Linux API
test completed a silent MP4 and rejected soundtrack parameters. The existing
H.264 first-frame test fails under the test container's FFmpeg 6.1 on both
the release baseline and this branch: its PNG decoder rejects coded width
320 when the configured pixel budget is exactly 301 × 171.

The pinned tokenizer matched Radiance on 64 token IDs across five English,
French, Chinese and Japanese prompts, before the end-of-sequence token.
The reference adds one EOS token. The patch geometry tests cover width-first
position order, first-frame timestep zeroing, and the different column order
of the convolution and output head. The native pipeline still needs to use
these helpers.

The UniPC BH2 helper matched all model timesteps and float32 sigmas from
the pinned Wan solver in eight CPU comparisons. The 50-step default shift
of 5 had a maximum scalar latent error of `9e-8`. Across 1 to 1000 steps
and shifts from 0.02 to 100, the maximum was `2.51e-6`. Both paths started
at step index zero. This matters when consecutive sigmas truncate to the
same integer timestep. The test covers solver arithmetic, before it is
connected to the transformer and device latent state.

The host and gfx1201 `flow_unipc_step` operations now apply those
coefficients to latent and history buffers. The host test checks numerical
results and rejects overlapping rows. On one R9700, three device updates
each with F32 and BF16 velocity matched the host arithmetic exactly,
including untouched tensor padding. The reference and device passes share
the elementwise update. The video driver now accepts a video-only UniPC
declaration. It stages the integer model timestep and eight coefficients
for each pass. The HTTP API selects and reports `unipc` for that declaration
and rejects alternate samplers. The Wan graph still needs to bind these
operands and supply its predicted velocity.

The verified Diffusers files store the text encoder in BF16 and the
transformer and VAE in FP32. The conversion recipe keeps the timestep
projections, normalization gains, modulation tables, and velocity head
in FP32. It casts the denoiser's other projections and the VAE weights
to BF16. The recipe matches declared weight names. The denoiser probe
checked each declared dtype against these rules. `rad-convert` still
requires the Wan architecture's declaration to emit a runnable container.

Wan's UMT5 relative-position attention now runs in libref and on gfx1201.
A seeded BF16 comparison with pinned Wan's PyTorch expression used 64 heads
and packed sequences of 127 and 511 tokens. The two R9700s produced identical
outputs. Against PyTorch, GPU RMSE was `5.79e-6` and host RMSE was `8.13e-6`.
Both paths had a maximum absolute error of `0.001953125`. The GPU regression
also covers a full 512-token sequence, direction-specific bias, and untouched
padding in pitched buffers. These checks precede the full encoder graph.

The conditioning graph declares all 24 UMT5 blocks and maps them to the
242 BF16 source tensors. The names, dimensions, and dtype were checked
against the three downloaded safetensors headers. The recording test
checks the per-layer relative bias, QKV slice offsets, and buffer lifetimes
with a positive and a negative prompt in the same pass. UMT5 rounds the
normalized rows before applying the learned gain. Its separate GELU op
also preserves the intermediate BF16 rounding in Wan's eager expression.

A GPU probe ran all 315 recorded operations with real BF16 weights and
two packed prompts of 13 and 14 tokens including EOS. On one R9700, the
27 × 4096 final embedding values were compared with the pinned Wan
encoder running each prompt separately on PyTorch's CPU backend. The
minimum per-token cosine similarity was `0.995916`. Relative RMSE across
all values was `2.53%` and maximum absolute difference was `0.03125`.
The text conditioning then right-pads each encoded prompt to 512 rows
before the biased projections. The projected padding participates in
cross-attention. On one R9700, the full 1024 × 3072 output from two prompts
had `0.028%` relative RMSE against the BF16 PyTorch projections with
identical encoder input. The projected padding had a maximum absolute
difference of `4.77e-7`.

The timestep graph computes unscaled integer-timestep sinusoids in FP64
before the FP32 embedding and six-way modulation. Image-to-video gets a
separate timestep-zero row for its first temporal patch plane. With the
real FP32 weights and timesteps 0 and 999, the R9700's time embedding and
modulation had maximum absolute differences of `9.54e-7` and `3.34e-6`
against the pinned PyTorch reference. The DiT still needs to consume the
projected prompt and timestep buffers in a video pass.

The denoiser now packs the F32 latent into BF16 patch rows and uses the
checkpoint's Conv3d weights in a biased GEMM. With the real weights on one
R9700, the projection had relative RMSE `2.59e-5` against PyTorch BF16 Conv3d.
The head output uses a different column order and is unpacked as F32
voxel velocity. The CPU and device fixtures cover that order across frames.

The rotary table splits 128-wide heads into 22 time pairs and 21 pairs
each for height and width. It uses FP64 angles and trigonometry, with
FP32 storage and rotation in the BF16 profile. The pinned reference
instead keeps FP64 through the rotation and narrows to FP32 and then BF16
at its attention boundary. A dense 24-head comparison used four grids,
including coordinate 1023 on each axis. The R9700 matched 18,947,937 of
18,948,096 BF16 outputs exactly, with maximum absolute error `0.0078125`.

The denoiser graph now issues all 30 self-attention, cross-attention,
and feed-forward blocks and the F32 velocity head. Residual and
modulation arithmetic stays in F32. Q and K normalization rounds to
BF16 before applying the F32 gain. Self-attention keeps the resulting
queries and keys in F32 through rotary and narrows them for attention.
With the real first-layer gains and dense inputs, that FP32 rotary path
matched 18,947,923 of 18,948,096 reference values after BF16 narrowing.
The maximum absolute difference was `0.03125`.

A device probe staged 821 source tensors on one R9700 and compared a
12-patch denoising pass at timestep 999 with the pinned Wan block and
head implementation. The PyTorch oracle used CPU scaled dot-product
attention with BF16 inputs and the reference's output dtype. Both ran
on the same latent and projected prompts.

| Conditioning | Velocity relative RMSE | Velocity cosine similarity |
| --- | ---: | ---: |
| Text only | 0.75% | 0.999973 |
| First temporal patch plane at timestep zero | 1.89% | 0.999823 |

This compares the denoiser before sampler updates or VAE decoding. The
second case uses the image-to-video time split with the test latent.
The complete video path still needs VAE encoding and decoding, guidance,
and latent updates in the serving architecture.

The VAE temporal operators now carry two causal frames per layer and
request lane. A temporal upsampler bypasses its convolution on the first
chunk. Its first convolution starts with zero history, and later chunks
reuse the cached tail. The shortcut uses the reference's channel
repetition and drops one expanded temporal plane on the first chunk.

With the pinned BF16-cast VAE weights on an R9700, the decoder input
convolution matched 184,294 of 184,320 output values exactly
(`2.82e-5` relative RMSE). The temporal convolution and shuffle matched
322,500 of 322,560 values (`3.75e-5` relative RMSE). The shortcut matched
the reference exactly with and without the first-chunk crop. The probe
used the registered 3D convolution weight relayout and ran ten chunks
through two interleaved lanes, including a new sequence in a reused lane.

The first-frame encoder reuses the image VAE graph. Causal convolutions
take the last temporal tap, spatial resamplers and attention projections
read their native 2D weights, and the first-frame shortcut includes left
zero padding on temporal downsample stages. The encoder opts into rounding
the L2 norm to BF16 before division, as in the pinned implementation.

Using a 64 × 96 input and the pinned BF16-cast weights, all 68 encoder ops
ran on one R9700. The latent mean matched the Diffusers encoder with 0.392%
relative RMSE and 0.999992 cosine similarity. The comparison is before
per-channel latent normalization. The same device passed the independent
640- and 1024-wide per-frame attention checks against the CPU reference.

The full decoder runs the post-quant projection, 14 residual blocks,
frame-local middle attention, three spatial upsample stages, and two
temporal upsample stages. It emits one frame from the first latent
and four from each later latent. At 32 × 48 output, three consecutive
latent chunks on one R9700 matched the pinned BF16-cast Diffusers
decoder with relative RMSE of 0.577%, 0.760%, and 0.552%. The RGB
packer widened the decoded patches to FP32 and matched reference
unpatching and normalization byte for byte on the same device patches.

1. Use the existing Diffusers component loader to retain component prefixes.
   Verify the transformer, UMT5 encoder, tokenizer, VAE, and sampler metadata
   against the pinned model. The transformer uses 30 blocks, width 3072,
   24 heads and 48 latent channels. The tokenizer is Unigram. Check its
   normalized text and EOS placement against the reference before testing
   embeddings. The UMT5 encoder uses a learned relative-position bias
   table in each block, gated GELU, and 24 blocks.
2. Implement and compare the transformer and VAE in BF16. Test patch
   order, three-axis rotary positions, text cross-attention and timestep
   modulation separately. The VAE scales time by 4 and space by 16. Its
   first frame and later causal chunks use different padding. Check both
   frame counts and pixels across a chunk boundary before using longer clips.
3. Match the pinned UniPC flow schedule and classifier-free guidance.
   In image-to-video, encode the first frame, use timestep zero for its
   latent tokens, and restore that latent after each sampler step. With
   a fixed latent and text embedding, compare the first transformer
   prediction, individual solver steps, and the final decoded frames.
4. Qualify a usable short clip on a single R9700, then a shared
   configuration with Qwen Flash Next. Record cold start, generation
   latency, peak VRAM and host RAM, and gaps in text output. If full BF16
   weights and activations exceed the target budget, keep the BF16
   intermediate checks and qualify quantization against those results.
