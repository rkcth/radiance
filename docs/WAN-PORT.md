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
the elementwise update. The Wan graph still needs to stage the coefficients
and supply its predicted velocity.

The verified Diffusers files store the text encoder in BF16 and the
transformer and VAE in FP32. The BF16 conversion recipe is limited to the
latter two components. `rad-convert` requires the Wan architecture's
declaration before emitting a runnable container.

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
