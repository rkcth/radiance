# FLUX.2 Klein 9B compatibility

The target is Black Forest Labs' four-step, distilled
[FLUX.2 Klein 9B at 92196c8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/92196c8e11f7b6cf2b7493e037d8c5345c559216).
The gated model has its own non-commercial license and is a separate
port from Radiance's `flux3distilled` model.

| Component | Port assessment |
| --- | --- |
| Checkpoint | The generic Diffusers loader can collect `Flux2KleinPipeline` component tensors. It derives a separate architecture identifier, `flux2klein`, which this checkout has no plugin for. Keep FLUX.2 tensor names during the first BF16 parity tests. |
| Prompt | FLUX.2 Klein uses the 8B Qwen3 text encoder and three hidden-state taps at layers 9, 18, and 27 in Diffusers. FLUX.3 uses a different Qwen3-VL encoder and eight taps. Reuse the relevant Qwen3 block operations and compare token IDs, padding, and tap concatenation with the reference. |
| Denoiser | Implement the `Flux2Transformer2DModel` dual-stream and single-stream blocks and their reference-token layout. Validate the four-step schedule and a fixed-noise first prediction before quantization. FLUX.3's three-mode blocks implement a different graph. |
| VAE | FLUX.3 already implements an `AutoencoderKLFlux2` path. Compare the exact checkpoint's configuration, tensor names and running statistics, then extract and share those helpers if they match. Check reference-image encoding as well as the decode. |
| Serving | Reuse the image API and catalog. Qualify generation plus one- and two-reference edits with fixed latents. Measure the actual one-R9700 footprint before attempting Swift cohosting. The publisher reports about 29 GB VRAM for the reference setup. |

[Publisher model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) ·
[Diffusers component reference](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2#diffusers.Flux2KleinPipeline)
