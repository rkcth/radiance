# FLUX.3 Klein IQ8R qualification

The tested checkpoint is the distilled
[FLUX.3 Klein IQ8R](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-iq8r/tree/24ab485221274cb964618e466c42e7d2ae833941)
container at revision `24ab485221274cb964618e466c42e7d2ae833941`. Its SHA-256 is
`d1f10ea160e293a7715f16b8d5c4be3c10fbd00f4fccee8aa5e6bb0affd4a800`.
The test build used Radiance commit `5719c16a`.

## Standalone

The server loaded the model in 2.5 seconds on one Radeon AI PRO R9700
(`tp: 1`) and reported `resident` in its health response. Both image requests
used the default model step count and seed 42.

| Request | Canvas | Request to PNG |
| --- | --- | ---: |
| Text-to-image | 512 × 512 | 1.15 s |
| One-reference edit | 512 × 512 | 2.15 s |

The requests returned valid 512 × 512 PNGs. The edit used the generated image
as its reference. The card reached 11.54 GiB of VRAM in one-second samples.
These timings describe a ready server and one request at a time.

The older FLUX.3 block-FP8 container is rejected by this build's
`Flux3DistilledModularPipeline` plugin. Use the IQ8R container above for
this profile.

## Shared with Qwen Flash Next

The two-card test used the Swift 1.5 AutoRound checkpoint of Qwen Flash Next
as the text model, Qwen Image 2.1 IQ8R as the starting image model, and the
distilled Klein IQ8R checkpoint as an image reservation. This was a smoke
test with 2,048 tokens of text context and two text slots.

```yaml
tp: 2
tp-wire: exact
placement: expert_tiered
host-pool-mib: 40960
kv-cache-dtype: bf16
image-model: /models/qwen-image-2.1-uc-iq8r.rad
image-model-reserve: /models/flux-3-klein-iq8r.rad
image-snapshot-memory: pageable
image-model-cache-mib: 0
image-embedding-placement: vram
```

The host pool is the total across both ranks. This test used 40 GiB so the
text model fit with the image reservation fully recalled. Speculative
decoding used four draft tokens with an adaptive window.

| Request at 512 × 512 | Image activation | Request to PNG | Nonempty text deltas during request |
| --- | --- | ---: | ---: |
| Qwen Image generation | Active | 4.53 s | 10 |
| Switch to Klein and generate | Cold | 19.68 s | 26 |
| Klein one-reference edit | Active | 4.76 s | 19 |
| Switch back to Qwen Image and generate | Cold | 29.16 s | 18 |

The two seeded Qwen Image PNGs were byte-identical. The text stream
completed normally, and a fresh completion returned eight tokens after
the final switch. The last two columns describe this test's client
measurements. A stream event may carry multiple tokens.

One-second samples showed peaks of 31.57 and 31.61 GiB of VRAM and a
minimum of 108.22 GiB of available system RAM on a host with 184 GiB
visible. These figures cover the tested shared profile and canvas size.
