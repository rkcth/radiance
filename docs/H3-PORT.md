# MiniMax H3 with a secondary video engine

The hardware measurements below were taken with the earlier secondary-models release. Qualification on the native fleet is pending.

MiniMax H3 generates short MP4 clips with synchronized sound. The FL2VA
checkpoint supports text and first/last-frame workflows. The Ref2VA checkpoint
uses subject references. Both Radiance packages include their text/vision
encoder and audio/video components.

| Package | Download | Size |
| --- | --- | --- |
| FL2VA W4A8 | [Hugging Face](https://huggingface.co/rkcth/MiniMax-H3-FL2VA-Radiance-W4A8) | 53.31 GiB |
| Ref2VA W4A8 | [Hugging Face](https://huggingface.co/rkcth/MiniMax-H3-Ref2VA-Radiance-W4A8) | 53.31 GiB |

The conversions use the official MiniMax H3 revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. FL2VA takes its denoiser from
`transformer/`, and Ref2VA from `transformer_ref/`. The main denoiser projections
use rotated four-bit weights with INT8 activations. Other components retain
INT8, BF16 or FP32 as specified in
[the conversion recipe](../data/recipes/minimax-h3-w4a8.recipe).
Each package includes its source attribution, complete license and checksums.

## Configure the fleet

Build the executable and bundled plugins from the same revision using the
[build instructions](../README.md#build-from-source). Set `-DRAD_WITH_FFMPEG=ON` when
configuring CMake so that missing video dependencies fail the build.

Download the models, verify `SHA256SUMS`, and adjust paths in
[the H3 fleet example](../deploy/compose/flashnext-h3.radiance.yaml).
Run `radiance --config FILE --check` before starting it.
The example binds to localhost on port 8000. Configure authentication before
exposing a listener to other machines.
The primary checkpoint path `swift15-flashnext-autoround.rad` refers to the
locally supplied Qwen conversion used in qualification. Supply your own
compatible Qwen Flash Next checkpoint and adjust that path. The H3 packages
contain the secondary models and their encoders.

The shared profile was tested on two 32 GiB Radeon AI PRO R9700s (`gfx1201`)
with about 184 GiB of host RAM visible to Linux. Qwen Flash Next used a
200,000-token per-request limit, eight slots, and a 16 GiB RAM prefix cache.
One-card operation and smaller host-memory configurations need qualification.
Host-memory needs depend on the primary model, context, and retained secondary
models. Cold loading and automatic memory recovery can delay Qwen requests
by tens of seconds.

In an existing fleet, add the H3 models under `models` and put them in the same
exclusive group as the other secondary generators. This lets each request
select its model while the primary text model stays resident.

## Animate a saved Krea image

Set `RADIANCE_URL` to the server base URL and `RADIANCE_API_KEY` to its key.
Use the FL2VA model for a first frame.

```bash
curl --fail-with-body --max-time 600 "$RADIANCE_URL/v1/videos" \
  -H "Authorization: Bearer $RADIANCE_API_KEY" \
  -F model=minimax-h3-fl2va -F task=i2va \
  -F first_frame=@krea-output.png -F first_frame_mode=center \
  -F 'prompt=A wooden sailboat rocks on a pond. Water laps and birds sing.' \
  -F size=864x480 -F seconds=5 -F num_inference_steps=50 -F seed=42
```

The POST can wait for model loading and queue admission before returning a job
ID. Poll `GET /v1/videos/{id}` until its status is `completed` or `failed`.
Download successful jobs from `GET /v1/videos/{id}/content`. Keep the job ID
with the original request to associate the result with its prompt and input.

The [Krea-to-H3 client](../scripts/krea2_to_h3.py) handles submission, polling,
and download. It accepts an existing still with `--source`, or generates one
with `--image-prompt`. Use `--video-model minimax-h3-fl2va` for first-frame
animation. See `--help` for steps, seed, image LoRAs and video LoRAs.

```bash
python3 scripts/krea2_to_h3.py --url "$RADIANCE_URL/v1" \
  --source krea-output.png --video-model minimax-h3-fl2va \
  --prompt 'The boat rocks gently. Water laps and birds sing.' \
  --video-steps 50 --output clip.mp4
```

For a subject-reference request, select `minimax-h3-ref2va`, send `task=ref2va`
and upload `ref_image_1`. Refer to it as `<Picture 1>` in the prompt.
Additional image references are numbered consecutively. They consume the
same processing budget as the prompt and clip. See the
[video API guide](GUIDE.md#516-video-v1videos) for keyframe and reference fields.
With the helper, select `--video-model minimax-h3-ref2va --mode reference`.

## Size, duration and quality

The example's `image-gen-max-pixels: 414720` limits output area to 414,720
pixels. Both dimensions must be multiples of 32, at least 256, and within a
4:1 aspect ratio. The tested output is 864×480. Portrait and square requests
within those bounds pass size validation, but have not received the same
visual qualification.

`max-num-batched-tokens: 27000` also limits clip length and reference capacity.
At this output cap, the tested model advertised approximately 5.17–5.88 seconds.
The duration limit applies to smaller canvases as well. A five-second request
produces 124 frames at 24 fps, about 5.17 seconds, with 32 kHz stereo audio.
Larger areas, longer clips and more references need additional memory planning.

`num_inference_steps` counts schedule points. A value of 50 produces 49
denoiser evaluations. Shorter schedules change quality and should be evaluated
for the chosen checkpoint and workload. The official checkpoints retain their
original denoising schedule. These packages have no merged LoRAs.

## Qualification and remaining limits

The official FL2VA W4A8 package completed a first-frame 864×480 clip alongside
Qwen on both R9700s using the shared 27,000-row profile. It held approximately
14.94 GiB per card. Qwen stayed resident and completed its concurrent requests.

An isolated comparison used a saved Krea mug image, seed 27, 15 Euler
evaluations and the same 864×480 canvas for INT8 and W4A8. Both outputs had
124 frames and stereo sound. Warm request-to-MP4 time was 65.1 seconds for
INT8 and 63.1 seconds for W4A8. Decoded video PSNR between the two outputs was
32.24 dB. This single-subject comparison provides limited quality evidence.

The official Ref2VA W4A8 package also completed 864×480 image-reference requests
alongside Qwen. A cold request took 128.8 seconds, including 83.7 seconds of
loading. The warm repeat took 45.6 seconds and produced the same MP4 SHA-256
at seed 27. These timings describe the measured profiles and cache state.

Image-reference generation and the separate reduced-resolution reference-video
path work. Full-resolution, timeline-aligned character-swap guide-video input
remains unfinished. Gesture and speech transfer need further validation.
Automatic offload and expert refill can continue after the clip is delivered,
so primary text throughput may recover gradually.

The matching release passed focused encoder, temporal tiling, sparse attention,
and LoRA device checks on both cards. CPU LoRA checks and the video API suite
also passed, with the existing H.264 round-trip fixture excluded. That fixture's
aligned-frame allocation failure was reproduced on the earlier baseline.
Actual MP4 generation and download passed on the deployed runtime.

MiniMax H3's Community License includes territorial and use restrictions.
Read the complete license in the model package and obtain any separate
authorization your use requires. Qwen encoder components retain their Apache
2.0 license.
