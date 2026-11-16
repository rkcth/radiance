## Shared text, image and CPU transcription endpoint

Run the existing text/image profile and CPU transcription in separate Radiance
processes. Radiance selects its device backend once per process. For the ASR
process, build a CPU-only Radiance binary or launch its container without GPU
devices. Use a machine with enough RAM for both processes, including the
primary process's pinned memory and image cache.

1. Bind the text/image listener to `127.0.0.1:18000` with its existing model and
   GPU options. Start the ASR process on the same host and network namespace:

   ```yaml
   server:
     host: 127.0.0.1
     port: 18001
     api-key-env: RADIANCE_API_KEY
     trusted-proxy: 127.0.0.1/32
   models:
     qwen3-asr:
       model: /models/Qwen3-ASR-1.7B-hf
   ```

   Start the CPU process with `radiance --config asr.yaml`. Set the same
   nonempty `RADIANCE_API_KEY` environment variable and `api-key-env`
   setting for the primary server. Both listeners authenticate their own
   requests. Keep their CORS settings consistent. Set `trusted-proxy` on
   both listeners if you want the original client IP in their logs.
2. Include `deploy/nginx/radiance-shared.conf` in Nginx's `http` block, then run
   `nginx -t` before starting or reloading Nginx. The example binds the proxy to
   loopback on port 8000. To share it remotely, configure TLS on the client-facing
   listener and keep the upstream listeners private.
3. Point clients at the proxy. Text, images, `/v1/models`, and `/health` reach
   the primary process. `/v1/audio/transcriptions`, its `/audio/transcriptions`
   alias, and the `/v1/realtime` WebSocket reach ASR. Model IDs for transcription
   come from the ASR process. Probe its `/health` on port 18001 separately.

The proxy forwards uploads and streamed responses without buffering their
entire contents. The body limit includes multipart and base64 overhead. Adjust
its idle timeouts if a cold model switch can leave the primary process silent
for more than ten minutes. See Nginx's [proxy module reference](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)
and [WebSocket guide](https://nginx.org/en/docs/http/websocket.html).
