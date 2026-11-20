#!/usr/bin/env python3
"""Model-gated ordinary AQL Vision/DFlash check; default only prepares YAML/fixtures.

Requires an ABI-matched current radiance binary/home and a user-supplied compatible
Gemma 4 MoE Vision+DFlash RAD. No weights or dependencies are downloaded. --execute
starts depth-zero then depth-15 servers sequentially. Use /tmp for output/scratch,
then preserve evidence elsewhere. This is a correctness test, not a benchmark.
"""
import argparse
import base64
import copy
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zlib


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def png(shape, color):
    def chunk(name, data):
        return (struct.pack('>I', len(data)) + name + data +
                struct.pack('>I', zlib.crc32(name + data) & 0xffffffff))
    rows = []
    for y in range(512):
        row = bytearray([0])
        for x in range(512):
            inside = ((x - 256) ** 2 + (y - 256) ** 2 <= 160 ** 2
                      if shape == 'circle' else abs(x - 256) <= 160 and abs(y - 256) <= 160)
            row.extend(color if inside else (255, 255, 255))
        rows.append(row)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 512, 512, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite returned numeric value'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def chosen(response):
    tokens = response['choices'][0]['logprobs']['content']
    assert tokens, 'Chosen-token IDs/logprobs are required'
    return [t['id'] for t in tokens], [t['logprob'] for t in tokens]


def counters(text):
    result = {}
    for name in ['radiance:draft_tokens_total', 'radiance:draft_accepted_total']:
        result[name] = sum(float(x) for x in re.findall(
            r'^' + re.escape(name) + r'(?:\{[^\n]*\})? ([^\n]+)$', text, re.M))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--home', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kv-dtype', choices=['bf16', 'i8'], default='i8')
    ap.add_argument('--port', type=int, default=18400)
    ap.add_argument('--gpu-uuids', help='Two comma-separated discrete GPU UUIDs')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    model = args.model.resolve()
    before = model.stat()
    fixtures = {}
    for key, shape, color in [('A', 'circle', (255, 0, 0)), ('B', 'square', (0, 0, 255))]:
        raw = png(shape, color)
        (args.output / (key + '.png')).write_bytes(raw)
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}}
        fixtures[key] = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What color and geometric shape is shown in the image?'}, image]},
            {'role': 'assistant', 'content': 'I can help with the image.'},
            {'role': 'user', 'content': '\n'.join(f'Record {i}: neutral context label.' for i in range(120)) +
             '\nUsing the image above, what color and geometric shape is shown?'}]
    fixtures['text'] = [{'role': 'user', 'content': 'What is 2+2? Answer with only the number.'}]
    sequence = [('text', 'text'), ('A', 'A'), ('A-repeat', 'A'), ('text-after-A', 'text'),
                ('B', 'B'), ('text-after-B', 'text'), ('B-repeat', 'B'), ('A-after-B', 'A'),
                ('A-final', 'A'), ('A-after-reset', 'A')]
    results = {}
    for depth in [0, 15]:
        sd = args.output / ('depth' + str(depth))
        sd.mkdir()
        (sd / 'scratch').mkdir()
        port = args.port + int(depth > 0)
        config = ('server:\n  host: 127.0.0.1\n  port: ' + str(port) + '\n  log: debug\n' +
                  '  radiance-home: ' + json.dumps(str(args.home.resolve())) + '\n' +
                  'memory:\n  gpu-headroom-mib: 2048\nmodels:\n  vision-check:\n' +
                  '    model: ' + json.dumps(str(model)) + '\n' +
                  '    tp: 2\n    tp-wire: exact\n    p2p: on\n    max-model-len: 8192\n' +
                  '    max-num-seqs: 1\n    max-num-batched-tokens: 512\n' +
                  '    kv-cache-dtype: ' + args.kv_dtype + '\n    num-speculative-tokens: ' + str(depth) + '\n' +
                  '    draft-window: fixed\n    draft-short-from: 0\n    draft-coupling: off\n' +
                  # The fixture is shorter than the default 2048-token window checkpoint.
                  '    checkpoint-interval: 1024\n' +
                  '    mm-max-patches: 2520\n    image-tokens: 280\n    prefix-cache-mib: 256\n' +
                  '    deterministic: true\n')
        cfg = sd / 'radiance.yaml'
        cfg.write_text(config)
        command = [str(args.binary.resolve()), '--config', str(cfg)]
        save(sd / 'command.json', command)
        if not args.execute:
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith(('RADIANCE_', 'ROCR_', 'HSA_'))
               and k not in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL']}
        mask = args.gpu_uuids or os.environ.get('ROCR_VISIBLE_DEVICES', '')
        assert len(mask.split(',')) == 2 and all(x.startswith('GPU-') for x in mask.split(',')), 'Select two GPU UUIDs'
        env.update(ROCR_VISIBLE_DEVICES=mask, TMPDIR=str(sd / 'scratch'),
                   XDG_CACHE_HOME=str(sd / 'scratch'), PYTHONDONTWRITEBYTECODE='1',
                   RADIANCE_LOG_STEPS='1', OMP_NUM_THREADS='4', OMP_WAIT_POLICY='passive')
        save(sd / 'runtime-environment.json', {k: env[k] for k in ['ROCR_VISIBLE_DEVICES', 'TMPDIR', 'RADIANCE_LOG_STEPS']})
        address = 'http://127.0.0.1:' + str(port)
        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urlopen(Request(address + path, data=data, headers={'Content-Type': 'application/json'}), timeout=180) as r:
                return r.read()
        with socket.socket() as check_port:
            check_port.bind(('127.0.0.1', port))
        with (sd / 'server.log').open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    assert process.poll() is None, 'Server exited during startup'
                    try:
                        request('/health'); break
                    except (OSError, URLError):
                        if time.monotonic() > deadline:
                            raise TimeoutError('Server startup')
                        time.sleep(.25)
                observed = {}
                for label, fixture in sequence:
                    if label == 'A-after-reset':
                        (sd / 'cache-reset.json').write_bytes(request('/reset_prefix_cache?model=vision-check', {}))
                    (sd / (label + '.before-metrics.txt')).write_bytes(request('/metrics?model=vision-check'))
                    body = dict(model='vision-check', messages=copy.deepcopy(fixtures[fixture]), temperature=0,
                                seed=42, top_p=1, top_k=0, min_p=0, repetition_penalty=1, max_tokens=64,
                                stream=False, logprobs=True, top_logprobs=5,
                                chat_template_kwargs={'enable_thinking': False})
                    save(sd / (label + '.request.json'), body)
                    response = json.loads(request('/v1/chat/completions', body))
                    finite(response); save(sd / (label + '.response.json'), response)
                    (sd / (label + '.after-metrics.txt')).write_bytes(request('/metrics?model=vision-check'))
                    content = response['choices'][0]['message']['content'].lower()
                    expected = ['red', 'circle'] if fixture == 'A' else ['blue', 'square'] if fixture == 'B' else ['4']
                    assert all(x in content for x in expected), (label, content)
                    if fixture == 'text':
                        assert content.strip() == '4', (label, content)
                    ids, probs = chosen(response)
                    observed[label] = {'ids': ids, 'logprobs': probs, 'cached': response['usage']['prompt_tokens_details']['cached_tokens']}
                assert observed['A-repeat']['cached'] >= 1024 and observed['B-repeat']['cached'] >= 1024, 'No genuine image-containing reuse'
                assert observed['A-after-reset']['cached'] == 0, 'Reset did not invalidate reused prompt'
                for label in ['A-repeat', 'A-after-B', 'A-final', 'A-after-reset']:
                    assert observed[label]['ids'] == observed['A']['ids'], ('Unstable repeated IDs', depth, label)
                assert observed['B-repeat']['ids'] == observed['B']['ids']
                save(sd / 'observed.json', observed); results[str(depth)] = observed
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            assert process.returncode == 0, process.returncode
        text = (sd / 'server.log').read_text()
        assert 'AQL backend' in text and '2 device(s)' in text, 'Requires ordinary TP2 AQL execution'
        assert 'recorded pass:' in text, 'Recorded execution was not exercised'
        assert not re.search(r'^E(?: |\[)', text, re.M), 'Runtime error; inspect server.log'
        if depth:
            for label in results['0']:
                assert results['0'][label]['ids'] == observed[label]['ids'], ('Depth-15 target mismatch', label)
            totals = counters((sd / 'A-after-reset.after-metrics.txt').read_text())
            proposed, accepted = totals.values()
            assert proposed > accepted > 0, 'Fixture did not exercise both acceptance and rejection'
            save(sd / 'draft-counters.json', totals)
    after = model.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Model changed'
    save(args.output / 'result.json', {'executed': args.execute, 'passed': True if args.execute else None,
         'kv_dtype': args.kv_dtype, 'model_size': before.st_size,
         'scope': 'Ordinary TP2 AQL greedy output IDs/cache/rejection; no full hidden-activation parity claim'})


if __name__ == '__main__':
    main()
