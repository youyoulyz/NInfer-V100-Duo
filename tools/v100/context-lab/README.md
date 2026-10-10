# V100 long-context lab

Two launcher + driver pairs for the two-card V100 build, used to characterise the retained-context
tiers and the decode batch of `qwen3.8-27b/nvfp4` at long context. They are run from a scratch
directory (`~/v100/ninfer-context-lab/{pool,batch}`), which symlinks the scripts and keeps the
server and driver logs; everything needed to reproduce a measurement lives here.

| Profile | Launcher | Driver | Shape |
|---|---|---|---|
| Session pool | [`serve-pool.sh`](serve-pool.sh) | [`pool_driver.py`](pool_driver.py) | Several full-length conversations retained across restarts, one request at a time |
| Batch decode | [`serve-batch-decode.sh`](serve-batch-decode.sh) | [`batch_decode.py`](batch_decode.py) | N concurrent streams decoding already-resident prefixes |
| Reuse probe | [`serve-batch-decode.sh`](serve-batch-decode.sh) | [`reuse_probe.py`](reuse_probe.py) | S retained prefixes replayed serially and concurrently |
| Adoption gate | [`serve-batch-decode.sh`](serve-batch-decode.sh) | [`adopt_probe.py`](adopt_probe.py) | One prompt decoded cold, then twice concurrently; the second stream can only reach the prefix by adopting it, and both must reproduce the cold text byte for byte |

Both launchers take `CONTEXT_LAB_ARTIFACT` (default `/home/luyzh/models/qwen3_8_27b_nvfp4.ninfer`)
and `CONTEXT_LAB_PORT` (default 8080), and require a build in `build-v100-duo/`.

## Session pool

`serve-pool.sh` runs one lane with the pinned host tier and a 96 GiB NVMe tier under
`CONTEXT_LAB_TIER_DIR`, at `--max-context 200000 --kv-capacity 200000`: a prefix displaced from
the paged KV pool is parked instead of discarded, so a conversation that comes back -- or a server
that restarts -- resumes it with no re-prefill.

```bash
cd ~/v100/ninfer-context-lab/pool
CONTEXT_LAB_TIER_DIR=/var/tmp/ninfer-context-lab-pool ./serve.sh > logs/cold.log 2>&1 &
python3 pool.py cold 8 192000 --serve-log logs/cold.log --state sessions.json
kill $(pgrep -x ninfer-serve)
CONTEXT_LAB_TIER_DIR=/var/tmp/ninfer-context-lab-pool ./serve.sh > logs/resume.log 2>&1 &
python3 pool.py resume 8 192000 --serve-log logs/resume.log --state sessions.json
```

`cold` builds the pool and `resume` replays it; the state file carries the calibrated line count so
both phases build byte-identical prompts, and `resume` exits non-zero unless every session resumed
with `reuse=append_frontier` and at most a two-token prefill.

Recorded with eight sessions at 192000 target tokens (194950 counted), `--max-concurrency 1`,
`--tp 2 --spec mtp --draft-tokens 3`, INT8 KV on 2 x V100-SXM2-16GB (logs in
`~/v100/ninfer-context-lab/pool/logs`):

| Phase | Result |
|---|---|
| Cold pool build | TTFT 298.7-327.8 s per session, 595-653 tok/s prefill, about 7 GiB per session in the tier |
| Two process restarts later | 8/8 resumed: `cache=194948/194950`, `reuse=append_frontier`, two-token append, no re-prefill, TTFT 5.2-6.2 s |

The per-session tier cost follows from the INT8 G64 KV layout (17952 B per token per device) plus
the GDN state; size `--disk-kv-mib` for the whole pool.

## Batch decode

`batch_decode.py` measures the decode batch of already-resident prefixes: each shape `NxLENGTH`
warms N prefixes, then times one round of N simultaneous requests with 64 decode tokens.
`run-ladder.sh` restarts the server between shapes, because one server cannot host the whole
ladder (see its header).

Recorded on 2 x V100-SXM2-16GB, `--tp 2 --spec mtp --draft-tokens 3`, INT8 KV, one round per shape,
prompt lengths as counted by `/v1/messages/count_tokens`:

| Shape | Per-stream tok/s (min-max) | Aggregate tok/s (wall) | Aggregate tok/s (lockstep) | 1 stream tok/s | Uplift |
|---|---|---|---|---|---|
| 8 x 9983 | 36.9 (35-39) | 221.1 | 281.2 | 95.8 | 2.31x |
| 8 x 20007 | 40.8 (40-43) | 228.6 | 316.5 | 104.8 | 2.18x |
| 5 x 29975 | 43.9 (42-45) | 176.7 | 212.1 | 88.4 | 2.00x |
| 4 x 39999 | 50.5 (49-53) | 167.9 | 195.5 | 84.8 | 1.98x |
| 2 x 79624 | 76.4 (76-77) | 124.7 | 152.8 | 80.6 | 1.55x |

Every measured request resumed its own prefix -- `reuse=restore_turn_checkpoint`, two-token
prefill -- so no timed round paid a re-prefill. Aggregate decode is flat near 220 tok/s for eight
lanes whatever the context length, while the per-stream rate falls as streams are added: decode is
bandwidth-bound, so the batch buys about 2.3x over one stream rather than 8x.

## Shared prefix versus one prefix per lane

`run-prefix-compare.sh LENGTH N...` times N streams at one prompt length twice: once with N
distinct prompts (a different cached prefix in every lane) and once with N byte-identical prompts
(`--shared-prefix`), with a fresh server per run.

Recorded on the same profile at about 10000 prompt tokens, 64 decode tokens, one round:

| N | Variant | Streams that re-prefilled | Window | Aggregate tok/s (wall) | Per-stream tok/s |
|---|---|---|---|---|---|
| 2 | distinct | 0/2 | 0.79 s | 160.5 | 91.5 |
| 2 | shared | 1/2 | 26.31 s | 4.8 | 101.6 |
| 4 | distinct | 0/4 | 1.29 s | 195.0 | 58.2 |
| 4 | shared | 2/4 | 52.09 s | 4.8 | 92.5 |
| 5 | distinct | 0/5 | 1.63 s | 193.0 | 47.9 |
| 5 | shared | 2/5 | 52.00 s | 6.1 | 100.3 |
| 8 | distinct | 0/8 | 2.27 s | 222.5 | 36.8 |
| 8 | shared | 3/8 | 78.43 s | 6.4 | 76.7 |

A repeated prompt does not fill the lanes. The admission pass prefers the lane that already holds
the prefix, so the shared variant's warm phase serialises every stream onto one lane and leaves a
single cached copy. A concurrent burst then finds exactly one lane with the cache; the rest land on
empty lanes and pay a full prefill, one at a time. A second round does not fix it: eight shared
streams re-prefilled 1/8 again and still took 27.8 s (18.1 tok/s), because the copy-holding lanes
are occupied by the earlier streams of the same round. Per-stream decode looks *faster* in the
shared case (76.7 vs 36.8 tok/s at N=8) precisely because the streams never batch -- they stagger,
so each runs near single-stream speed while the wall clock accumulates. The KV content does not
change the kernels; what collapses is whether the batch forms at all.

## Reuse probe

`reuse_probe.py` is the regression check for lane admission. It warms one retained prefix per lane
and then replays the prompt set serially, concurrently, and once more serially; every replay must
resume its own prefix with at most a two-token prefill, and the probe exits non-zero if any replay
had to re-prefill. It runs against the batch-decode server:

```bash
cd ~/v100/ninfer-context-lab/batch
python3 reuse_probe.py --request-log logs/probe.jsonl
```
