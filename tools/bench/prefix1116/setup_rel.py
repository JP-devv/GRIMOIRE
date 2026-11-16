#!/usr/bin/env python3
# Tower-side setup for the v1.10.7 release tests (2026-11-16): harness run_rel1116.py = run_pfx1116.py plus
#   --image IMG --hostbin 0   run the image's OWN binaries (no tree bin / entrypoint mounts)
#   --profile-env             start from the Unraid profile base env instead of the bench defaults (MTP K6 etc.)
#   --benchy-args '...'       llama-benchy matrix; tokenizer from the model's host path
#   /mnt/storage/Models mounted at /models (Ornith lives there)
import os
B = "/mnt/storage/isos/grimoire-runs/bench-1022/"
s = open(B + "run_pfx1116.py").read()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


sub("P.add_argument('--bb-args', default='')",
    "P.add_argument('--bb-args', default='')\n"
    "P.add_argument('--image', default='grimoire-b70:dev-cancel-1111')\n"
    "P.add_argument('--hostbin', type=int, default=1)\n"
    "P.add_argument('--profile-env', action='store_true')\n"
    "P.add_argument('--benchy-args', default='--pp 4096 16384 --tg 128 --concurrency 1 2 4 8 --runs 3 --no-cache')")
sub("""for kv in O.env:""", """if O.profile_env:   # the Unraid GRIMOIRE profiles' base variables only
    env = {'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu', 'ZE_AFFINITY_MASK': '0',
           'UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS': '1'}
for kv in O.env:""")
sub("""    a += ['-v', str(TREE / 'bin') + ':/grimoire/bin:ro',
          '-v', str(TREE / 'docker-entrypoint.sh') + ':/usr/local/bin/docker-entrypoint.sh:ro', '-v', '/mnt/models:/models2:ro']""",
    """    if O.hostbin:
        a += ['-v', str(TREE / 'bin') + ':/grimoire/bin:ro',
              '-v', str(TREE / 'docker-entrypoint.sh') + ':/usr/local/bin/docker-entrypoint.sh:ro']
    a += ['-v', '/mnt/models:/models2:ro', '-v', '/mnt/storage/Models:/models:ro']""")
sub("""    a += ['grimoire-b70:dev-cancel-1111'] + entry""", """    a += [O.image] + entry""")
sub("""               '--tokenizer', '/mnt/models/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8',
               '--book-url', 'http://127.0.0.1:8999/1661-0.txt',
               '--pp', '4096', '16384', '--tg', '128', '--concurrency', '1', '2', '4', '8',
               '--runs', '3', '--no-cache', '--format', 'md', '--save-result', str(OUT / 'benchy.md')]""",
    """               '--tokenizer', MODEL.replace('/models2/', '/mnt/models/', 1) if MODEL.startswith('/models2/')
               else MODEL.replace('/models/', '/mnt/storage/Models/', 1),
               '--book-url', 'http://127.0.0.1:8999/1661-0.txt'] + O.benchy_args.split() + [
               '--format', 'md', '--save-result', str(OUT / 'benchy.md')]""")
open(B + "run_rel1116.py", "w").write(s)
print("setup ok")
