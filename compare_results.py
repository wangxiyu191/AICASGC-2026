import json, sys

b = json.load(open('results/result_baseline_nosplit_off.json'))
n = json.load(open('results/result_nosplit_on.json'))

bp = b['performance']
np_ = n['performance']

print('=== Performance ===')
print(f"Baseline: TTFT={bp['avg_ttft_ms']:.2f}ms, Throughput={bp['avg_throughput_tokens_per_sec']:.2f} tok/s")
print(f"Nosplit:  TTFT={np_['avg_ttft_ms']:.2f}ms, Throughput={np_['avg_throughput_tokens_per_sec']:.2f} tok/s")
print(f"Throughput improvement: {(np_['avg_throughput_tokens_per_sec']/bp['avg_throughput_tokens_per_sec']-1)*100:.1f}%")
print()

print('=== Accuracy ===')
ba = b['answers']
na = n['answers']
for i, (bs, ns) in enumerate(zip(ba, na)):
    bans = bs.get('model_answer', bs.get('answer', ''))
    nans = ns.get('model_answer', ns.get('answer', ''))
    match = bans == nans
    print(f"Sample {i}: match={match}")
    if not match:
        print(f"  baseline: {bans[:120]}")
        print(f"  nosplit:  {nans[:120]}")
