# IntentDB benchmark results

Two tracks. **Easy** = topical ambiguity (a diagonal lens nearly saturates it). **Hard** = pragmatic intents (tutorial / reference / troubleshooting / concept) over a shared-topic corpus, which leaves headroom for evaluating further ranking refinements.

Easy track: 42 docs, 6 intents. Hard track: 48 docs, 4 intents (one tutorial/reference/troubleshooting/concept doc per topic).
Reproduce: `python -m bench.run --embedder <spec> --out bench/RESULTS.md`

## Easy track

### easy track · `hashing:dim=512` · 25 paired cases

| config | top-1 | nDCG@10 (95% CI) | p-MRR |
|---|---|---|---|
| plain | 44% | 0.770 [0.685, 0.849] | +0.000 |
| auto-intent | 40% | 0.737 [0.648, 0.824] | +0.000 |
| lens-only | 44% | 0.762 [0.677, 0.847] | +0.041 |
| affinity-only | 20% | 0.402 [0.262, 0.548] | +0.342 |
| full | 64% | 0.839 [0.747, 0.921] | +0.253 |
| full+hybrid | 52% | 0.804 [0.716, 0.886] | +0.118 |
| full+rerank | 84% | 0.933 [0.873, 0.985] | +0.674 |
| full+hybrid+rerank | 84% | 0.933 [0.873, 0.985] | +0.674 |

```
plain                ##################------ 0.770
auto-intent          ##################------ 0.737
lens-only            ##################------ 0.762
affinity-only        ##########-------------- 0.402
full                 ####################---- 0.839
full+hybrid          ###################----- 0.804
full+rerank          ######################-- 0.933
full+hybrid+rerank   ######################-- 0.933
```

Full vs plain nDCG@10 delta (paired bootstrap 95% CI): **+0.069** [+0.002, +0.152], **significant**.

### easy track · `ollama:model=nomic-embed-text` · 25 paired cases

| config | top-1 | nDCG@10 (95% CI) | p-MRR |
|---|---|---|---|
| plain | 44% | 0.775 [0.695, 0.857] | +0.000 |
| auto-intent | 40% | 0.587 [0.437, 0.733] | +0.000 |
| lens-only | 92% | 0.970 [0.926, 1.000] | +0.754 |
| affinity-only | 12% | 0.469 [0.375, 0.573] | +0.322 |
| full | 96% | 0.985 [0.956, 1.000] | +0.766 |
| full+hybrid | 84% | 0.941 [0.882, 0.985] | +0.439 |
| full+rerank | 84% | 0.933 [0.873, 0.985] | +0.674 |
| full+hybrid+rerank | 84% | 0.933 [0.873, 0.985] | +0.674 |

```
plain                ###################----- 0.775
auto-intent          ##############---------- 0.587
lens-only            #######################- 0.970
affinity-only        ###########------------- 0.469
full                 ######################## 0.985
full+hybrid          #######################- 0.941
full+rerank          ######################-- 0.933
full+hybrid+rerank   ######################-- 0.933
```

Full vs plain nDCG@10 delta (paired bootstrap 95% CI): **+0.210** [+0.129, +0.294], **significant**.

## Hard track

### hard track · `hashing:dim=512` · 48 paired cases

| config | top-1 | nDCG@10 (95% CI) | p-MRR |
|---|---|---|---|
| plain | 23% | 0.554 [0.473, 0.639] | +0.000 |
| auto-intent | 21% | 0.525 [0.437, 0.612] | +0.000 |
| lens-only | 21% | 0.542 [0.453, 0.625] | +0.014 |
| affinity-only | 6% | 0.145 [0.073, 0.233] | +0.066 |
| full | 29% | 0.582 [0.488, 0.674] | +0.099 |
| full+hybrid | 31% | 0.610 [0.527, 0.699] | +0.069 |
| full+rerank | 50% | 0.712 [0.622, 0.799] | +0.256 |
| full+hybrid+rerank | 50% | 0.712 [0.622, 0.799] | +0.256 |

```
plain                #############----------- 0.554
auto-intent          #############----------- 0.525
lens-only            #############----------- 0.542
affinity-only        ###--------------------- 0.145
full                 ##############---------- 0.582
full+hybrid          ###############--------- 0.610
full+rerank          #################------- 0.712
full+hybrid+rerank   #################------- 0.712
```

Full vs plain nDCG@10 delta (paired bootstrap 95% CI): **+0.028** [-0.029, +0.089], not significant.

### hard track · `ollama:model=nomic-embed-text` · 48 paired cases

| config | top-1 | nDCG@10 (95% CI) | p-MRR |
|---|---|---|---|
| plain | 25% | 0.619 [0.553, 0.690] | +0.000 |
| auto-intent | 25% | 0.584 [0.507, 0.669] | +0.000 |
| lens-only | 56% | 0.778 [0.699, 0.856] | +0.323 |
| affinity-only | 6% | 0.258 [0.180, 0.344] | +0.150 |
| full | 62% | 0.798 [0.716, 0.875] | +0.362 |
| full+hybrid | 48% | 0.736 [0.659, 0.814] | +0.224 |
| full+rerank | 50% | 0.710 [0.620, 0.798] | +0.252 |
| full+hybrid+rerank | 50% | 0.710 [0.620, 0.798] | +0.252 |

```
plain                ###############--------- 0.619
auto-intent          ##############---------- 0.584
lens-only            ###################----- 0.778
affinity-only        ######------------------ 0.258
full                 ###################----- 0.798
full+hybrid          ##################------ 0.736
full+rerank          #################------- 0.710
full+hybrid+rerank   #################------- 0.710
```

Full vs plain nDCG@10 delta (paired bootstrap 95% CI): **+0.179** [+0.106, +0.254], **significant**.

## Notes

- **p-MRR** is the paired reciprocal-rank delta (FollowIR-style); ~0 means the configuration is blind to intent (plain cosine returns the same ranking for every intent, so its p-MRR is exactly 0).
- The significance line is a paired bootstrap CI on the per-case nDCG@10 difference between the full stack and plain cosine.
- Rerank rows use FlashRank's TinyBERT, which does topical steering from the injected intent text, not true instruction following (see FollowIR in REFERENCES.md).
- The hard track's headroom (the full stack well below 1.0) leaves room for future ranking refinements such as per-intent low-rank adapters.
