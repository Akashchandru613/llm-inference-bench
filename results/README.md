# Results layout

```
results/
├── runs/                  raw per-config records (gitignored by default)
│   └── <config_fp>/
│       └── <timestamp>__r<repeat>.json
└── summary/               aggregated tables produced by `make analyze`
    └── summary.json
```

To make a result reviewable, commit specific files explicitly with
`git add -f results/runs/<fp>/<file>.json` after a sweep finishes.

The result JSON schema is defined by `RunRecord` in
`src/llm_bench/records.py`. Every record carries enough metadata (config
fingerprint, env snapshot, prompts fingerprint) to be replayable.
