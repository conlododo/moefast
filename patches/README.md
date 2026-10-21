# patches/

`moefast.patch` — the complete diff against upstream llama.cpp commit `1ba568c` (2026-10-20 master).
Four files, +123/−4 lines:

| File | What |
|---|---|
| `common/arg.cpp` | `--moe-cache-mib auto` |
| `src/llama-context.cpp` | auto sizing from measured free VRAM + `LLAMA_MOE_CACHE_MIB` / `LLAMA_MOE_CACHE_RESERVE_MIB` |
| `src/llama-moe-cache.cpp` | heat-table pre-seeding (`LLAMA_EXPERT_HEAT`) |
| `src/llama-moe-cache.h` | the seeding entry point |

## Regenerate (maintainer)

```bash
# inside the fork checkout, on the feature branch
git diff <upstream-commit>..HEAD -- . ':(exclude)moefast' > moefast/patches/moefast.patch

# or, if the branch was rebased onto a newer upstream:
git diff $(git merge-base HEAD origin/master)..HEAD -- . ':(exclude)moefast' \
  > moefast/patches/moefast.patch
```

## Apply

```bash
git apply moefast.patch          # exact base
git apply -3 moefast.patch       # tolerate small upstream drift
```
