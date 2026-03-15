# Upstream Version

Synced to: **real-stanford/diffusion_policy** HEAD of `main` (2026-03-15)

- Repository: https://github.com/real-stanford/diffusion_policy
- Paper: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
- Commit: HEAD of main branch as of 2026-03-15

## Sync Protocol

When upstream updates:

1. `cd repositories/diffusion-policy-upstream && git fetch && git pull`
2. Review changes in `diffusion_policy/model/` and `diffusion_policy/policy/`
3. Config-only changes: copy verbatim
4. New nn types: add to compat layer
5. Structural changes: mirror in MLX implementation
6. Update this file with new sync date
