# Sampling reproduction check (Phase 2)

## hotpot
- Eligible population (reproduced, read-only): 7405
- Example-ID SET match: True
- Example-ID ORDER match: True
- eligible_population_sha256 match: True (reproduced=b14d5bfd2b23afc87c6f42aeed1ff2898e3c204a2058c7a50c2a0d0ec2b30558, existing=b14d5bfd2b23afc87c6f42aeed1ff2898e3c204a2058c7a50c2a0d0ec2b30558)
- ordered_sample_id_sha256 match: True (reproduced=f44015238effdcbc554751e1c9cce859cae70e41f0bc88d51c9aa93c6fc97fb6, existing=f44015238effdcbc554751e1c9cce859cae70e41f0bc88d51c9aa93c6fc97fb6)
- Manifest file SHA-256 note: this script's `manifest_rows` are recomputed independently and hashed as newline-joined sorted-key JSON for a structural check; this does NOT need to equal the on-disk manifest file's raw-bytes SHA-256 (1c44a02e3c1d8b17e7a5bff3c666586526513536a0c8d85ead50bad7dee88a92) because JSON key order in the raw file may differ -- the authoritative reproducibility signals are the ID set/order and the two SHA-256 digests already embedded in the production provenance file (`ordered_sample_id_sha256`, `eligible_population_sha256`), both checked above.
- Overall exact reproduction: PASS

## 2wiki
- Eligible population (reproduced, read-only): 12576
- Example-ID SET match: True
- Example-ID ORDER match: True
- eligible_population_sha256 match: True (reproduced=30b61493923b6c8a038bfaa96af7065cda7bddfbea416c20d34ea290a342f0b0, existing=30b61493923b6c8a038bfaa96af7065cda7bddfbea416c20d34ea290a342f0b0)
- ordered_sample_id_sha256 match: True (reproduced=ae7359ceb46e06aadd29983bc4d312c3b9a60155c4d0f85ad30e6e2d674b1a2e, existing=ae7359ceb46e06aadd29983bc4d312c3b9a60155c4d0f85ad30e6e2d674b1a2e)
- Manifest file SHA-256 note: this script's `manifest_rows` are recomputed independently and hashed as newline-joined sorted-key JSON for a structural check; this does NOT need to equal the on-disk manifest file's raw-bytes SHA-256 (4f1cc6236375201a128e03c0d074578e5fe93e75ebf4a32427fcf3d61ebe864f) because JSON key order in the raw file may differ -- the authoritative reproducibility signals are the ID set/order and the two SHA-256 digests already embedded in the production provenance file (`ordered_sample_id_sha256`, `eligible_population_sha256`), both checked above.
- Overall exact reproduction: PASS

## musique
- Eligible population (reproduced, read-only): 2417
- Example-ID SET match: True
- Example-ID ORDER match: True
- eligible_population_sha256 match: True (reproduced=35204bfe763b1091e075af33eae08c789c4b1b0f0e3e6d1da40ba6de82f16d03, existing=35204bfe763b1091e075af33eae08c789c4b1b0f0e3e6d1da40ba6de82f16d03)
- ordered_sample_id_sha256 match: True (reproduced=1cc2cce1e67761819ced7e093dc7fd81905850da93614776250a7882b93e29ff, existing=1cc2cce1e67761819ced7e093dc7fd81905850da93614776250a7882b93e29ff)
- Manifest file SHA-256 note: this script's `manifest_rows` are recomputed independently and hashed as newline-joined sorted-key JSON for a structural check; this does NOT need to equal the on-disk manifest file's raw-bytes SHA-256 (995f584e1c466347d4c30deebbab7e38411170269bb077011061b09dd5d0ad2e) because JSON key order in the raw file may differ -- the authoritative reproducibility signals are the ID set/order and the two SHA-256 digests already embedded in the production provenance file (`ordered_sample_id_sha256`, `eligible_population_sha256`), both checked above.
- Overall exact reproduction: PASS

## multihoprag
- Eligible population (reproduced, read-only): 2255
- Example-ID SET match: True
- Example-ID ORDER match: True
- eligible_population_sha256 match: True (reproduced=70778ef85e8c8ab4aa06dde143bc13529492a4270606703f284be5c11abb00b8, existing=70778ef85e8c8ab4aa06dde143bc13529492a4270606703f284be5c11abb00b8)
- ordered_sample_id_sha256 match: True (reproduced=a8aef57352e9de592baa0c5a9feef3b3047cfd6bed1b400475224e5605b3768f, existing=a8aef57352e9de592baa0c5a9feef3b3047cfd6bed1b400475224e5605b3768f)
- Manifest file SHA-256 note: this script's `manifest_rows` are recomputed independently and hashed as newline-joined sorted-key JSON for a structural check; this does NOT need to equal the on-disk manifest file's raw-bytes SHA-256 (dcc3c19c08b319299a55e4bfc2d8b781a2e9918b0f260eba67e64bf2a375fa92) because JSON key order in the raw file may differ -- the authoritative reproducibility signals are the ID set/order and the two SHA-256 digests already embedded in the production provenance file (`ordered_sample_id_sha256`, `eligible_population_sha256`), both checked above.
- Overall exact reproduction: PASS

