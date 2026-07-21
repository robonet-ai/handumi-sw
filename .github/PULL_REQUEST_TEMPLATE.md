## Summary

Describe the change and the user-visible contract it preserves or changes.

## Verification

List exact commands and results. Separate software, simulator, Quest, and
physical-hardware evidence; do not imply hardware validation from offline tests.

## Release and safety review

- [ ] Controller-only behavior remains covered, or the compatibility change is documented.
- [ ] Raw observations, masks, timing, calibration, and provenance are preserved.
- [ ] New estimated signals fail closed and are not described as measured.
- [ ] No participant data, private addresses, device identifiers, secrets,
      signing material, local rig config, caches, outputs, or generated Unity state is included.
- [ ] Asset/dependency provenance and redistribution terms are documented.
- [ ] Physical robot changes retain independent emergency-stop, watchdog,
      workspace, motion-limit, collision, and preview requirements.

## Blocked gates

List skipped hardware, signing, legal, owner, CI, or laboratory gates and who
can provide the missing evidence.
