# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-26

### Added
- Cells of the coarse-graining map may overlap (`CoarseGrainingMap(..., allow_overlap=True)` by
  default; `overlaps` lists the overlapping pairs).
- `ValueMap.ground_joint`: optional joint grounding for intervened macro-variables whose cells
  overlap. Without it, such samples are skipped and reported as `overlap_skipped` in the results.
- ISIL/IC baselines on every benchmark system; gas case-study script.

### Changed
- Camera-ready revisions of the benchmark systems (gas, predator-prey, heat equations, Ising, GRN,
  6502, experiments 2 and 5); see the paper's appendix.
- Faithful CAE no longer reserves an intervention slot for Phi when Phi is empty.

## [1.0.0] - 2026-07-01

- Initial release.
