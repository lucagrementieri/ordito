# Releasing triwarp

Maintainer checklist, run in order before every tag push.

Pushing a `v*` tag is the only trigger: it fires
[`release.yml`](workflows/release.yml), which builds, verifies, publishes to PyPI over OIDC and
cuts a GitHub Release. Nothing else publishes, and nothing publishes from `main`.

---

## One-time setup, before the first tag ever

- [ ] **PyPI Trusted Publishing** configured at <https://pypi.org/manage/account/publishing/>,
      pointing at owner `lucagrementieri`, repository `triwarp`, workflow `release.yml`,
      environment `pypi`. There is no API token to store; if this is not set up, the `publish`
      job fails with an OIDC error and nothing is uploaded.
- [ ] **A `pypi` environment** exists in the repository settings (Settings → Environments). The
      workflow names it, and a missing environment blocks the job rather than skipping the gate.
- [ ] **A Test PyPI dry run.** Build locally and upload to `test.pypi.org` by hand, then install
      *from there* into a clean virtualenv and run the smoke test. This is the only way to catch
      a metadata problem without burning a version number — PyPI never allows re-uploading a
      filename, so a botched `0.1.0` upload retires that version string permanently.

      ```bash
      uv build
      uvx twine upload --repository testpypi dist/*
      python -m venv /tmp/testpypi && /tmp/testpypi/bin/pip install \
          --index-url https://test.pypi.org/simple/ \
          --extra-index-url https://pypi.org/simple/ triwarp
      /tmp/testpypi/bin/python -c "import triwarp as tw; print(tw.creation.icosphere(subdivisions=2)[0].shape)"
      ```

- [ ] **GitHub Pages source** set to "GitHub Actions" (Settings → Pages), which `docs.yml` needs.

---

## Every release

### 1. Changelog

- [ ] `CHANGELOG.md`'s `## [Unreleased]` section is complete and accurate.
- [ ] Retitle it `## [X.Y.Z] - YYYY-MM-DD` and open a fresh empty `## [Unreleased]` above it.
- [ ] Add the link-reference line for the new version at the bottom of the file.

`release.yml` cuts the GitHub Release body from the section matching the tag and **fails if there
is none**, so a missing or mistitled heading stops the release rather than shipping empty notes.
Only version headings may be level-2 — the extractor stops at the next `##`.

### 2. Version bump

- [ ] `pyproject.toml` `[project] version` set to `X.Y.Z`.
- [ ] `CITATION.cff` `version` and `date-released` updated to match.
- [ ] The BibTeX block in `README.md` still shows the right `version` and `year`.

`triwarp.__version__` needs no bump — it reads the installed distribution metadata, so it follows
`pyproject.toml` automatically and cannot drift from it.

Follow SemVer with the pre-1.0 caveat: while on `0.x`, **minor** for anything that changes a
public signature, adds or removes a public module, or raises the `warp-lang` floor; **patch** for
bug and documentation fixes only.

`release.yml` asserts the tag matches `pyproject.toml` before building, so a forgotten bump fails
loudly instead of re-publishing the previous version under a new tag.

### 3. The full local gate, on the CUDA box

CI covers the CPU device only. These are the checks that need the GPU and the full
eleven-library reference stack, and they are the maintainer's responsibility:

```bash
uv sync --all-groups

uv run ruff format triwarp tests benchmarks
uv run ruff check triwarp tests benchmarks
uv run basedpyright                        # must be 0 errors

uv run python -m tests.devices             # both devices, as two processes
uv run python -m tests.parity              # pair count unchanged unless intentional
uv run pytest benchmarks/test_meshes.py    # registry/topology self-check, not in the default run

uv run zensical build --strict
```

- [ ] All green; `tests.parity` reports the same pair count as the last release, or the change is
      intentional and described in the changelog.

> Do not substitute `pytest --device=both` for `tests.devices`. In one process, Warp's CPU work
> runs ~36x slower once CUDA has been initialised; the runner buys both devices as two processes
> and hides CUDA from the CPU pass.

### 4. Benchmarks and quoted numbers

Only if a benchmarked function changed since the last release:

- [ ] Re-run the benchmark suite and regenerate the hero charts. Both scripts take a **directory**
      of `--benchmark-json` files, not a single file, so collect the run into one:

      ```bash
      mkdir -p /tmp/bench-run
      uv run pytest benchmarks --benchmark-json=/tmp/bench-run/results.json
      uv run python benchmarks/aggregate.py /tmp/bench-run     # read the suspect report first
      uv run python benchmarks/plot_comparison.py /tmp/bench-run \
          --out docs/assets/benchmarks/ --hero
      ```

- [ ] `aggregate.py`'s suspect report (on by default at 1.5x; tune with `--suspect`) is clean, or
      every flagged cell was re-measured. A cell whose
      median sits far above its own minimum is a one-off (a Warp module load, a scheduler
      hiccup), and an inflated *reference* median flatters triwarp exactly as much as an inflated
      triwarp median hurts it — so a suspect cell must never be published as a ratio.

- [ ] Every ratio quoted in `README.md`, `docs/performance.md` and `docs/benchmarks.md` still
      holds within noise (±10 %, and ±30 % under 100 µs — anything inside that band is drift, not
      a regression).
- [ ] `docs/benchmarks.md`'s methodology paragraph names the version and hardware the charts were
      measured on, and both are current.
- [ ] The refreshed PNGs are committed.

A benchmark box must be quiet: check `nvidia-smi` for foreign processes first, and never run a
timing while the test suite or a docs build is running.

### 5. Build and inspect the artifact

- [ ] Local build is clean and installs into a bare interpreter:

      ```bash
      rm -rf dist && uv build
      uvx twine check dist/*
      unzip -l dist/*.whl | grep py.typed        # must be present
      unzip -l dist/*.whl | awk '{print $4}' | cut -d/ -f1 | sort -u
      #   -> only `triwarp` and `triwarp-X.Y.Z.dist-info`
      ```

- [ ] CPU-only smoke test, with CUDA hidden, which is the actual end-user environment:

      ```bash
      python -m venv /tmp/wheel-check && /tmp/wheel-check/bin/pip install dist/*.whl
      CUDA_VISIBLE_DEVICES="" /tmp/wheel-check/bin/python -c \
          "import triwarp as tw; m = tw.Trimesh(*tw.creation.icosphere(subdivisions=2)); print(m.area, m.is_watertight)"
      ```

### 6. Tag and push

- [ ] Working tree clean and everything above committed to `main`.

      ```bash
      git tag vX.Y.Z && git push origin vX.Y.Z
      ```

### 7. Verify what shipped

- [ ] `release.yml` succeeded end to end — build, publish and github-release.
- [ ] The [PyPI listing](https://pypi.org/project/triwarp/) renders the README correctly, and
      shows the author, license and classifiers.
- [ ] The GitHub Release body is the changelog section, with the wheel and sdist attached.
- [ ] A clean install of the published version works:

      ```bash
      python -m venv /tmp/released && /tmp/released/bin/pip install triwarp==X.Y.Z
      /tmp/released/bin/python -c "import triwarp; print(triwarp.__name__)"
      ```

- [ ] <https://lucagrementieri.github.io/triwarp/> rebuilt and is current.
