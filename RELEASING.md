# Releasing vecshift

Pushing a version tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml).
It runs the checks, then publishes:

- the package to [PyPI](https://pypi.org/project/vecshift/), with trusted publishing and
  attestations, so no API token is stored anywhere
- the Docker image to `ghcr.io/osamamu64/vecshift`, for amd64 and arm64, with a signed
  build provenance attestation and an SBOM
- a GitHub release with the wheel, the sdist, and the CHANGELOG section as notes

The workflow stops before publishing anything if the tag doesn't match the package
version or CHANGELOG.md has no section for it.

## One-time setup

1. **PyPI trusted publisher.** Sign in to PyPI, open
   [Publishing](https://pypi.org/manage/account/publishing/), and add a pending publisher:

   | Field | Value |
   |---|---|
   | PyPI project name | `vecshift` |
   | Owner | `Osamamu64` |
   | Repository name | `vecshift` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first release creates the project and turns the pending publisher into a regular
   one.
2. **GitHub environment.** In the repository's Settings → Environments, create `pypi`.
   Add yourself as a required reviewer, so every upload waits for your approval, and
   limit deployments to tags matching `v*`.
3. **Tag protection.** In Settings → Rules, add a tag ruleset for `v*` that only you can
   create, update, or delete.
4. **After the first release**, open the `vecshift` package under your GitHub profile's
   Packages, set its visibility to public, and connect it to this repository.

## Each release

1. Pick the version, following [Semantic Versioning](https://semver.org/). Before 1.0, a
   minor version (0.2.0) may change behaviour or the `--json` output; a patch version
   (0.1.1) only fixes things.
2. In a pull request:
   - set `__version__` in `src/vecshift/__init__.py`
   - in CHANGELOG.md, move the entries under `[Unreleased]` into a new
     `## [X.Y.Z] - YYYY-MM-DD` section, and update the links at the bottom
3. Merge it once CI passes, then tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vecshift X.Y.Z"
   git push origin vX.Y.Z
   ```

4. Approve the `pypi` deployment when the workflow asks, and wait for it to finish.
5. Check what was published:

   ```bash
   uvx vecshift@X.Y.Z --version
   docker run --rm ghcr.io/osamamu64/vecshift:X.Y.Z --version
   gh attestation verify oci://ghcr.io/osamamu64/vecshift:X.Y.Z --repo Osamamu64/vecshift
   ```

Pre-releases work the same way, with versions such as `0.2.0rc1` and tags such as
`v0.2.0rc1`; the GitHub release is marked as a pre-release.

## If a release fails

- **Before the PyPI upload** (checks, version, notes): fix it on `main`, delete the tag
  locally and on GitHub, and tag again.
- **After the PyPI upload**: PyPI never accepts the same version twice, even if you
  delete it. Fix the problem and release the next patch version. To steer people away
  from a broken release, [yank it](https://pypi.org/help/#yanked) on PyPI.
- **Only the image or the GitHub release failed**: re-run the failed jobs from the
  workflow run's page.
