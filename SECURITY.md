# Security policy

## Reporting a vulnerability

Report privately through GitHub's advisory workflow:
<https://github.com/kogunlowo123/domain-slm/security/advisories/new>

Please do not open a public issue for a vulnerability.

Include what you have: the corpus, model directory or command that triggers it,
what you expected, what happened, and the version or commit. A proof of concept
helps but is not required.

**What to expect.** This is a portfolio project maintained by one person, not a
funded product, and it is fairer to say so than to publish a service-level
agreement nobody is on call for. Acknowledgement within a week is realistic. If
a report is valid it will be fixed and credited, and the fix will say what was
wrong rather than describing it as "hardening".

## Supported versions

The `main` branch. There is no backport policy for older tags.

## What this protects, and what it does not

[THREAT-MODEL.md](THREAT-MODEL.md) is the document to read before running this
on anything that matters. The short version:

**In scope.** An evaluation that can be made to report a number it should have
refused; resource exhaustion through a hostile corpus, including a decompression
bomb; a model artefact that is not what it claims to be; operational text
leaking through a schema that grew a field for it.

**Out of scope.** Data that is wrong rather than malformed — there is no
provenance check on a record's *content* and no attempt at one. An attacker who
can already edit the objectives in your repository. Adversarial examples against
the classifier, which is a chapter classifier and not a security control.

## The things most worth understanding

**This package has no network path at all.** No HTTP client, no socket, no
credential setting. Unusually for this series that is not a policy but a
property, and `tests/security/` asserts it over the source rather than over one
run.

**A record has nowhere to put free text beyond the work order.** `extra="forbid"`
refuses an added field at validation — `prompt`, `email`, `user_id` and four
others are asserted refused — and `text` is bounded so the field that does exist
cannot become a transcript by accident.

**When a gate fails, no metric is reported at all.** Not a number with a warning
above it: the JSON has no `metrics` key, the Markdown prints no accuracy, and the
JUnit XML has a failing case. A number that exists is a number somebody quotes
out of context, and there is a test that would fail if the refusal were removed.

**Compressed input is bounded.** Accepting `.gz` corpora means accepting a
decompression bomb, so the size limit applies to the decompressed stream and not
only to the file on disk. Without that a small file expands until the runner is
killed — which reaches an operator as a flaky build rather than as an attack.

**Model weights are arrays, never a pickle.** Loading a pickle from an untrusted
path executes whatever it was told to. `allow_pickle` is left at its safe
default and the security tests assert the string does not appear in the module.

## Credentials

This project handles none. It reads files and writes reports; there is nothing
to authenticate to. `.env.example` contains no credential field because there is
nothing for one to configure, and the settings model has no `api_key`, `token`
or `secret` field — asserted by a test.

Never commit a populated `.env`. It is gitignored, and CI runs `gitleaks` over
both the working tree and the **full git history** on every push.

## What CI enforces

| Check | Tool | Scope |
| --- | --- | --- |
| Secrets | gitleaks | Working tree and full history |
| Static analysis | bandit | `src/` |
| Dependency vulnerabilities | pip-audit | The locked set, `--strict --no-deps` |
| Code scanning | CodeQL | `security-extended` queries |
| Filesystem and config | trivy | HIGH and CRITICAL |
| Container image | trivy | HIGH and CRITICAL, plus a non-root assertion |

A HIGH or CRITICAL dependency finding fails the build. Time-boxed exceptions are
recorded in [`security/audit-exceptions.md`](security/audit-exceptions.md) with
an identifier, a reason, a compensating control and a review date — never as a
bare entry in an ignore list.

## Container

The image runs as uid 10001 with no build toolchain in the final layer, and
`docker-compose.yml` sets `network_mode: none` on every service — which costs
nothing here, because nothing in the package would use a network anyway.

## Dependencies

Five at runtime: `numpy`, `pydantic`, `pydantic-settings`, `structlog`,
`pyyaml`. The model is written out in NumPy rather than delegated to a deep
learning framework, which avoids a 900 MB dependency and its transitive tree for
a model with three weight matrices. `uv.lock` is committed and CI installs with
`--locked`.
