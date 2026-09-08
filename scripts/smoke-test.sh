#!/usr/bin/env bash
# Exercises the built container image, not the source tree.
#
# The point is to catch the class of defect every other gate here is blind to:
# the image builds, starts, and is still broken because something the code needs
# is not in it. Everything else in the pipeline tests the source; this is the
# only thing that tests the artefact.
#
# `dslm` is a command-line tool with no listener, so there is no health endpoint
# to curl. The equivalent — and the stronger check — is to run the project's own
# gates *inside* the image and assert both that they pass on clean data and that
# they still fail, with the right exit code, on data that should fail.
#
# Rules learned the expensive way, kept because each cost a debugging session:
#
#   1. **Assert against the shipped artefact's own quality gate**, not just its
#      liveness. An image that starts is not an image that works.
#   2. **Docker Desktop on Windows does not share `mktemp -d` paths.** A bind
#      mount of one silently yields an *empty directory* — no error. Put fixture
#      workspaces under the project directory and convert with `pwd -W`.
#   3. **chmod a bind-mounted directory the image must write to.** The image
#      does not run as root; on Linux a bind mount carries the host's ownership
#      through, so a directory owned by the CI runner's user is one the
#      container's user cannot write to. Docker Desktop on Windows ignores
#      ownership entirely, so this failure appears only in CI.
#   4. **Print the container's stderr when a step fails**, or the assertion
#      names a symptom and hides its cause.
#
# Failures are counted rather than fatal, so one run reports everything that is
# wrong instead of the first thing.
set -euo pipefail

IMAGE="${1:?usage: smoke-test.sh <image[:tag]>}"
failures=0

# Rule 2: a path Docker Desktop will actually share.
REPORTS_LOCAL="${PWD}/var/smoke-reports"
mkdir -p "${REPORTS_LOCAL}"
rm -f "${REPORTS_LOCAL}"/* 2>/dev/null || true
# Rule 3: the container writes here as uid 10001.
chmod 0777 "${REPORTS_LOCAL}" 2>/dev/null || true
REPORTS_HOST="$(cd "${REPORTS_LOCAL}" && pwd -W 2>/dev/null || printf %s "${REPORTS_LOCAL}")"

# The examples are baked into the image, so most checks need no mount at all.
# --network none throughout: this package has no HTTP client, and running it
# with the network removed proves that rather than asserting it.
run() { docker run --rm --network none "${IMAGE}" "$@"; }

SPLITS="examples/splits"
MODEL="examples/model"
NULL="examples/holdout.jsonl.gz"

check() { # check <description> <command...>
  local description="$1"
  shift
  local output
  if output="$("$@" 2>&1)"; then
    echo "  ok    ${description}"
  else
    echo "  FAIL  ${description}"
    printf '%s\n' "${output}" | tail -15 | sed 's/^/          /'   # rule 4
    failures=$((failures + 1))
  fi
}

check_exit() { # check_exit <expected> <description> <command...>
  local expected="$1" description="$2"
  shift 2
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  if [ "${status}" -eq "${expected}" ]; then
    echo "  ok    ${description} (exit ${status})"
  else
    echo "  FAIL  ${description}: expected exit ${expected}, got ${status}"
    printf '%s\n' "${output}" | tail -15 | sed 's/^/          /'   # rule 4
    failures=$((failures + 1))
  fi
}

echo "--- the image is what it claims to be"
check "the default command runs the self-check" \
  run doctor
check "the process does not run as root" \
  docker run --rm --network none --entrypoint sh "${IMAGE}" -c '[ "$(id -u)" != "0" ]'
check "the package is installed, not left as a source path" \
  docker run --rm --network none --entrypoint python "${IMAGE}" -c "import dslm; assert 'site-packages' in dslm.__file__, dslm.__file__"
check "numpy is present and importable" \
  docker run --rm --network none --entrypoint python "${IMAGE}" -c "import numpy; print(numpy.__version__)"

echo
echo "--- rule 1: the shipped image passes its own gates"
# The control. If this is not green then nothing below distinguishes a working
# gate from a broken one.
check_exit 0 "the shipped evaluation passes, contamination and regression" \
  run evaluate --splits "${SPLITS}" --model "${MODEL}" --null "${NULL}" --drop-shared \
      --baseline examples/baseline.json
check_exit 0 "the committed corpus still matches its plan" \
  run check --plan main --corpus examples/corpus.jsonl.gz
check_exit 0 "and so does the holdout" \
  run check --plan holdout --corpus "${NULL}"

echo
echo "--- the gates can still fail, inside the image"
# Exit 2 specifically, not merely non-zero: 3 would mean the tool broke, and a
# job that cannot tell them apart gets retried until it goes green.
check_exit 2 "without a null corpus it refuses to gate" \
  run evaluate --splits "${SPLITS}" --model "${MODEL}"
check_exit 2 "a corpus evaluated against itself is contaminated" \
  run contamination --train "${SPLITS}/train.jsonl.gz" \
      --evaluate "${SPLITS}/train.jsonl.gz" --allow-uncalibrated

# Exit 3 is a different fact and has to stay distinguishable from exit 2.
check_exit 3 "a missing corpus is 'could not run', not 'a gate failed'" \
  run doctor --corpus examples/absent.jsonl.gz
check_exit 3 "a model directory that is not one is refused" \
  run classify --model examples --text "shock strut pressure low"

# And exit 1 is a third fact: argparse's default of 2 would collide with "a
# gate failed", so the parser is subclassed. This is where that is checked in
# the artefact rather than in the source tree.
check_exit 1 "a usage error exits 1, not 2" \
  run nonsense

echo
echo "--- rule 3: it can write reports to a bind mount"
check "reports are written to a mounted directory" \
  docker run --rm --network none -v "${REPORTS_HOST}:/app/reports" "${IMAGE}" \
    evaluate --splits "${SPLITS}" --model "${MODEL}" --null "${NULL}" --drop-shared \
      --json-out reports/evaluation.json --junit-out reports/evaluation.xml \
      --markdown-out reports/evaluation.md
for report in evaluation.json evaluation.xml evaluation.md; do
  if [ -s "${REPORTS_LOCAL}/${report}" ]; then
    echo "  ok    ${report} exists on the host and is not empty"
  else
    echo "  FAIL  ${report} was not written to the bind mount"
    failures=$((failures + 1))
  fi
done

# The report must say it passed. A file that exists and says nothing useful is
# the failure mode a `-s` check alone would miss.
if grep -q '"passed": true' "${REPORTS_LOCAL}/evaluation.json" 2>/dev/null; then
  echo "  ok    the report says the gates held"
else
  echo "  FAIL  the report does not say the gates held"
  failures=$((failures + 1))
fi

echo
if [ "${failures}" -eq 0 ]; then
  echo "smoke test passed for ${IMAGE}"
else
  echo "smoke test FAILED for ${IMAGE}: ${failures} assertion(s)"
  exit 1
fi
