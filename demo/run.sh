#!/bin/sh
# The whole migration, start to finish, against the demo database.
# Run by `docker compose up` in this directory; see docs/demo.md.
set -eu

out=/demo/out
if ! touch "$out/.write-test" 2>/dev/null; then
    echo "Can't write to demo/out (it must be writable by uid 1000); reports stay in the container."
    out="$HOME/out"
    mkdir -p "$out"
fi
rm -f "$out/.write-test"

step() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
    printf '$ %s\n\n' "$2"
}

cd "$HOME"
# Each run starts over: seed.py recreates the table, so earlier progress no longer applies.
rm -rf .vecshift vecshift.yaml
python /demo/seed.py

step "1. Check the current index" "vecshift doctor --table documents"
vecshift doctor --table documents --html "$out/doctor.html" || true

step "2. Describe the migration" "cat vecshift.yaml"
cat > vecshift.yaml <<'YAML'
version: 1
name: demo
source:
  table: documents
  text_column: body
  model: hash/64        # the model that made the current vectors
model: hash/256         # the new model
target:
  column: embedding_v2
YAML
cat vecshift.yaml

step "3. Plan it, without changing anything" "vecshift plan"
vecshift plan

step "4. Fill the new column while the table stays in use" "vecshift apply --yes"
vecshift apply --yes

step "5. Compare search quality and latency, old against new" "vecshift eval --yes"
status=0
vecshift eval --yes --html "$out/eval.html" || status=$?
if [ "$status" -ne 0 ]; then
    echo "eval didn't recommend switching (exit $status), so the demo stops before cutover."
    exit 0
fi

step "6. Switch searches to the new vectors" "vecshift cutover --yes"
vecshift cutover --yes

step "7. Check where it stands" "vecshift status"
vecshift status

printf '\nDone. Searches now use the new vectors, under the same column name.\n'
printf 'Reports: demo/out/doctor.html and demo/out/eval.html\n'
printf 'To switch back: docker compose run --rm --entrypoint vecshift vecshift rollback --yes\n'
