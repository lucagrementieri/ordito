#!/usr/bin/env bash
# Survey a Warp release before touching the lock: changelog section, release notes, closed
# milestone, and the commit range filtered by the Warp files ordito depends on.
#
#   release_survey.sh OLD NEW [OUT_DIR]     e.g.  release_survey.sh 1.17.0 1.18.0
#
# Writes into OUT_DIR (default: ./warp-survey-NEW):
#   changelog.md       the `## [NEW]` section of CHANGELOG.md at tag vNEW
#   release.md         the GitHub release body
#   milestone.tsv      number / IS|PR / title of every closed item in the milestone titled NEW
#   commits.txt        `git log --oneline vOLD..vNEW`
#   surface.txt        per watched file: commit count in the range, then its commits
#   warp-src/          a blobless clone, for `git diff vOLD vNEW -- <file>` on demand
set -euo pipefail
OLD=${1:?old version, e.g. 1.17.0}
NEW=${2:?new version, e.g. 1.18.0}
OUT=${3:-./warp-survey-$NEW}
mkdir -p "$OUT"
cd "$OUT"

curl -sfL "https://raw.githubusercontent.com/NVIDIA/warp/v$NEW/CHANGELOG.md" -o CHANGELOG.full.md
awk -v v="$NEW" '$0 ~ "^## \\[" v "\\]" {on=1; print; next} on && /^## \[/ {exit} on {print}' \
    CHANGELOG.full.md > changelog.md
gh api "repos/NVIDIA/warp/releases/tags/v$NEW" --jq '.body' > release.md

MILESTONE=$(gh api 'repos/NVIDIA/warp/milestones?state=all&per_page=100' \
    --jq ".[] | select(.title == \"$NEW\" or .title == \"${NEW%.0}\") | .number" | head -1)
if [ -n "$MILESTONE" ]; then
    gh api --paginate "repos/NVIDIA/warp/issues?milestone=$MILESTONE&state=closed&per_page=100" \
        --jq '.[] | "\(.number)\t\(if .pull_request then "PR" else "IS" end)\t\(.title)"' \
        > milestone.tsv
else
    echo "no milestone titled $NEW" > milestone.tsv
fi

[ -d warp-src ] || git clone -q --filter=blob:none --no-checkout https://github.com/NVIDIA/warp.git warp-src
git -C warp-src fetch -q --tags
git -C warp-src log --oneline "v$OLD..v$NEW" > commits.txt

# The Warp files ordito's behaviour or private access rests on (see SKILL.md, "dependency surface").
WATCHED=(
    warp/_src/context.py warp/_src/codegen.py warp/_src/types.py warp/_src/utils.py
    warp/_src/sparse.py warp/_src/tape.py warp/_src/marching_cubes.py
    warp/native/builtin.h warp/native/bvh.h warp/native/mesh.h warp/native/hashgrid.h
    warp/native/tile.h warp/native/tile_reduce.h warp/native/range.h warp/native/volume.h
    warp/native/sort.cu warp/native/scan.cu warp/native/warp.cu
    warp/optim/linear.py warp/_src/optim/linear.py warp/_src/fem warp/_src/geometry
)
: > surface.txt
for f in "${WATCHED[@]}"; do
    n=$(git -C warp-src log --oneline "v$OLD..v$NEW" -- "$f" | wc -l)
    [ "$n" -eq 0 ] && continue
    printf '\n== %s: %s commits\n' "$f" "$n" >> surface.txt
    git -C warp-src log --oneline "v$OLD..v$NEW" -- "$f" >> surface.txt
done

echo "survey in $(pwd):"
wc -l changelog.md milestone.tsv commits.txt surface.txt
grep -E '^### ' changelog.md
