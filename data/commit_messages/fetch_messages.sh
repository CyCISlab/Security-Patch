#!/bin/bash
# Downloads the commit header (message) for every commit in urls_core.tsv
# (and urls_extra.tsv if you pass "extra"). Safe to stop and rerun: it skips
# commits already downloaded. Run from this folder:
#     bash fetch_messages.sh            # core list (~7,700 commits, ~45-60 min)
#     bash fetch_messages.sh extra      # the remaining ~10,700 commits (optional)
cd "$(dirname "$0")"
LIST=urls_core.tsv; [ "$1" = "extra" ] && LIST=urls_extra.tsv
mkdir -p messages
total=$(($(wc -l < "$LIST") - 1)); n=0
tail -n +2 "$LIST" | while IFS=$'\t' read -r repo hash url; do
  n=$((n+1)); out="messages/$hash.txt"
  [ -s "$out" ] && continue
  for try in 1 2 3; do
    # keep only the first 6 KB: the header with the commit message, not the full diff
    curl -sL --max-time 20 "$url" 2>/dev/null | head -c 6000 > "$out.tmp"
    if grep -q "^Subject:" "$out.tmp"; then mv "$out.tmp" "$out"; break; fi
    rm -f "$out.tmp"; sleep $((try*5))
  done
  [ -s "$out" ] || echo -e "$repo\t$hash" >> failed.tsv
  [ $((n % 200)) -eq 0 ] && echo "$n / $total done"
  sleep 0.3
done
echo "Finished. Downloaded: $(ls messages | wc -l). Failed: $( [ -f failed.tsv ] && wc -l < failed.tsv || echo 0)"
