mkdir -p outputs/logs
python -m src.incident.windows
python -m src.scale.embed_all --workers 6 > outputs/logs/embed_all.log 2>&1 &
python -m src.scale.embed_windows --workers 4 > outputs/logs/embed_windows.log 2>&1 &
wait
for f in embed_all embed_windows; do echo "== $f"; tail -n 3 outputs/logs/$f.log; done
python -m src.scale.tally
