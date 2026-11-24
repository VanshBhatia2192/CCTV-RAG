mkdir -p outputs/logs
timeout 150m python -m src.captions.caption_windows > outputs/logs/cap_windows.log 2>&1; echo "windows captions exit $?"; tail -n 2 outputs/logs/cap_windows.log
timeout 480m python -m src.captions.caption_tracks --kind all > outputs/logs/cap_tracks.log 2>&1; echo "track captions exit $?"; tail -n 2 outputs/logs/cap_tracks.log
python - <<'PY'
from src.index.sql_db import connect
d = connect(); d.execute("UPDATE tracks SET caption=REPLACE(caption,' none.','.') WHERE caption LIKE '% none.'"); d.commit()
PY
python -m src.captions.text_index build
python -m src.scale.tally
