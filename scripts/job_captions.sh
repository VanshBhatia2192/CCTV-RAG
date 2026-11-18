python -m src.captions.caption_tracks --kind all > outputs/logs/cap_tracks.log 2>&1 || { echo "CAPTION STEP FAILED"; tail -n 30 outputs/logs/cap_tracks.log; exit 1; }
tail -n 3 outputs/logs/cap_tracks.log
python -m src.captions.text_index build
python -m eval.e2e.eval_activities
