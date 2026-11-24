python -m eval.person.eval_rstp || echo "RSTP EVAL FAILED"
python -m eval.e2e.eval_activities --modes visual || echo "ACTIVITY EVAL FAILED"
