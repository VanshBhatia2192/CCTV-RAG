from src.index.sql_db import connect
from src.captions import hybrid
Q = [("vehicle", "hybrid", q) for q in ["a white car turning left", "a black SUV", "a red truck", "a car that is stopped"]] + \
    [("person", "hybrid_ft", q) for q in ["a person in a red jacket", "a person carrying a backpack", "a person wearing blue jeans and a black jacket", "a person running"]]
db = connect()
for kind, mode, q in Q:
    for boost in (False, True):
        print(f"\n[{kind}] {q!r}  boost={boost}")
        for h in hybrid.retrieve(kind, q, 3, mode, boost=boost):
            c = db.execute("select caption from tracks where id=?", (h["id"],)).fetchone()[0]
            print("   ", round(h["score"], 4), "|", (c or "")[:95])
