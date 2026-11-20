from src.index.sql_db import connect
from src.index import vector_db
d = connect(); q = lambda s: tuple(d.execute(s).fetchone())
print("chunks (total, kept, cameras):", q("select count(*), sum(kept), count(distinct camera) from chunks"))
print("tracks (kind, total, embedded, captioned):", [tuple(r) for r in d.execute("select kind, count(*), sum(attrs_json is not null), sum(caption is not null) from tracks group by kind")])
print("windows (total, embedded, captioned):", q("select count(*), sum(embedded), sum(caption is not null) from windows"))
c = vector_db.client(); print("qdrant:", {n: c.count(n).count for n in vector_db.DIMS})
