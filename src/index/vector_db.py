import os
from pathlib import Path
from qdrant_client import QdrantClient, models
from src.common.config import PATHS

DIMS = {"vehicle": 512, "vehicle_ft": 512, "person": 512, "person_ft": 512, "incident": 512, "person_gait": 4096, "text_person": 384, "text_vehicle": 384, "text_incident": 384}
_client = None
EUCLID = {"person_gait"}   # OpenGait embeddings use Euclidean distance (score = distance, lower is better)

def client():
    global _client
    if _client is None:
        Path(PATHS["qdrant_path"]).mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(url=os.getenv("QDRANT_URL"), timeout=60.0) if os.getenv("QDRANT_URL") else QdrantClient(path=PATHS["qdrant_path"])
        have = {c.name for c in _client.get_collections().collections}
        for name, d in DIMS.items():
            if name not in have:
                _client.create_collection(name, vectors_config=models.VectorParams(
                    size=d, distance=models.Distance.EUCLID if name in EUCLID else models.Distance.COSINE))
    return _client

def recreate(name):
    c = client()
    if name in {x.name for x in c.get_collections().collections}:
        c.delete_collection(name)
    c.create_collection(name, vectors_config=models.VectorParams(
        size=DIMS[name], distance=models.Distance.EUCLID if name in EUCLID else models.Distance.COSINE))

def upsert(collection, ids, vectors, payloads):
    client().upsert(collection, points=[
        models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)])

def search(collection, vector, k=50, where=None, ids=None):
    must = []
    if where:
        must += [models.FieldCondition(key=k_, match=models.MatchValue(value=v)) for k_, v in where.items()]
    if ids is not None:
        must.append(models.HasIdCondition(has_id=list(ids)))
    flt = models.Filter(must=must) if must else None
    return client().query_points(collection, query=vector, limit=k, query_filter=flt).points
