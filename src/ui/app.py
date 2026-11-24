import gradio as gr
from src.common.config import PIPE
from src.query.search import search

N = 3

def run(query, rerank, use_llm, sr):
    if not query.strip():
        raise gr.Error("Enter a query.")
    r = search(query, rerank=rerank, use_llm=use_llm, n_clips=N)
    print("[query]", repr(query), r["query"]["type"], r["mode"], r["latency_s"], r["timings"], flush=True)
    res = r["results"]
    vids = [res[i]["clip"] if i < len(res) else None for i in range(N)]
    rows = [[i + 1, x.get("source", x["camera"]), round(x["start_s"], 1), round(x["end_s"], 1), x["score"],
             "" if x["vlm"] is None else round(x["vlm"], 2), x["caption"] or ""] for i, x in enumerate(res)]
    info = f'type: {r["query"]["type"]} | text: {r["query"]["text"]} | camera: {r["query"]["camera"]} | latency: {r["latency_s"]} s | {r["timings"]}'
    if sr:
        info += " | SR not wired yet (original shown)"
    return [info, rows] + vids

with gr.Blocks(title="CCTV Search") as demo:
    gr.Markdown("## Natural-language CCTV search\nVehicles, people (by appearance and motion), incidents. No face recognition.")
    q = gr.Textbox(label="Query", placeholder="a person in a red jacket carrying a backpack")
    with gr.Row():
        rerank = gr.Checkbox(False, label="VLM re-rank (slow)")
        use_llm = gr.Checkbox(True, label="LLM query parser")
        sr = gr.Checkbox(False, label="Super resolution (final clip)")
        gait = gr.Checkbox(False, label="Gait (off, coming later)", interactive=False)
    btn = gr.Button("Search", variant="primary")
    info = gr.Markdown()
    with gr.Row():
        vids = [gr.Video(label=f"Result {i + 1}") for i in range(N)]
    table = gr.Dataframe(headers=["rank", "camera", "start_s", "end_s", "retrieval", "vlm", "caption"], wrap=True)
    gr.Examples(["a person in a red jacket", "a white car turning left", "two people fighting",
                 "a person running", "a truck stopped near the entrance"], inputs=q)
    btn.click(run, [q, rerank, use_llm, sr], [info, table] + vids)
    q.submit(run, [q, rerank, use_llm, sr], [info, table] + vids)

if __name__ == "__main__":
    try:
        search("a person walking", rerank=False, use_llm=True, n_clips=1); print("warm-up done", flush=True)
    except Exception as e:
        print("warm-up failed:", e, flush=True)
    demo.launch(server_port=PIPE["ui"]["port"])
