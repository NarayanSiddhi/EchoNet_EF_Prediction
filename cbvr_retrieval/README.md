## Content-Based Video Retrieval (CBVR) – Mini Project

This mini project retrieves the most visually similar echocardiogram videos to a given query using pretrained video embeddings (R3D‑18) and a FAISS index.

### Workflow
1) Extract embeddings
```bash
python cbvr_retrieval/embed_videos.py \
  --video_dir data/processed/videos \
  --output_dir cbvr_retrieval/embeddings \
  --target_frames 32 --target_size 128 --batch_size 8 --device cuda
```

2) Build FAISS index (cosine similarity)
```bash
python cbvr_retrieval/build_faiss.py \
  --emb_dir cbvr_retrieval/embeddings \
  --index_out cbvr_retrieval/faiss_index_cosine.bin \
  --map_out cbvr_retrieval/id_map.json
```

3) Query by video path or id (filename)
```bash
# by path
python cbvr_retrieval/query_retrieval.py \
  --query_path data/processed/videos/XXXX.avi \
  --index cbvr_retrieval/faiss_index_cosine.bin \
  --id_map cbvr_retrieval/id_map.json \
  --topk 5 --device cuda

# by id (filename without path)
python cbvr_retrieval/query_retrieval.py \
  --query_id XXXX.avi \
  --index cbvr_retrieval/faiss_index_cosine.bin \
  --id_map cbvr_retrieval/id_map.json \
  --topk 5 --device cuda
```

### Outputs
- `embeddings/*.npy`: L2‑normalized embedding vectors per video
- `id_map.json`: mapping from integer ids to filenames
- `faiss_index_cosine.bin`: FAISS flat inner‑product index (cosine search)

### Notes
- Uses torchvision R3D‑18 pretrained weights. If weight download fails, script falls back to random‑initialized model with a warning (works for demo, lower quality).
- Videos are loaded as grayscale, resized to `target_size`, uniformly sampled/padded to `target_frames`, then repeated to 3 channels for the backbone.

