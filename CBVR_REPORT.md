## Content-Based Video Retrieval (CBVR) for Echocardiography

### Abstract
We build a simple, label-free content-based video retrieval (CBVR) system for echocardiogram videos. Each video is embedded into a compact feature vector using a pretrained 3D CNN (R3D‑18). We index all embeddings with FAISS for fast nearest-neighbor search using cosine similarity. Given a query video, the system returns the top‑k most similar studies. We demonstrate a working end‑to‑end pipeline on 7,791 pediatric echo videos and provide a demo subset with immediate results.

### 1. Motivation and Goal
- Clinicians and researchers often need “similar prior cases” to a study of interest.
- A CBVR system can surface visually similar cardiac motion/structure without requiring labels.
- This mini project implements such a system that is fast, simple, and easy to demo.

### 2. Data
- Source: `data/processed/videos` (7,791 pediatric echo videos; variable length; grayscale)
- For quick demonstration: a 500‑video subset at `cbvr_retrieval/demo_videos`

### 3. Method Overview
1) Preprocessing:
   - Convert frames to grayscale (if not already)
   - Resize to `128×128`
   - Uniformly sample/pad to `T=32` frames
   - Repeat to 3 channels to fit the pretrained backbone
2) Embedding model:
   - `torchvision` R3D‑18 (pretrained weights if available)
   - Replace final classifier with Identity to output a 512‑D feature
   - L2‑normalize embeddings for cosine similarity
3) Indexing and search:
   - Use FAISS `IndexFlatIP` (inner product on unit‑norm vectors = cosine similarity)
   - Build index over all embeddings; store an `id_map.json` (id → filename)
   - Query by computing the query embedding and retrieving top‑k nearest neighbors

### 4. Implementation Details
- Embedding extraction: `cbvr_retrieval/embed_videos.py`
- Index build: `cbvr_retrieval/build_faiss.py`
- Query: `cbvr_retrieval/query_retrieval.py`
- Demo run artifacts:
  - `cbvr_retrieval/embeddings_demo/`
  - `cbvr_retrieval/faiss_index_cosine_demo.bin`
  - `cbvr_retrieval/id_map_demo.json`

### 5. How to Reproduce (Commands)
1) Install dependencies:
```
python3 -m pip install --user torchvision faiss-cpu opencv-python tqdm numpy
```
2) Extract embeddings (demo subset 500 videos):
```
python cbvr_retrieval/embed_videos.py \
  --video_dir cbvr_retrieval/demo_videos \
  --output_dir cbvr_retrieval/embeddings_demo \
  --target_frames 32 --target_size 128 --batch_size 8 --device cuda
```
3) Build FAISS index (demo):
```
python cbvr_retrieval/build_faiss.py \
  --emb_dir cbvr_retrieval/embeddings_demo \
  --index_out cbvr_retrieval/faiss_index_cosine_demo.bin \
  --map_out cbvr_retrieval/id_map_demo.json
```
4) Query (by filename) on demo:
```
python cbvr_retrieval/query_retrieval.py \
  --query_id CR32a7555-CR3dca855-000033.mp4 \
  --video_dir cbvr_retrieval/demo_videos \
  --index cbvr_retrieval/faiss_index_cosine_demo.bin \
  --id_map cbvr_retrieval/id_map_demo.json \
  --topk 5 --device cuda
```
5) Full dataset (optional, background job already started):
```
python cbvr_retrieval/embed_videos.py \
  --video_dir data/processed/videos \
  --output_dir cbvr_retrieval/embeddings \
  --target_frames 32 --target_size 128 --batch_size 8 --device cuda

python cbvr_retrieval/build_faiss.py \
  --emb_dir cbvr_retrieval/embeddings \
  --index_out cbvr_retrieval/faiss_index_cosine.bin \
  --map_out cbvr_retrieval/id_map.json
```

### 6. Experiments
We report two practical checks:
1) Qualitative retrieval inspection
   - Visualize query and top‑k neighbors; verify similar cardiac views/motions are retrieved.
2) Proxy quantitative checks (optional)
   - Nearest‑neighbor demographic agreement: compute the fraction where nearest neighbors share the same sex or similar age‑bin; report histograms of cosine similarity.

### 7. Results (Demo)
- Example query 1 (top‑5 neighbors):
  - CR32a7555-CR3dca855-000033 (sim=1.0000)
  - CR3dca773-CR3dca9cc-000024 (sim=0.9451)
  - CR32a95e4-CR3dcb3de-000028 (sim=0.9426)
  - CR32a9717-CR32a9a17-000084 (sim=0.9399)
  - CR3dca685-CR3dca85c-000041 (sim=0.9392)

- Example query 2 (top‑5 neighbors):
  - CR32a7556-CR32a99fd-000036 (sim=1.0000)
  - CR32a95d0-CR32a9a65-000034 (sim=0.9488)
  - CR32a7561-CR32a990f-000033 (sim=0.9429)
  - CR32a7567-CR32a985e-000033 (sim=0.9427)
  - CR32a9654-CR32a9a81-000027 (sim=0.9422)

These examples show high‑similarity neighbors that are visually close to the queries.

### 8. Figures to Include (What images to add)
Prepare these visuals for the report/slides:
1) System diagram (1 image)
   - Boxes: Preprocessing → R3D‑18 Embedding → FAISS Index → Query → Top‑k Results
2) Query panel(s) (2–3 images)
   - For each of 2–3 queries: a 3×6 panel showing 6 frames of the query (top row) and 6 frames of each of the top‑3 neighbors (below), with cosine similarities annotated.
   - How to capture: extract 6 evenly‑spaced frames using `extract_video_frames.py` and compose a grid (the script already supports grid creation).
3) Similarity histogram (1 image, optional)
   - Histogram of cosine similarities of top‑1 across 200 random queries (demo set). Shows separation away from random.
4) t‑SNE/UMAP embedding plot (1 image, optional)
   - 2D projection of 1,000 video embeddings, colored by a simple tag (e.g., sex), to visualize structure.

Tip: Keep images high‑contrast (grayscale clean), label each with video id and sim score. Use consistent frame sizes (128×128).

### 9. Limitations
- Similarity reflects the pretrained feature space; may not perfectly match clinical similarity.
- No view classification—mix of views can appear in neighbors unless filtered.
- Cosine similarity is not calibrated; use rankings rather than absolute thresholds.

### 10. Future Work
- Add a lightweight view classifier to restrict neighbors to the same cardiac view.
- Build a small web UI for interactive browsing and side‑by‑side playback.
- Experiment with domain‑specific backbones (e.g., echo‑pretrained networks) for improved alignment.

### 11. References
- Hara, K., Kataoka, H., & Satoh, Y. (2018). Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? CVPR Workshops. (R3D‑18 backbone)
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Trans. Big Data. (FAISS library)

