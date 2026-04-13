import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .demographics_utils import indices_from_demo_vector, row_to_demo_vector


class DualVideoEFDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        video_root_dir,
        synthetic_root_dir=None,
        video_length=32,
        video_size=128,
        fused=False
    ):
        self.df = pd.read_csv(manifest_path)

        # EF handling
        if "ef" in self.df.columns:
            self.df = self.df.dropna(subset=["ef"]).reset_index(drop=True)
            self.ef_col = "ef"
        elif "EF" in self.df.columns:
            self.df = self.df.dropna(subset=["EF"]).reset_index(drop=True)
            self.ef_col = "EF"
        else:
            raise ValueError("No EF column found")

        self.video_root_dir = video_root_dir
        self.synthetic_root_dir = synthetic_root_dir
        self.video_length = video_length
        self.video_size = video_size
        self.fused = fused

    def __len__(self):
        return len(self.df)

    # =========================
    # VIDEO LOADER
    # =========================
    def load_video(self, path):
        if not os.path.exists(path):
            return None

        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.video_size, self.video_size))
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        frames = np.stack(frames)

        if len(frames) >= self.video_length:
            idx = np.linspace(0, len(frames) - 1, self.video_length).astype(int)
            frames = frames[idx]
        else:
            pad = self.video_length - len(frames)
            frames = np.pad(frames, ((0, pad), (0, 0), (0, 0)), mode="edge")

        frames = frames.astype(np.float32) / 255.0
        return torch.from_numpy(frames).unsqueeze(0)

    # =========================
    # GET ITEM
    # =========================
    def __getitem__(self, idx):

        for _ in range(10):

            row = self.df.iloc[idx]

            if "original_path" in row.index and pd.notna(row.get("original_path", np.nan)):
                real_path = row["original_path"]
            else:
                real_path = row["processed_path"]
            real_video = self.load_video(real_path)

            if self.fused:
                syn_path = row["synthetic_path"]
                syn_video = self.load_video(syn_path)
            else:
                syn_video = None

            if real_video is not None and (not self.fused or syn_video is not None):

                ef = float(row[self.ef_col]) / 100.0
                ef = torch.tensor(ef, dtype=torch.float32)

                demo_np = row_to_demo_vector(row)
                sex, age, bmi = indices_from_demo_vector(demo_np)
                demo_vec = torch.from_numpy(demo_np.copy())

                sex = torch.tensor(sex, dtype=torch.long)
                age = torch.tensor(age, dtype=torch.long)
                bmi = torch.tensor(bmi, dtype=torch.long)

                if not self.fused:
                    return real_video, ef, sex, age, bmi, demo_vec

                return real_video, syn_video, ef, sex, age, bmi, demo_vec

            idx = (idx + 1) % len(self.df)

        raise RuntimeError("Too many missing videos")