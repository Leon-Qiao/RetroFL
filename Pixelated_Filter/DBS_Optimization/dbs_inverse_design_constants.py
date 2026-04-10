from __future__ import annotations

import math
import random
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["white", "red", "blue"])

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = object
    F = None

# =========================================================
# User constants: edit here directly
# =========================================================
USE_SURROGATE = False          # True -> surrogate, False -> ADS
CREATE_FINAL_LAYOUT = True    # Only meaningful when ADS env is available

# search settings
N_RESTARTS = 8
MAX_OUTER_ITERS = 16
SHUFFLE_SEED = 42
TARGET_LOSS = 0.02
MIN_IMPROVEMENT_PER_ROUND = 1e-4
STRICT_CONNECTIVITY = True
STRICT_MANHATTAN_CLEAN = False
ALLOW_TOGGLE_ZERO_ONE = True
ALLOW_TOGGLE_ONE_TWO = False   # superseded by edge-row 0->1 special rule below
ALLOW_TOGGLE_TWO_ONE = False    # user requested 2 toggles back to 0 directly
ALLOW_TOGGLE_TWO_ZERO = True
EDGE_NEW_ONE_TO_TWO_PROB = 0.75  # when first/last row 0->1, chance to turn it into the row's new 2

# initialization
INIT_FROM_FILE = None  # e.g. "./my_grid.npy"
INIT_RANDOM_RATIO = 0.45
INIT_RANDOM_INCLUDE_VIA = True
MAX_INIT_TRIES = 5000

# files / checkpoint
CHECKPOINT_PATH = "./RetroFL_surrogate.pt"
OUTPUT_DIR = "./dbs_run"
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

# same geometry defaults as PIXELED_diag_sampling.py
N_ROWS = 16
N_COLS = 16
CELL = 0.65
DIRS8 = [(-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1)]
SIMU_FREQ_LOW_GHZ = 0.1
SIMU_FREQ_HIGH_GHZ = 6.0
SIMU_POINTS = 256

# objective, copied from PIXELED_diag_sampling.py defaults
F_PASS_LO_GHZ = 2
F_PASS_HI_GHZ = 3
F_STOP1_LO_GHZ = 0.1
F_STOP1_HI_GHZ = 1.5
F_STOP2_LO_GHZ = 3.5
F_STOP2_HI_GHZ = 6
A_P = 0.891
A_R = 0.316
A_S = 0.1

# surrogate freq query size, consistent with training.py default FrequencyEmbedding(num_freqs=120)
N_SURROGATE_FREQ = 120

# =========================================================
# Optional ADS imports
# =========================================================
ADS_AVAILABLE = False
try:
    import keysight.edatoolbox.multi_python as multi_python
    from autoem.ads_tools import ads_create_layout, ads_delete_rfpro_view
    from autoem.xxpro_tools import xxpro_run_simulation
    import keysight.ads.dataset as dataset
    ADS_AVAILABLE = True
except Exception:
    multi_python = None
    ads_create_layout = None
    ads_delete_rfpro_view = None
    xxpro_run_simulation = None
    dataset = None


# =========================================================
# Surrogate model copied/adapted from training.py
# =========================================================
class FrequencyEmbedding(nn.Module):
    def __init__(self, num_freqs=120, embed_dim=64):
        super().__init__()
        # 使用极其稳定的离散查表法，字典大小为120，映射到64维
        self.embed = nn.Embedding(num_freqs, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, freq_idx):
        # freq_idx 的形状现在是 [B, Nf]
        x = self.embed(freq_idx)       # 出来自动变成 [B, Nf, 64]
        x = self.norm(x)
        x = self.mlp(x)                # [B, Nf, 64]
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, base_channels=64, out_dim=256):
        super().__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.ReLU(inplace=True),
        )

        # stages（类似 ResNet18 但更轻）
        self.layer1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(base_channels * 2, out_dim, stride=2),
            ResidualBlock(out_dim, out_dim),
            ResidualBlock(out_dim, out_dim),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)       # [B,64,16,16]
        x = self.layer1(x)     # [B,64,16,16]
        x = self.layer2(x)     # [B,128,8,8]
        x = self.layer3(x)     # [B,256,4,4]
        x = self.pool(x)       # [B,256,1,1]
        x = x.flatten(1)       # [B,256]
        return x

class ResNetFreqSurrogate(nn.Module):
    def __init__(
        self,
        freq_dim=64,
        structure_dim=256,
        hidden_dim=512,
        dropout=0.3,
        num_freqs=120,
    ):
        super().__init__()

        self.feature_extractor = ResNetFeatureExtractor(
            base_channels=64,
            out_dim=structure_dim,
        )

        self.freq_embed = FrequencyEmbedding(num_freqs=num_freqs, embed_dim=freq_dim)

        self.head = nn.Sequential(
            nn.Linear(structure_dim + freq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 4),
        )

        self.out_act = nn.Tanh()

    def forward(self, matrix, freq):
        struct_feat = self.feature_extractor(matrix)   # [B,256]
        freq_feat = self.freq_embed(freq)              # [B,Nf,64]

        B, Nf, Df = freq_feat.shape
        Ds = struct_feat.shape[1]

        struct_feat = struct_feat.unsqueeze(1).expand(B, Nf, Ds)
        fusion = torch.cat([struct_feat, freq_feat], dim=-1)

        out = self.head(fusion)        # [B,Nf,4]
        out = self.out_act(out)

        return out.transpose(1, 2)

# =========================================================
# Utilities
# =========================================================
def ensure_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.uint8)
    if g.shape != (N_ROWS, N_COLS):
        raise ValueError(f"grid shape must be {(N_ROWS, N_COLS)}, got {g.shape}")
    bad = np.setdiff1d(np.unique(g), np.array([0, 1, 2], dtype=np.uint8))
    if len(bad) > 0:
        raise ValueError(f"grid contains invalid values: {bad.tolist()} ; only 0/1/2 allowed")
    return g


def get_mid_two_rows() -> Tuple[int, int]:
    return N_ROWS // 2 - 1, N_ROWS // 2


def conductive_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, [1, 2])


def remove_isolated_pixels_by_manhattan_rule(grid: np.ndarray, keep_points=None) -> np.ndarray:
    grid = ensure_grid(grid).copy()
    keep_points = set() if keep_points is None else set(keep_points)
    conductive_coords = np.argwhere((grid == 1) | (grid == 2))
    to_remove = []
    for r, c in conductive_coords:
        if (int(r), int(c)) in keep_points:
            continue
        has_neighbor = False
        for rr, cc in conductive_coords:
            if rr == r and cc == c:
                continue
            if abs(int(rr) - int(r)) + abs(int(cc) - int(c)) < 3:
                has_neighbor = True
                break
        if not has_neighbor:
            to_remove.append((int(r), int(c)))
    for r, c in to_remove:
        grid[r, c] = 0
    return grid


def get_conductive_neighbors_8(grid: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    nbrs = []
    for dr, dc in DIRS8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < N_ROWS and 0 <= cc < N_COLS and grid[rr, cc] in (1, 2):
            nbrs.append((rr, cc))
    return nbrs


def is_mid_connected_dfs(grid: np.ndarray) -> Tuple[bool, Tuple[int, int], np.ndarray]:
    grid = ensure_grid(grid)
    mid_r1, mid_r2 = get_mid_two_rows()
    starts = [(mid_r1, 0), (mid_r2, 0)]
    goals = {(mid_r1, N_COLS - 1), (mid_r2, N_COLS - 1)}
    valid_starts = [p for p in starts if grid[p] in (1, 2)]
    if not valid_starts:
        return False, (mid_r1, mid_r2), grid
    if not any(grid[g] in (1, 2) for g in goals):
        return False, (mid_r1, mid_r2), grid
    visited = np.zeros((N_ROWS, N_COLS), dtype=bool)
    stack = valid_starts.copy()
    for s in valid_starts:
        visited[s] = True
    while stack:
        r, c = stack.pop()
        if (r, c) in goals:
            return True, (mid_r1, mid_r2), grid
        for rr, cc in get_conductive_neighbors_8(grid, r, c):
            if not visited[rr, cc]:
                visited[rr, cc] = True
                stack.append((rr, cc))
    return False, (mid_r1, mid_r2), grid


def enforce_port_pixels(grid: np.ndarray) -> np.ndarray:
    grid = ensure_grid(grid).copy()
    r1, r2 = get_mid_two_rows()
    grid[r1, 0] = max(grid[r1, 0], 1)
    grid[r2, 0] = max(grid[r2, 0], 1)
    grid[r1, N_COLS - 1] = max(grid[r1, N_COLS - 1], 1)
    grid[r2, N_COLS - 1] = max(grid[r2, N_COLS - 1], 1)
    return grid


def repair_grid(grid: np.ndarray) -> np.ndarray:
    port_points = {
        (get_mid_two_rows()[0], 0), (get_mid_two_rows()[1], 0),
        (get_mid_two_rows()[0], N_COLS - 1), (get_mid_two_rows()[1], N_COLS - 1),
    }
    grid = enforce_port_pixels(grid)
    if STRICT_MANHATTAN_CLEAN:
        grid = remove_isolated_pixels_by_manhattan_rule(grid, keep_points=port_points)
    return grid


def _enforce_edge_row_single_via(grid: np.ndarray) -> np.ndarray:
    grid = ensure_grid(grid).copy()
    for row in (0, N_ROWS - 1):
        via_cols = np.where(grid[row] == 2)[0]
        if len(via_cols) <= 1:
            continue
        keep = int(via_cols[0])
        for c in via_cols[1:]:
            grid[row, int(c)] = 1
        grid[row, keep] = 2
    return grid


def random_initial_grid(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_INIT_TRIES):
        base = (rng.random((N_ROWS, N_COLS)) < INIT_RANDOM_RATIO).astype(np.uint8)
        if INIT_RANDOM_INCLUDE_VIA:
            top_choices = np.where(base[0] == 1)[0]
            bot_choices = np.where(base[-1] == 1)[0]
            if len(top_choices) > 0 and rng.random() < EDGE_NEW_ONE_TO_TWO_PROB:
                chosen = int(rng.choice(top_choices))
                base[0, chosen] = 2
            if len(bot_choices) > 0 and rng.random() < EDGE_NEW_ONE_TO_TWO_PROB:
                chosen = int(rng.choice(bot_choices))
                base[-1, chosen] = 2
        base = _enforce_edge_row_single_via(base)
        base = repair_grid(base)
        ok, _, cleaned = is_mid_connected_dfs(base)
        if ok:
            return cleaned
    raise RuntimeError("failed to sample a connected initial grid")


def load_init_grid() -> np.ndarray:
    if INIT_FROM_FILE:
        grid = np.load(INIT_FROM_FILE)
        grid = _enforce_edge_row_single_via(grid)
        grid = repair_grid(grid)
        ok, _, _ = is_mid_connected_dfs(grid)
        if STRICT_CONNECTIVITY and not ok:
            raise ValueError("INIT_FROM_FILE is not mid-connected")
        return grid
    rng = np.random.default_rng(SHUFFLE_SEED)
    return random_initial_grid(rng)


def toggle_candidates_for_value(v: int) -> List[int]:
    outs: List[int] = []
    if v == 0 and ALLOW_TOGGLE_ZERO_ONE:
        outs.append(1)
    elif v == 1:
        if ALLOW_TOGGLE_ZERO_ONE:
            outs.append(0)
        if ALLOW_TOGGLE_ONE_TWO:
            outs.append(2)
    elif v == 2:
        if ALLOW_TOGGLE_TWO_ONE:
            outs.append(1)
        if ALLOW_TOGGLE_TWO_ZERO:
            outs.append(0)
    return outs


def apply_single_cell_move(grid: np.ndarray, r: int, c: int, requested_new_value: int, rng: np.random.Generator) -> np.ndarray:
    candidate = ensure_grid(grid).copy()
    old_value = int(candidate[r, c])

    if old_value == requested_new_value:
        return candidate

    # user rule: 2 flips directly back to 0
    if old_value == 2:
        candidate[r, c] = 0
        return _enforce_edge_row_single_via(candidate)

    candidate[r, c] = requested_new_value

    # special rule for first/last row: when 0->1, with 50% probability make it the row's new 2
    if old_value == 0 and requested_new_value == 1 and r in (0, N_ROWS - 1):
        if rng.random() < EDGE_NEW_ONE_TO_TWO_PROB:
            candidate[r, candidate[r] == 2] = 1
            candidate[r, c] = 2

    return _enforce_edge_row_single_via(candidate)


def all_single_cell_moves(grid: np.ndarray, rng: np.random.Generator) -> List[Tuple[int, int, int]]:
    moves = []
    port_cells = {(get_mid_two_rows()[0], 0), (get_mid_two_rows()[1], 0), (get_mid_two_rows()[0], N_COLS - 1), (get_mid_two_rows()[1], N_COLS - 1)}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if (r, c) in port_cells:
                continue
            v = int(grid[r, c])
            if v == 2:
                allowed = [0] if ALLOW_TOGGLE_TWO_ZERO else []
            elif v == 0:
                allowed = [1] if ALLOW_TOGGLE_ZERO_ONE else []
            elif v == 1:
                allowed = [0] if ALLOW_TOGGLE_ZERO_ONE else []
            else:
                allowed = []

            if (r, c) in port_cells:
                allowed = [nv for nv in allowed if nv in (1, 2)]

            for nv in allowed:
                moves.append((r, c, nv))
    rng.shuffle(moves)
    return moves


# =========================================================
# Geometry generation from PIXELED_diag_sampling.py logic
# =========================================================
def add_state(state5, state14, r, c, s):
    if not (0 <= r < N_ROWS and 0 <= c < N_COLS):
        return
    if s == 5:
        state5[r, c] = True
        state14[r][c].clear()
    else:
        if state5[r, c]:
            return
        state14[r][c].add(s)


def triangle_from_state_in_cell(r, c, s):
    x0 = c * CELL
    y0 = r * CELL
    tl = (x0, y0)
    tr = (x0 + CELL, y0)
    bl = (x0, y0 + CELL)
    br = (x0 + CELL, y0 + CELL)
    if s == 1:
        pts = [tl, tr, br]
    elif s == 2:
        pts = [tl, tr, bl]
    elif s == 3:
        pts = [tl, bl, br]
    elif s == 4:
        pts = [tr, bl, br]
    else:
        raise ValueError(f"illegal state {s}")
    return [(round(x, 4), round(y, 4)) for x, y in pts]


def four_small_triangles_in_center_cell(r, c):
    x0 = c * CELL
    y0 = r * CELL
    h = CELL / 2.0
    tl = (x0, y0)
    tr = (x0 + CELL, y0)
    bl = (x0, y0 + CELL)
    br = (x0 + CELL, y0 + CELL)
    top_mid = (x0 + h, y0)
    bottom_mid = (x0 + h, y0 + CELL)
    left_mid = (x0, y0 + h)
    right_mid = (x0 + CELL, y0 + h)
    tri_tl = [tl, top_mid, left_mid]
    tri_tr = [tr, top_mid, right_mid]
    tri_bl = [bl, left_mid, bottom_mid]
    tri_br = [br, right_mid, bottom_mid]
    return [[(round(x, 4), round(y, 4)) for x, y in tri] for tri in [tri_tl, tri_tr, tri_bl, tri_br]]


def build_anti_aliasing_triangles_from_2x2_for_conductors(grid: np.ndarray):
    grid = ensure_grid(grid)
    def is_cond(v):
        return int(v in (1, 2))
    state5 = np.zeros((N_ROWS, N_COLS), dtype=bool)
    state14 = [[set() for _ in range(N_COLS)] for _ in range(N_ROWS)]
    for r in range(1, N_ROWS - 1):
        for c in range(1, N_COLS - 1):
            if grid[r, c] != 0:
                continue
            up = is_cond(grid[r - 1, c]); down = is_cond(grid[r + 1, c])
            left = is_cond(grid[r, c - 1]); right = is_cond(grid[r, c + 1])
            if up == 1 and down == 1 and left == 1 and right == 1:
                add_state(state5, state14, r, c, 5)
    for r in range(N_ROWS - 1):
        for c in range(N_COLS - 1):
            a = is_cond(grid[r, c]); b = is_cond(grid[r, c + 1]); c_ = is_cond(grid[r + 1, c]); d = is_cond(grid[r + 1, c + 1])
            s = a + b + c_ + d
            if a == 1 and b == 0 and c_ == 0 and d == 1:
                add_state(state5, state14, r, c + 1, 3)
                add_state(state5, state14, r + 1, c, 1)
            elif a == 0 and b == 1 and c_ == 1 and d == 0:
                add_state(state5, state14, r, c, 4)
                add_state(state5, state14, r + 1, c + 1, 2)
            elif s == 3:
                if a == 0:
                    add_state(state5, state14, r, c, 4)
                elif b == 0:
                    add_state(state5, state14, r, c + 1, 3)
                elif c_ == 0:
                    add_state(state5, state14, r + 1, c, 1)
                elif d == 0:
                    add_state(state5, state14, r + 1, c + 1, 2)
    polygons = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if state5[r, c]:
                polygons.extend(four_small_triangles_in_center_cell(r, c))
            else:
                for s in sorted(state14[r][c]):
                    polygons.append(triangle_from_state_in_cell(r, c, s))
    return polygons


def get_shapes(grid: np.ndarray, pair: Optional[Tuple[int, int]] = None):
    grid = ensure_grid(grid)
    rectangles = []
    circles = []
    polygons = build_anti_aliasing_triangles_from_2x2_for_conductors(grid)
    width = round(N_COLS * CELL, 4)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if grid[r, c] not in (1, 2):
                continue
            x1 = round(c * CELL, 4); y1 = round(r * CELL, 4)
            x2 = round(x1 + CELL, 4); y2 = round(y1 + CELL, 4)
            rectangles.append(((x1, y1), (x2, y2)))
            if grid[r, c] == 2:
                xc = round(x1 + CELL / 2.0, 4); yc = round(y1 + CELL / 2.0, 4)
                radius = round(CELL / 3.0, 4)
                circles.append(((xc, yc), radius))
    left_r1, left_r2 = get_mid_two_rows()
    right_r1, right_r2 = get_mid_two_rows()
    for rr in [left_r1, left_r2]:
        x1 = round(-CELL, 4); y1 = round(rr * CELL, 4)
        x2 = round(0.0, 4); y2 = round(y1 + CELL, 4)
        rectangles.append(((x1, y1), (x2, y2)))
    for rr in [right_r1, right_r2]:
        x1 = round(width, 4); y1 = round(rr * CELL, 4)
        x2 = round(width + CELL, 4); y2 = round(y1 + CELL, 4)
        rectangles.append(((x1, y1), (x2, y2)))
    y_center = ((left_r1 + 0.5) * CELL + (left_r2 + 0.5) * CELL) / 2.0
    centers = ((round(-CELL, 4), round(y_center, 4)), (round(width + CELL, 4), round(y_center, 4)))
    return rectangles, polygons, circles, centers


def draw_ads_layout(ax, rects, polys, circles, margin_cells=1.0):
    width = N_COLS * CELL
    height = N_ROWS * CELL
    pad = margin_cells * CELL
    ax.set_facecolor("#f2f2f2")
    for i in range(N_COLS + 1):
        x = i * CELL
        ax.plot([x, x], [0, height], color="#666666", linewidth=0.7, zorder=1)
    for j in range(N_ROWS + 1):
        y = j * CELL
        ax.plot([0, width], [y, y], color="#666666", linewidth=0.7, zorder=1)
    for ((x1, y1), (x2, y2)) in rects:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="#f48a8a", edgecolor="#444444", linewidth=0.6, zorder=3))
    for poly in polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#ff0000", edgecolor="none", zorder=4))
    for center, radius in circles:
        ax.add_patch(Circle(center, radius=radius, facecolor="#2b6cb0", edgecolor="none", zorder=5))
    ax.set_xlim(-CELL - pad, width + CELL + pad)
    ax.set_ylim(-pad, height + pad)
    ax.set_aspect("equal")
    ax.axis("off")


# =========================================================
# Evaluation
# =========================================================
def compute_loss_from_arrays(freq_hz, s11_complex, s21_complex):
    freq = np.asarray(freq_hz)
    s11 = np.asarray(s11_complex)
    s21 = np.asarray(s21_complex)
    bandpass_mask = (freq >= F_PASS_LO_GHZ * 1e9) & (freq <= F_PASS_HI_GHZ * 1e9)
    bandstop_mask = ((freq >= F_STOP1_LO_GHZ * 1e9) & (freq <= F_STOP1_HI_GHZ * 1e9)) | ((freq >= F_STOP2_LO_GHZ * 1e9) & (freq <= F_STOP2_HI_GHZ * 1e9))
    mag_s11 = np.abs(s11)
    mag_s21 = np.abs(s21)
    P_IL = np.maximum(A_P - mag_s21[bandpass_mask], 0).mean() if np.any(bandpass_mask) else 1e6
    P_RL = np.maximum(mag_s11[bandpass_mask] - A_R, 0).mean() if np.any(bandpass_mask) else 1e6
    P_SB = np.maximum(mag_s21[bandstop_mask] - A_S, 0).mean() if np.any(bandstop_mask) else 1e6
    J = float(P_IL + P_RL + P_SB)
    return J, float(P_IL), float(P_RL), float(P_SB), mag_s11, mag_s21

class SurrogateEvaluator:
    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch not available, cannot use surrogate")
        self.freq_hz = np.linspace(SIMU_FREQ_LOW_GHZ, SIMU_FREQ_HIGH_GHZ, N_SURROGATE_FREQ, dtype=np.float32) * 1e9
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.model = ResNetFreqSurrogate().to(DEVICE)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, grid: np.ndarray) -> Dict:
        grid = ensure_grid(grid)
        # important: training.py learned on raw matrix values, so keep 2 as 2 instead of collapsing to 1
        matrix_one = grid.astype(np.float32)[None, None, :, :]
        freq_indices = np.arange(len(self.freq_hz), dtype=np.int64)[None, :]
        matrix_t = torch.from_numpy(matrix_one).to(DEVICE)
        freq_t = torch.from_numpy(freq_indices).to(DEVICE)
        pred = self.model(matrix_t, freq_t).squeeze(0).cpu().numpy()  # [4, Nf]
        s11 = pred[0] + 1j * pred[1]
        s21 = pred[2] + 1j * pred[3]
        J, P_IL, P_RL, P_SB, mag_s11, mag_s21 = compute_loss_from_arrays(self.freq_hz, s11, s21)
        return {"J": J, "P_IL": P_IL, "P_RL": P_RL, "P_SB": P_SB, "freq_hz": self.freq_hz.copy(), "s11": s11, "s21": s21, "mag_s11": mag_s11, "mag_s21": mag_s21}


class ADSEvaluator:
    def __init__(self):
        if not ADS_AVAILABLE:
            raise RuntimeError("ADS environment not available")

    def evaluate(self, grid: np.ndarray) -> Dict:
        rects, polys, circles, prts = get_shapes(grid)
        with multi_python.ads_context() as ads_ctx:
            _ = ads_ctx.call(ads_delete_rfpro_view)
        with multi_python.ads_context() as ads_ctx:
            _ = ads_ctx.call(ads_create_layout, args=[rects, polys, circles, prts])
        with multi_python.xxpro_context() as empro_ctx:
            ds_filename = empro_ctx.call(xxpro_run_simulation, args=[SIMU_FREQ_LOW_GHZ, SIMU_FREQ_HIGH_GHZ, SIMU_POINTS])
        with dataset.open(ds_filename) as output_data:
            sparams = output_data["data"].to_dataframe()
        freq_hz = sparams.index.to_numpy()
        s11 = sparams["S[1,1]"].to_numpy()
        s21 = sparams["S[2,1]"].to_numpy()
        J, P_IL, P_RL, P_SB, mag_s11, mag_s21 = compute_loss_from_arrays(freq_hz, s11, s21)
        return {"J": J, "P_IL": P_IL, "P_RL": P_RL, "P_SB": P_SB, "freq_hz": freq_hz, "s11": s11, "s21": s21, "mag_s11": mag_s11, "mag_s21": mag_s21, "sparams_df": sparams}


# =========================================================
# Visualization
# =========================================================
def save_step_figure(out_dir: Path, restart_id: int, step_idx: int, r: Optional[int], c: Optional[int], 
                     current_grid: np.ndarray, global_best_grid: np.ndarray, 
                     current_eval: Dict, global_best_eval: Dict, accepted_losses: List[float]):
    # 获取布局形状，注意这里中间和右边用的都是 global_best_grid
    rects_c, polys_c, circles_c, _ = get_shapes(current_grid)
    rects_b, polys_b, circles_b, _ = get_shapes(global_best_grid)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])
    
    # [左图] 当前 Grid (增加了 origin="lower" 对齐物理坐标系)
    ax1.imshow(current_grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="lower")
    ax1.set_title(f"Restart {restart_id} | Step {step_idx} (Current)")
    # 🌟 画绿色高亮框指出刚刚翻转的像素 (r, c 可能是 None，代表这是初始的第0步)
    if r is not None and c is not None:
        ax1.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='lime', linewidth=3))
        
    # [中图] 全局最优 Grid
    ax2.imshow(global_best_grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="lower")
    ax2.set_title("Global Best Grid")
    
    # [右图] 全局最优 Layout
    draw_ads_layout(ax3, rects_b, polys_b, circles_b)
    ax3.set_title("Global Best Layout")
    
    # [下左] 当前 S 参数
    freq_c = current_eval["freq_hz"] / 1e9
    ax4.plot(freq_c, 20 * np.log10(np.maximum(current_eval["mag_s11"], 1e-12)), label="S11")
    ax4.plot(freq_c, 20 * np.log10(np.maximum(current_eval["mag_s21"], 1e-12)), label="S21")
    ax4.grid(True); ax4.legend(); ax4.set_title(f"Current J={current_eval['J']:.5f}")
    
    # [下中] 全局最优 S 参数
    freq_b = global_best_eval["freq_hz"] / 1e9
    ax5.plot(freq_b, 20 * np.log10(np.maximum(global_best_eval["mag_s11"], 1e-12)), label="S11")
    ax5.plot(freq_b, 20 * np.log10(np.maximum(global_best_eval["mag_s21"], 1e-12)), label="S21")
    ax5.grid(True); ax5.legend(); ax5.set_title(f"Global Best J={global_best_eval['J']:.5f}")
    
    # [下右] 接受曲线与全局红线
    ax6.plot(np.arange(0, len(accepted_losses)), accepted_losses, label="Current Restart")
    ax6.axhline(global_best_eval["J"], color="red", linestyle="--", label="Global Best J")
    ax6.grid(True); ax6.legend(); ax6.set_title("Accepted Loss History")
    
    for ax in [ax4, ax5]:
        ax.set_xlabel("GHz"); ax.set_ylabel("dB")
    ax6.set_xlabel("Accepted Steps")
    
    plt.tight_layout()
    # 🌟 文件名按 restart_id 分开，再也不会覆盖了
    out_path = out_dir / "steps" / f"restart_{restart_id:02d}_step_{step_idx:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    dyno_path = out_dir / "dyno.png"
    plt.savefig(dyno_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# DBS
# =========================================================
@dataclass
class SearchRecord:
    restart_id: int
    outer_iter: int
    accepted_step: int
    r: int
    c: int
    old_value: int
    new_value: int
    J: float
    P_IL: float
    P_RL: float
    P_SB: float
    elapsed_sec: float


def evaluate_candidate(evaluator, grid: np.ndarray) -> Optional[Dict]:
    grid = repair_grid(grid)
    if STRICT_CONNECTIVITY:
        ok, _, _ = is_mid_connected_dfs(grid)
        if not ok:
            return None
    return evaluator.evaluate(grid)


def run_dbs():

    start_time = time.time()
    time_to_best = None

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SHUFFLE_SEED)
    evaluator = SurrogateEvaluator() if USE_SURROGATE else ADSEvaluator()

    all_records: List[SearchRecord] = []
    
    # 🌟 将全局最优的初始化提到大循环最外面
    global_best_grid = None
    global_best_eval = None

    for restart_id in range(N_RESTARTS):
        current_grid = load_init_grid() if restart_id == 0 else random_initial_grid(rng)
        current_eval = evaluate_candidate(evaluator, current_grid)
        if current_eval is None:
            raise RuntimeError("initial grid invalid after repair")
            
        # 初始化本趟的最优（仅用于判断 Early Stop）
        best_grid = current_grid.copy()
        best_eval = dict(current_eval)
        
        # 实时检查并更新全局最优
        if global_best_eval is None or current_eval["J"] < global_best_eval["J"]:
            global_best_grid = current_grid.copy()
            global_best_eval = dict(current_eval)
            
        accepted_losses = [current_eval["J"]]
        accepted_step = 0
        print(f"\n[Restart {restart_id}] init J={current_eval['J']:.6f} | Global Best={global_best_eval['J']:.6f}")
        
        # 🌟 画出第 0 步（降落时的初始状态）
        save_step_figure(out_dir, restart_id, accepted_step, None, None, 
                         current_grid, global_best_grid, current_eval, global_best_eval, accepted_losses)

        for outer_iter in range(1, MAX_OUTER_ITERS + 1):
            round_best_before = best_eval["J"]
            moves = all_single_cell_moves(current_grid, rng)
            
            for r, c, new_value in moves:
                old_value = int(current_grid[r, c])
                candidate = apply_single_cell_move(current_grid, r, c, new_value, rng)
                start = time.time()
                cand_eval = evaluate_candidate(evaluator, candidate)
                elapsed = time.time() - start
                
                if cand_eval is None:
                    continue
                    
                # 贪心下降逻辑
                if cand_eval["J"] < current_eval["J"]:
                    current_grid = candidate
                    current_eval = cand_eval
                    accepted_step += 1
                    
                    rec = SearchRecord(restart_id, outer_iter, accepted_step, r, c, old_value, new_value, 
                                       cand_eval["J"], cand_eval["P_IL"], cand_eval["P_RL"], cand_eval["P_SB"], elapsed)
                    all_records.append(rec)
                    accepted_losses.append(cand_eval["J"])
                    
                    # 更新本趟最优
                    if cand_eval["J"] < best_eval["J"]:
                        best_grid = candidate.copy()
                        best_eval = dict(cand_eval)
                        
                    # 🌟 更新全局最优
                    is_new_global_best = False
                    if cand_eval["J"] < global_best_eval["J"]:
                        global_best_grid = candidate.copy()
                        global_best_eval = dict(cand_eval)
                        is_new_global_best = True

                        time_to_best = time.time() - start_time
                        np.save(out_dir / "best_grid_binary012.npy", global_best_grid)
                        print(time_to_best)
                        
                    # 🌟 只要被接受，就画图记录
                    save_step_figure(out_dir, restart_id, accepted_step, r, c, 
                                     current_grid, global_best_grid, current_eval, global_best_eval, accepted_losses)
                
                    if is_new_global_best:
                        print(f"  🏆 [NEW GLOBAL BEST] restart={restart_id} step={accepted_step:04d} iter={outer_iter} cell=({r},{c}) {old_value}->{new_value} J={cand_eval['J']:.6f}")
                    else:
                        print(f"  ✅ [Accept] restart={restart_id} step={accepted_step:04d} iter={outer_iter} cell=({r},{c}) {old_value}->{new_value} J={cand_eval['J']:.6f}")
                        
                    # 如果达到了最终目标，直接停止一切搜索
                    if global_best_eval["J"] <= TARGET_LOSS:
                        print("  🎯 Target loss reached, stop everything!")
                        break # 跳出 inner loop
                        
            # 检查外层是否需要跳出
            if global_best_eval["J"] <= TARGET_LOSS:
                break # 跳出 outer iter
                
            improve = round_best_before - best_eval["J"]
            print(f"[Restart {restart_id}] outer_iter={outer_iter} local_best_J={best_eval['J']:.6f} improve={improve:.6g}")
            if improve < MIN_IMPROVEMENT_PER_ROUND:
                print(f"  ⚠️ Improvement too small, early stopping this restart.")
                break

        if global_best_eval["J"] <= TARGET_LOSS:
            break # 跳出 restart loop

    if global_best_grid is None or global_best_eval is None:
        raise RuntimeError("search failed, no valid design found")

    # ----- 下面的保存最终结果代码保持不变 -----
    np.save(out_dir / "best_grid_binary012.npy", global_best_grid)
    pd.DataFrame([asdict(r) for r in all_records]).to_csv(out_dir / "accepted_history.csv", index=False)
    freq_ghz = global_best_eval["freq_hz"] / 1e9
    pd.DataFrame({
        "freq_ghz": freq_ghz,
        "S11_real": np.real(global_best_eval["s11"]),
        "S11_imag": np.imag(global_best_eval["s11"]),
        "S21_real": np.real(global_best_eval["s21"]),
        "S21_imag": np.imag(global_best_eval["s21"]),
        "S11_mag": global_best_eval["mag_s11"],
        "S21_mag": global_best_eval["mag_s21"],
    }).to_csv(out_dir / "best_response.csv", index=False)

    rects, polys, circles, prts = get_shapes(global_best_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    draw_ads_layout(ax1, rects, polys, circles)
    ax1.set_title("Final Best Layout")

    ax2.plot(freq_ghz, 20 * np.log10(np.maximum(global_best_eval["mag_s11"], 1e-12)), label="S11")
    ax2.plot(freq_ghz, 20 * np.log10(np.maximum(global_best_eval["mag_s21"], 1e-12)), label="S21")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("dB")
    ax2.set_title(f"Best response (J={global_best_eval['J']:.5f})")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "final_best_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "mode": "surrogate" if USE_SURROGATE else "ads",
        "best_J": global_best_eval["J"],
        "best_P_IL": global_best_eval["P_IL"],
        "best_P_RL": global_best_eval["P_RL"],
        "best_P_SB": global_best_eval["P_SB"],
        "constants": {
            "N_RESTARTS": N_RESTARTS,
            "MAX_OUTER_ITERS": MAX_OUTER_ITERS,
            "TARGET_LOSS": TARGET_LOSS,
            "MIN_IMPROVEMENT_PER_ROUND": MIN_IMPROVEMENT_PER_ROUND,
            "STRICT_CONNECTIVITY": STRICT_CONNECTIVITY,
            "ALLOW_TOGGLE_ZERO_ONE": ALLOW_TOGGLE_ZERO_ONE,
            "ALLOW_TOGGLE_ONE_TWO": ALLOW_TOGGLE_ONE_TWO,
            "ALLOW_TOGGLE_TWO_ONE": ALLOW_TOGGLE_TWO_ONE,
            "ALLOW_TOGGLE_TWO_ZERO": ALLOW_TOGGLE_TWO_ZERO,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if CREATE_FINAL_LAYOUT and not USE_SURROGATE and ADS_AVAILABLE:
        with multi_python.ads_context() as ads_ctx:
            _ = ads_ctx.call(ads_delete_rfpro_view)
            _ = ads_ctx.call(ads_create_layout, args=[rects, polys, circles, prts])

    print("\n✅ Done!")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    total_time = time.time() - start_time

    print("\n========== Time Summary ==========")
    print(f"Total runtime        : {total_time:.2f} s")
    print(f"Time to global best  : {time_to_best:.2f} s")
    print("==================================")


if __name__ == "__main__":
    run_dbs()
