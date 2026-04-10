import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Polygon, Circle
from PIL import Image
import os

# =========================================================
# paths
# =========================================================
best_csv_path = "best_response.csv"
em_csv_path = "EM_simulated.csv"
s2p_path = "result.s2p"

# 新增：要显示的 layout npy
layout_npy_path = "best_grid_binary012.npy"   # 改成你的 npy 路径

# 实物照片
circuit_img_path = "IMG_0371.png"

output_pdf = "filter.pdf"
output_svg = "filter.svg"

fmin_ghz = 0.1
fmax_ghz = 6.0

size_text = "10.4 × 10.4"

# =========================================================
# layout config copied/adapted from your layout script
# =========================================================
N_COLS = 16
N_ROWS = 16
CELL = 0.65

# =========================================================
# helpers
# =========================================================
def mag_to_db(x, floor_db=-120):
    x = np.maximum(np.asarray(x, float), 10 ** (floor_db / 20))
    return 20 * np.log10(x)

def restrict3(f, y1, y2):
    m = (f >= fmin_ghz) & (f <= fmax_ghz)
    return f[m], y1[m], y2[m]

def parse_freq_to_ghz(s):
    m = re.match(r"([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)\s*([a-zA-Z]+)", str(s))
    v = float(m.group(1))
    u = m.group(2).lower()
    return v * {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1}[u]

def parse_complex(s):
    s = str(s).replace(" ", "")
    if "+j" in s:
        a, b = s.split("+j")
        return complex(float(a), float(b))
    if "-j" in s:
        a, b = s.split("-j")
        return complex(float(a), -float(b))
    return complex(float(s), 0)

def sparam_pair(v1, v2, fmt):
    if fmt == "RI":
        return complex(v1, v2)
    if fmt == "MA":
        return v1 * np.exp(1j * np.deg2rad(v2))
    if fmt == "DB":
        return 10 ** (v1 / 20) * np.exp(1j * np.deg2rad(v2))
    raise ValueError(f"Unsupported format: {fmt}")

def interp(x1, y1, x2, y2, t):
    return x1 + (t - y1) * (x2 - x1) / (y2 - y1)

# =========================================================
# loaders
# =========================================================
def load_best(p):
    df = pd.read_csv(p)
    f = df["freq_ghz"].to_numpy()
    s11 = df["S11_mag"].to_numpy() if "S11_mag" in df else np.hypot(df["S11_real"], df["S11_imag"])
    s21 = df["S21_mag"].to_numpy() if "S21_mag" in df else np.hypot(df["S21_real"], df["S21_imag"])
    return f, s11, s21

def load_em(p):
    fL, s11L, s21L = [], [], []
    for l in open(p, encoding="utf-8", errors="ignore"):
        l = l.strip()
        if not l or not re.match(r"^[+-]?\d", l):
            continue
        a = [x.strip() for x in l.split(",")]
        fL.append(parse_freq_to_ghz(a[0]))
        s11L.append(abs(parse_complex(a[1])))
        s21L.append(abs(parse_complex(a[2])))
    return np.array(fL), np.array(s11L), np.array(s21L)

def load_s2p(p):
    fmt = "MA"
    unit = "HZ"
    fL, s11L, s21L = [], [], []
    for l in open(p, encoding="utf-8", errors="ignore"):
        l = l.strip()
        if not l or l.startswith("!"):
            continue
        if l.startswith("#"):
            h = l[1:].split()
            unit = h[0]
            fmt = h[2]
            continue
        a = l.split()
        if len(a) < 9:
            continue

        f = float(a[0])
        if unit == "HZ":
            fghz = f / 1e9
        elif unit == "MHZ":
            fghz = f / 1e3
        elif unit == "GHZ":
            fghz = f
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        s11 = sparam_pair(float(a[1]), float(a[2]), fmt)
        s21 = sparam_pair(float(a[3]), float(a[4]), fmt)

        fL.append(fghz)
        s11L.append(abs(s11))
        s21L.append(abs(s21))

    return np.array(fL), np.array(s11L), np.array(s21L)

# =========================================================
# metrics
# =========================================================
def compute_metrics(f, s21):
    idx = np.argsort(f)
    f = f[idx]
    s21 = s21[idx]
    pk = np.argmax(s21)

    fL = None
    for i in range(pk, 0, -1):
        if s21[i - 1] < -1 <= s21[i]:
            fL = interp(f[i - 1], s21[i - 1], f[i], s21[i], -1)
            break

    fH = None
    for i in range(pk, len(f) - 1):
        if s21[i] >= -1 > s21[i + 1]:
            fH = interp(f[i], s21[i], f[i + 1], s21[i + 1], -1)
            break

    if fL is None or fH is None:
        raise RuntimeError("Cannot find -1 dB passband edges.")

    f0 = np.sqrt(fL * fH)
    fbw = (fH - fL) / f0 * 100
    IL = -np.max(s21)

    f20L = None
    for i in range(pk, 0, -1):
        if s21[i - 1] < -20 <= s21[i]:
            f20L = interp(f[i - 1], s21[i - 1], f[i], s21[i], -20)
            break

    f20R = None
    for i in range(pk, len(f) - 1):
        if s21[i] >= -20 > s21[i + 1]:
            f20R = interp(f[i], s21[i], f[i + 1], s21[i + 1], -20)
            break

    if f20L is None or f20R is None:
        raise RuntimeError("Cannot find -20 dB roll-off points.")

    rollL = 19 / (fL - f20L)
    rollR = 19 / (f20R - fH)
    return fL, fH, fbw, IL, rollL, rollR

# =========================================================
# layout drawing functions adapted from your npy/layout script
# =========================================================
def ensure_grid(grid):
    g = np.asarray(grid).astype(np.uint8)
    if g.shape != (N_ROWS, N_COLS):
        raise ValueError(f"grid shape must be {(N_ROWS, N_COLS)}, got {g.shape}")
    return g

def get_mid_two_rows():
    r1 = N_ROWS // 2 - 1
    r2 = N_ROWS // 2
    return r1, r2

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
        raise ValueError(f"invalid state: {s}")
    return pts

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

    return [
        [tl, top_mid, left_mid],
        [tr, top_mid, right_mid],
        [bl, left_mid, bottom_mid],
        [br, right_mid, bottom_mid],
    ]

def build_anti_aliasing_triangles_from_2x2_for_conductors(grid):
    grid = ensure_grid(grid)

    def is_cond(v):
        return int(v in (1, 2))

    state5 = np.zeros((N_ROWS, N_COLS), dtype=bool)
    state14 = [[set() for _ in range(N_COLS)] for _ in range(N_ROWS)]

    for r in range(1, N_ROWS - 1):
        for c in range(1, N_COLS - 1):
            if grid[r, c] != 0:
                continue
            up = is_cond(grid[r - 1, c])
            down = is_cond(grid[r + 1, c])
            left = is_cond(grid[r, c - 1])
            right = is_cond(grid[r, c + 1])
            if up == 1 and down == 1 and left == 1 and right == 1:
                add_state(state5, state14, r, c, 5)

    for r in range(N_ROWS - 1):
        for c in range(N_COLS - 1):
            a = is_cond(grid[r, c])
            b = is_cond(grid[r, c + 1])
            c_ = is_cond(grid[r + 1, c])
            d = is_cond(grid[r + 1, c + 1])
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

def infer_pair_from_grid(grid):
    """
    对于你保存下来的 grid，左右端口一般固定在中间两行。
    所以这里直接复用原脚本的中间两行定义。
    """
    r1, r2 = get_mid_two_rows()
    return (r1, r2)

def get_shapes(grid, pair):
    grid = ensure_grid(grid)
    rectangles = []
    polygons = []
    circles = []
    width = round(N_COLS * CELL, 4)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            if grid[r, c] not in (1, 2):
                continue
            x1 = round(c * CELL, 4)
            y1 = round(r * CELL, 4)
            x2 = round(x1 + CELL, 4)
            y2 = round(y1 + CELL, 4)
            rectangles.append(((x1, y1), (x2, y2)))

            if grid[r, c] == 2:
                xc = round(x1 + CELL / 2.0, 4)
                yc = round(y1 + CELL / 2.0, 4)
                radius = round(CELL / 3.0, 4)
                circles.append(((xc, yc), radius))

    polygons = build_anti_aliasing_triangles_from_2x2_for_conductors(grid)

    left_r1, left_r2 = pair
    right_r1, right_r2 = pair

    for rr in [left_r1, left_r2]:
        rectangles.append(((-CELL, rr * CELL), (0.0, rr * CELL + CELL)))
    for rr in [right_r1, right_r2]:
        rectangles.append(((width, rr * CELL), (width + CELL, rr * CELL + CELL)))

    y_center = ((left_r1 + 0.5) * CELL + (left_r2 + 0.5) * CELL) / 2.0
    centers = ((-CELL, y_center), (width + CELL, y_center))
    return rectangles, polygons, circles, centers

def draw_ads_layout(ax, rects, polys, circles, margin_cells=0.6):
    bg_color = "#f2f2f2"
    grid_color = "#666666"
    metal_color = "#f48a8a"
    tri_color = "#ff0000"
    via_color = "#2b6cb0"
    edge_color = "#444444"

    width = N_COLS * CELL
    height = N_ROWS * CELL
    pad = margin_cells * CELL

    ax.set_facecolor(bg_color)

    for i in range(N_COLS + 1):
        x = i * CELL
        ax.plot([x, x], [0, height], color=grid_color, linewidth=0.55, zorder=1)
    for j in range(N_ROWS + 1):
        y = j * CELL
        ax.plot([0, width], [y, y], color=grid_color, linewidth=0.55, zorder=1)

    for ((x1, y1), (x2, y2)) in rects:
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=metal_color, edgecolor=edge_color,
            linewidth=0.45, zorder=3
        ))

    for poly in polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor=tri_color, edgecolor="none", zorder=4))

    for (center, radius) in circles:
        ax.add_patch(Circle(center, radius=radius, facecolor=via_color, edgecolor="none", zorder=5))

    ax.set_xlim(-CELL - pad, width + CELL + pad)
    ax.set_ylim(-pad, height + pad)
    ax.set_aspect("equal")
    ax.axis("off")

# =========================================================
# global style
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# load data
# =========================================================
bf, b11, b21 = load_best(best_csv_path)
ef, e11, e21 = load_em(em_csv_path)
sf, s11, s21 = load_s2p(s2p_path)

b11, b21 = mag_to_db(b11), mag_to_db(b21)
e11, e21 = mag_to_db(e11), mag_to_db(e21)
s11, s21 = mag_to_db(s11), mag_to_db(s21)

bf, b11, b21 = restrict3(bf, b11, b21)
ef, e11, e21 = restrict3(ef, e11, e21)
sf, s11, s21 = restrict3(sf, s11, s21)

fL, fH, fbw, IL, rollL, rollR = compute_metrics(sf, s21)

# 读 layout npy，并画成中间那块
layout_grid = np.load(layout_npy_path)
layout_grid = ensure_grid(layout_grid)
pair = infer_pair_from_grid(layout_grid)
rects, polys, circles, _ = get_shapes(layout_grid, pair)

# =========================================================
# layout
# =========================================================
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(
    3, 2,
    width_ratios=[0.3, 0.7],   # 左窄右宽（右边放曲线）
    height_ratios=[1.1, 0.7, 0.7],
    wspace=0.08,
    hspace=0.03
)

axD = fig.add_subplot(gs[0, 0])   # layout
axC = fig.add_subplot(gs[1, 0])   # photo
axB = fig.add_subplot(gs[2, 0])   # table

axA = fig.add_subplot(gs[:, 1])   # 曲线占右边整列

# =========================================================
# A: S-parameters
# =========================================================
c_pred_s11 = "#1f77b4"
c_pred_s21 = "#ff7f0e"
c_simu_s11 = "#2ca02c"
c_simu_s21 = "#d62728"
c_meas_s11 = "#9467bd"
c_meas_s21 = "#8c564b"

axA.plot(bf, b11, ":", color=c_pred_s11, linewidth=3, label="S11_Pred.")
axA.plot(ef, e11, "--", color=c_simu_s11, linewidth=3, label="S11_Simu.")
axA.plot(sf, s11, "-", color=c_meas_s11, linewidth=3, label="S11_Meas.")

axA.plot(bf, b21, ":", color=c_pred_s21, linewidth=3, label="S21_Pred.")
axA.plot(ef, e21, "--", color=c_simu_s21, linewidth=3, label="S21_Simu.")
axA.plot(sf, s21, "-", color=c_meas_s21, linewidth=3, label="S21_Meas.")

axA.set_xlabel("freq, GHz")
# axA.set_ylabel(r"$|S_{ij}|$ (dB)")
axA.set_xlim(fmin_ghz, fmax_ghz)
axA.set_ylim(-50, 0)
axA.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
axA.minorticks_on()

axA.legend(
    ncol=2, loc="lower left",
    frameon=True, columnspacing=1.3, handlelength=2.2, borderpad=0.5
)

# =========================================================
# B: three-line table
# =========================================================
axB.axis("off")
axB.set_xlim(0, 1)
axB.set_ylim(0, 1)

rows = [
    ("Passband (GHz)", f"{fL:.2f}–{fH:.2f}"),
    ("FBW (%)", f"{fbw:.1f}"),
    ("IL (dB)", f"{IL:.2f}"),
    ("Roll-off (dB/GHz)", f"{rollL:.1f} / {rollR:.1f}"),
    ("Size (mm)", size_text),
]

x_left = 0.08
x_mid = 0.68
x_right = 0.95
y_top = 0.88
row_h = 0.2

axB.plot([x_left, x_right], [y_top, y_top], color="black", lw=1.3, clip_on=False)
axB.text((x_left + x_mid) / 2, y_top - row_h / 2, "Parameter",
         ha="center", va="center", fontsize=18, fontweight="bold")
axB.text((x_mid + x_right) / 2, y_top - row_h / 2, "Value",
         ha="center", va="center", fontsize=18, fontweight="bold")

y_midrule = y_top - row_h
axB.plot([x_left, x_right], [y_midrule, y_midrule], color="black", lw=0.8, clip_on=False)

for i, (k, v) in enumerate(rows):
    y = y_top - row_h * (i + 1.5)
    axB.text(x_left + 0.005, y, k, ha="left", va="center", fontsize=18)
    axB.text(x_right - 0.005, y, v, ha="right", va="center", fontsize=18)

y_bottom = y_top - row_h * (len(rows) + 1)
axB.plot([x_left, x_right], [y_bottom, y_bottom], color="black", lw=1.3, clip_on=False)

# =========================================================
# D: layout from npy
# =========================================================
draw_ads_layout(axD, rects, polys, circles, margin_cells=0.35)

# =========================================================
# C: circuit photo
# =========================================================
axC.set_xticks([])
axC.set_yticks([])
for spine in axC.spines.values():
    spine.set_linewidth(0.8)

if os.path.exists(circuit_img_path):
    img = Image.open(circuit_img_path)
    axC.imshow(img)
    axC.set_aspect("equal")
else:
    axC.text(0.5, 0.5, "Circuit Photo", ha="center", va="center", fontsize=18)

plt.tight_layout(pad=0.35)
fig.savefig(output_pdf, bbox_inches="tight")
fig.savefig(output_svg, bbox_inches="tight")
plt.show()