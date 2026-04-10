import os
import copy
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# config
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_FILE_LOW = "./DualBand/dual-FL/20-24Trainset.csv"
TRAIN_FILE_HIGH = "./DualBand/dual-FL/50-54Trainset.csv"
TEST_FILE = "./DualBand/dual-FL/testset.csv"

INPUT_COLS = ["freq", "WS1", "WS2", "W1", "W2", "LS1", "LS2", "L1", "L2"]
TARGET_COLS = ["S11r", "S11i", "S21r", "S21i", "S31r", "S31i", "S41r", "S41i"]

DUAL_NODE_SPECS = {
    0: [2.0, 5.0],
    1: [2.1, 5.1],
    2: [2.2, 5.2],
    3: [2.3, 5.3],
    4: [2.4, 5.4],
}
BASELINE_NODE_ID = 2

BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
TOTAL_ROUNDS = 40          # 先用 300 调通；后面可改回 600
SINGLE_NODE_EPOCHS = 200
WEIGHT_DECAY = 1e-4
RETRO_LAMBDA = 0.5
KBNN_L1_ALPHA = 1e-7

PRINT_EVERY = 20
RESULT_CSV = "./DualBand/dual-FL/dualband_compare_results.csv"


# ============================================================
# utils
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def sec_to_str(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h} h {m} m {s} s"
    return f"{m} m {s} s"


class SimpleStandardizer:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-8
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-8

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_dualband_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_low = pd.read_csv(TRAIN_FILE_LOW)
    df_high = pd.read_csv(TRAIN_FILE_HIGH)
    train_df = pd.concat([df_low, df_high], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def df_to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = df[INPUT_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)
    return x, y


def make_dualband_client_specs(train_df: pd.DataFrame):
    client_specs = []
    for cid, freqs in DUAL_NODE_SPECS.items():
        df_c = train_df[train_df["freq"].isin(freqs)].copy().reset_index(drop=True)
        client_specs.append({
            "client_id": cid,
            "freqs": freqs,
            "df": df_c,
        })
    return client_specs


# ============================================================
# models
# ============================================================
class DualBandMLP(nn.Module):
    def __init__(self, input_dim=9, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MappingBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.nonlinear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class CoarseBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualBandKBNN(nn.Module):
    def __init__(self, input_dim=9, output_dim=8, map_hidden=64):
        super().__init__()
        self.input_map = MappingBlock(input_dim, input_dim, hidden_dim=map_hidden)
        self.coarse = CoarseBlock(input_dim, output_dim)
        self.output_map = MappingBlock(output_dim, output_dim, hidden_dim=map_hidden)

    def forward(self, x):
        x_m = self.input_map(x)
        y_c = self.coarse(x_m)
        y = self.output_map(y_c)
        return y


def kbnn_l1_penalty(model: nn.Module, alpha: float = KBNN_L1_ALPHA):
    reg = 0.0
    for name, p in model.named_parameters():
        if "input_map.linear.weight" in name or "output_map.linear.weight" in name:
            reg = reg + p.abs().sum()
        if "input_map.nonlinear.2.weight" in name or "output_map.nonlinear.2.weight" in name:
            reg = reg + p.abs().sum()
    return alpha * reg


def build_dualband_model(arch: str = "mlp", input_dim: int = 9, output_dim: int = 8):
    if arch == "mlp":
        return DualBandMLP(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    elif arch == "kbnn":
        return DualBandKBNN(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ============================================================
# metrics
# ============================================================
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def compute_r2_full(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    ss_res = 0.0
    y_sum = 0.0
    y_sq_sum = 0.0
    n_elem = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        pred = model(xb)
        diff = pred - yb
        ss_res += float((diff * diff).sum().item())
        y_sum += float(yb.sum().item())
        y_sq_sum += float((yb * yb).sum().item())
        n_elem += int(yb.numel())

    if n_elem == 0:
        return 0.0

    y_mean = y_sum / n_elem
    ss_tot = y_sq_sum - n_elem * (y_mean ** 2)
    if abs(ss_tot) < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def eval_dualband(model: nn.Module, df_eval: pd.DataFrame, scaler: SimpleStandardizer):
    model.eval()

    x_raw, y_raw = df_to_xy(df_eval)
    x_std = scaler.transform_x(x_raw)
    y_std = scaler.transform_y(y_raw)

    x_t = torch.tensor(x_std, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_std, dtype=torch.float32, device=DEVICE)

    pred = model(x_t)

    mse = torch.mean((pred - y_t) ** 2).item()
    mae = torch.mean(torch.abs(pred - y_t)).item()

    ss_res = torch.sum((pred - y_t) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t, dim=0, keepdim=True)) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()

    out = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }

    for f in sorted(df_eval["freq"].unique()):
        mask = df_eval["freq"].to_numpy() == f
        pf = pred[mask]
        yf = y_t[mask]
        ss_res_f = torch.sum((pf - yf) ** 2)
        ss_tot_f = torch.sum((yf - torch.mean(yf, dim=0, keepdim=True)) ** 2)
        out[f"r2@{f:.1f}"] = (1.0 - ss_res_f / (ss_tot_f + 1e-12)).item()

    return out


# ============================================================
# BigTable-style FL
# ============================================================
class ParaServer:
    def __init__(self, model: nn.Module, learning_rate: float):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TOTAL_ROUNDS,
            eta_min=1e-6,
        )
        self.client_scores: Dict[str, float] = {}

    def upload(self, grads: List[torch.Tensor], client_id: str, score: float):
        score_value = float(score)
        if not np.isfinite(score_value):
            score_value = 0.0
        self.client_scores[client_id] = max(0.0, score_value)

        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g.detach()
        self.optimizer.step()
        return self.download()

    def download(self):
        return self.model.state_dict()


class RetroClient:
    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        all_freqs_by_client: Dict[str, List[float]],
        my_freqs: List[float],
        scaler: SimpleStandardizer,
        arch: str,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.all_freqs_by_client = all_freqs_by_client
        self.my_freqs = list(my_freqs)
        self.scaler = scaler
        self.arch = arch

        self.anchor_model = None
        self.other_scores: Dict[str, float] = {}

    def train_one_round(self, ps: ParaServer, model_builder, global_round: int, first_epoch: bool = False):
        if first_epoch:
            self.anchor_model = model_builder().to(DEVICE)
            self.anchor_model.load_state_dict(ps.download())
            self.anchor_model.eval()
            self.other_scores = dict(ps.client_scores)

        model = model_builder().to(DEVICE)
        model.load_state_dict(ps.download())
        model.train()

        for xb, yb in self.train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            model.zero_grad(set_to_none=True)
            pred = model(xb)
            tr_mse = F.mse_loss(pred, yb)
            if self.arch == "kbnn":
                tr_mse = tr_mse + kbnn_l1_penalty(model)
            tr_mse.backward()
            grads = [p.grad.detach().clone() for p in model.parameters()]

            ss_res = torch.sum((pred - yb) ** 2)
            y_mean = torch.mean(yb)
            ss_tot = torch.sum((yb - y_mean) ** 2)
            if torch.abs(ss_tot) < 1e-12:
                tr_r2 = torch.tensor(0.0, device=DEVICE)
            else:
                tr_r2 = 1.0 - ss_res / ss_tot

            sum_r2 = 1.0

            if self.anchor_model is not None:
                for other_client_id, weight in self.other_scores.items():
                    if other_client_id == self.client_id or abs(weight) < 1e-12:
                        continue

                    other_freqs = self.all_freqs_by_client[str(other_client_id)]
                    for f in other_freqs:
                        xb_alt = xb.clone()

                        # xb 当前已标准化，所以要把 raw freq 转成标准化 freq
                        f_std = (float(f) - float(self.scaler.x_mean[0, 0])) / float(self.scaler.x_std[0, 0])
                        xb_alt[:, 0] = f_std

                        with torch.no_grad():
                            y_anchor = self.anchor_model(xb_alt)

                        model.zero_grad(set_to_none=True)
                        y_pred_other = model(xb_alt)
                        loss_other = F.mse_loss(y_pred_other, y_anchor)
                        if self.arch == "kbnn":
                            loss_other = loss_other + kbnn_l1_penalty(model)
                        loss_other.backward()

                        grad_other = [p.grad.detach().clone() for p in model.parameters()]
                        grads = [g + go * float(weight) for g, go in zip(grads, grad_other)]
                        sum_r2 += float(weight)

            avg_grads = [g / sum_r2 for g in grads]
            model.load_state_dict(ps.upload(avg_grads, self.client_id, float(tr_r2.detach().item())))

        train_r2 = compute_r2_full(model, self.train_loader)
        return train_r2


# ============================================================
# scheduler
# ============================================================
@dataclass
class ClientScheduleState:
    is_online: bool
    remaining_rounds: int


class OnlineOfflineScheduler:
    def __init__(
        self,
        num_clients: int,
        online_span_range=(8, 16),
        offline_span_range=(4, 10),
        init_online_prob=0.6,
        min_online_clients=1,
        seed=42,
    ):
        self.num_clients = num_clients
        self.online_span_range = online_span_range
        self.offline_span_range = offline_span_range
        self.init_online_prob = init_online_prob
        self.min_online_clients = min_online_clients
        self.rng = random.Random(seed)

        self.states: List[ClientScheduleState] = []
        for _ in range(num_clients):
            is_online = self.rng.random() < init_online_prob
            remaining = self._sample_online_span() if is_online else self._sample_offline_span()
            self.states.append(ClientScheduleState(is_online=is_online, remaining_rounds=remaining))

        self._enforce_min_online()

    def _sample_online_span(self):
        return self.rng.randint(self.online_span_range[0], self.online_span_range[1])

    def _sample_offline_span(self):
        return self.rng.randint(self.offline_span_range[0], self.offline_span_range[1])

    def _enforce_min_online(self):
        online_ids = [i for i, st in enumerate(self.states) if st.is_online]
        while len(online_ids) < self.min_online_clients:
            idx = self.rng.randrange(self.num_clients)
            if not self.states[idx].is_online:
                self.states[idx].is_online = True
                self.states[idx].remaining_rounds = self._sample_online_span()
                online_ids.append(idx)

    def get_online_clients(self):
        return [i for i, st in enumerate(self.states) if st.is_online]

    def step(self):
        for st in self.states:
            st.remaining_rounds -= 1
            if st.remaining_rounds <= 0:
                st.is_online = not st.is_online
                st.remaining_rounds = self._sample_online_span() if st.is_online else self._sample_offline_span()
        self._enforce_min_online()


# ============================================================
# experiment helpers
# ============================================================
def build_train_loader_from_df(df_local: pd.DataFrame, scaler: SimpleStandardizer) -> DataLoader:
    x_raw, y_raw = df_to_xy(df_local)
    x_std = scaler.transform_x(x_raw)
    y_std = scaler.transform_y(y_raw)
    ds = ArrayDataset(x_std, y_std)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


def make_test_loader(test_df: pd.DataFrame, scaler: SimpleStandardizer) -> DataLoader:
    x_raw, y_raw = df_to_xy(test_df)
    x_std = scaler.transform_x(x_raw)
    y_std = scaler.transform_y(y_raw)
    ds = ArrayDataset(x_std, y_std)
    return DataLoader(ds, batch_size=4096, shuffle=False, drop_last=False)


# ============================================================
# single-node baseline
# ============================================================
def run_dualband_single_node_experiment(
    arch="mlp",
    node_id=BASELINE_NODE_ID,
    num_epochs=SINGLE_NODE_EPOCHS,
):
    print(f"\n===== Single-node | arch={arch} | node={node_id} =====")

    train_df, test_df = load_dualband_frames()
    client_specs = make_dualband_client_specs(train_df)

    df_local = None
    for spec in client_specs:
        if spec["client_id"] == node_id:
            df_local = spec["df"]
            break
    if df_local is None:
        raise RuntimeError(f"Node {node_id} not found.")

    x_train, y_train = df_to_xy(df_local)
    scaler = SimpleStandardizer()
    scaler.fit(x_train, y_train)

    train_loader = build_train_loader_from_df(df_local, scaler)
    test_loader = make_test_loader(test_df, scaler)

    model = build_dualband_model(arch=arch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    t0 = time.time()

    epoch_bar = tqdm(range(num_epochs), desc=f"Single-{arch}", leave=True)
    for epoch in epoch_bar:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            if arch == "kbnn":
                loss = loss + kbnn_l1_penalty(model)
            loss.backward()
            optimizer.step()

        if epoch % PRINT_EVERY == 0 or epoch == num_epochs - 1:
            train_r2 = compute_r2_full(model, train_loader)
            test_r2 = compute_r2_full(model, test_loader)
            epoch_bar.set_postfix({
                "train_r2": f"{train_r2:.4f}",
                "test_r2": f"{test_r2:.4f}",
            })

    train_time = time.time() - t0
    metrics = eval_dualband(model, test_df, scaler)

    out = {
        "arch": arch,
        "mode": "w/o RetroFL",
        "node_id": node_id,
        "metrics": metrics,
        "train_time_sec": train_time,
        "n_params": count_trainable_params(model),
        "model_size_mb": model_size_mb(model),
        "model": model,
        "scaler": scaler,
    }
    return out


# ============================================================
# retroFL multi-client experiment
# ============================================================
def run_dualband_retrofl_experiment(
    arch="mlp",
    total_rounds=TOTAL_ROUNDS,
):
    print(f"\n===== RetroFL | arch={arch} =====")

    train_df, test_df = load_dualband_frames()
    x_all, y_all = df_to_xy(train_df)

    scaler = SimpleStandardizer()
    scaler.fit(x_all, y_all)

    client_specs = make_dualband_client_specs(train_df)
    test_loader = make_test_loader(test_df, scaler)

    def model_builder():
        return build_dualband_model(arch=arch)

    base_model = model_builder()
    ps = ParaServer(base_model, learning_rate=LEARNING_RATE)

    all_freqs_by_client = {
        str(spec["client_id"]): list(spec["freqs"])
        for spec in client_specs
    }

    clients = []
    client_prev_online = {}

    for spec in client_specs:
        loader = build_train_loader_from_df(spec["df"], scaler)
        client_id = str(spec["client_id"])
        clients.append(
            RetroClient(
                client_id=client_id,
                train_loader=loader,
                all_freqs_by_client=all_freqs_by_client,
                my_freqs=spec["freqs"],
                scaler=scaler,
                arch=arch,
            )
        )
        client_prev_online[client_id] = False

    global_r2_history = []
    client_r2_history = {client.client_id: [] for client in clients}
    client_window_start_marks = {client.client_id: [] for client in clients}

    client_scheduler = OnlineOfflineScheduler(
        num_clients=len(clients),
        online_span_range=(8, 16),
        offline_span_range=(4, 10),
        init_online_prob=0.6,
        min_online_clients=1,
        seed=42,
    )

    t0 = time.time()

    round_bar = tqdm(range(total_rounds), desc=f"RetroFL-{arch}", leave=True)
    for global_round in round_bar:
        online_ids = client_scheduler.get_online_clients()

        for client in clients:
            client_r2_history[client.client_id].append(np.nan)

        order = online_ids[:]
        random.shuffle(order)

        cur_train_r2 = []
        cur_client_id = []

        for idx in order:
            client = clients[idx]
            window_start = not client_prev_online[client.client_id]

            train_r2 = client.train_one_round(
                ps,
                model_builder=model_builder,
                global_round=global_round,
                first_epoch=window_start,
            )
            client_r2_history[client.client_id][-1] = float(train_r2)

            if window_start:
                client_window_start_marks[client.client_id].append(global_round)

            cur_train_r2.append(round(train_r2, 3))
            cur_client_id.append(client.client_id)

        for idx, client in enumerate(clients):
            client_prev_online[client.client_id] = (idx in online_ids)

        client_scheduler.step()
        ps.scheduler.step()

        global_model = model_builder().to(DEVICE)
        global_model.load_state_dict(ps.download())
        g_r2 = compute_r2_full(global_model, test_loader)
        global_r2_history.append(g_r2)

        if global_round % PRINT_EVERY == 0 or global_round == total_rounds - 1:
            round_bar.set_postfix({
                "online": len(order),
                "g_r2": f"{g_r2:.4f}",
                "train_r2": cur_train_r2,
                "clients": cur_client_id,
            })

    train_time = time.time() - t0
    final_model = model_builder().to(DEVICE)
    final_model.load_state_dict(ps.download())

    metrics = eval_dualband(final_model, test_df, scaler)
    tail_len = max(1, int(math.ceil(0.2 * len(global_r2_history))))
    tail_std = float(np.std(np.asarray(global_r2_history[-tail_len:], dtype=np.float64)))

    out = {
        "arch": arch,
        "mode": "RetroFL",
        "metrics": metrics,
        "train_time_sec": train_time,
        "n_params": count_trainable_params(final_model),
        "model_size_mb": model_size_mb(final_model),
        "tail_std": tail_std,
        "global_r2_history": global_r2_history,
        "client_r2_history": client_r2_history,
        "model": final_model,
        "scaler": scaler,
    }
    return out


# ============================================================
# summary table
# ============================================================
def build_summary_rows(result_mlp_single, result_mlp_retro, result_kbnn_single, result_kbnn_retro):
    rows = []

    def add_row(name, result):
        rows.append({
            "Method": name,
            "Data Gen": "TBD",
            "Training": sec_to_str(result["train_time_sec"]),
            "Optimization": "TBD",
            "Total": "TBD",
            "Loss(MSE)": result["metrics"]["mse"],
            "R2": result["metrics"]["r2"],
            "Params": result["n_params"],
            "Model Size (MB)": result["model_size_mb"],
        })

    add_row("MLP (w/o RetroFL)", result_mlp_single)
    add_row("KBNN (w/o RetroFL)", result_kbnn_single)
    add_row("MLP (RetroFL)", result_mlp_retro)
    add_row("KBNN (RetroFL)", result_kbnn_retro)

    return pd.DataFrame(rows)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    print("DEVICE =", DEVICE)

    result_mlp_single = run_dualband_single_node_experiment(
        arch="mlp",
        node_id=BASELINE_NODE_ID,
        num_epochs=SINGLE_NODE_EPOCHS,
    )

    result_mlp_retro = run_dualband_retrofl_experiment(
        arch="mlp",
        total_rounds=TOTAL_ROUNDS,
    )

    result_kbnn_single = run_dualband_single_node_experiment(
        arch="kbnn",
        node_id=BASELINE_NODE_ID,
        num_epochs=SINGLE_NODE_EPOCHS,
    )

    result_kbnn_retro = run_dualband_retrofl_experiment(
        arch="kbnn",
        total_rounds=TOTAL_ROUNDS,
    )

    print("\n================ FINAL METRICS ================")
    print("MLP (w/o RetroFL):", result_mlp_single["metrics"])
    print("MLP (RetroFL):    ", result_mlp_retro["metrics"])
    print("KBNN (w/o RetroFL):", result_kbnn_single["metrics"])
    print("KBNN (RetroFL):    ", result_kbnn_retro["metrics"])

    summary_df = build_summary_rows(
        result_mlp_single,
        result_mlp_retro,
        result_kbnn_single,
        result_kbnn_retro,
    )
    print("\n================ SUMMARY TABLE ================")
    print(summary_df)

    summary_df.to_csv(RESULT_CSV, index=False)
    print(f"\nSaved summary to: {RESULT_CSV}")