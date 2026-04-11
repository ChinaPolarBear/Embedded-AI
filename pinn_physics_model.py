import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from plot_output_utils import make_figure_output_dir, save_figure

# ============================================================
# 1) 全局配置 & 物理参数 (NLSE / 光纤参数 + 训练超参)
# ============================================================

# 自动选择 GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 光纤参数（NLSE） ---
alpha = 0.081      # loss coefficient（损耗项系数）
beta2 = -1.0       # group velocity dispersion (GVD) coefficient（二阶色散）
gamma = 1.0        # nonlinearity coefficient（克尔非线性系数）

# --- 链路/采样设置 ---
L = 1.0            # fiber length / propagation distance（传播距离）
T_max = 1.0        # time window is [-T_max, +T_max]
N_t = 256          # number of time samples（时间采样点数）

# --- QAM / RRC 波形生成设置（输入 A(0,t) 的分布）---
num_symbols = 64   # 64-QAM symbol count
sps = 8            # samples per symbol（每个符号对应的采样点）
rrc_beta = 0.2     # RRC roll-off
rrc_span = 10      # RRC span in symbols（滤波器跨越多少个符号）

# --- 数据集大小 ---
N_train = 1024
N_test  = 128

# --- 两阶段训练轮数 ---
epochs_pretrain = 1500   # Phase 1: physics-first (PDE + IC)
epochs_hybrid   = 3500   # Phase 2: physics + anchors at z=L

# --- 一次迭代“抽多少条函数”作为 DeepONet 的 batch（每条函数=一条 A0(t)）---
batch_size_fns = 64

# --- PINN 采样点数量（越大越准但越慢）---
N_ic   = 1024     # IC: 初值约束采样点数（z=0）
N_data = 512      # data anchor: 端点监督采样点数（z=L）
N_pde  = 4096     # PDE: 物理残差采样点数（域内）

# 学习率（配合 StepLR）
lr = 8e-4

# --- 训练/评估噪声（观测=always noisy）---
snr_db_train = 25.0
snr_db_eval  = 25.0

# --- DeepONet 结构超参 ---
P_dim         = 128  # latent basis size（branch/trunk 输出维度的一半，复数所以会*2）
branch_hidden = 128
trunk_hidden  = 256
fourier_dim   = 64   # Fourier feature dimension for trunk input (t,z)

# 模型保存路径
MODEL_PATH = "hybrid_pinn_deeponet.pth"


# ============================================================
# 2) 时间轴 & SSFM 求解器（用来生成“真值”数据）
# ============================================================

# 生成时间网格 t ∈ [-T_max, T_max]
t_grid = torch.linspace(-T_max, T_max, N_t, device=device)

# 采样间隔 dt（用于频域采样）
dt = (t_grid[1] - t_grid[0]).item()

# 频域网格 f 和角频率 ω（SSFM 线性步用）
freq  = torch.fft.fftfreq(N_t, d=dt).to(device)
omega = 2 * np.pi * freq


def ssfm_propagate(A0_complex, L_dist, n_steps=200):
    """
    Split-Step Fourier Method (SSFM) for NLSE.
    输入:
      - A0_complex: shape [N_t] complex tensor, A(0,t)
      - L_dist: propagation distance
      - n_steps: number of z-steps
    输出:
      - A(L,t): shape [N_t] complex tensor
    """
    # z方向步长
    dz = L_dist / n_steps

    # copy输入，避免原地修改
    A = A0_complex.clone()

    # 线性半步传输算子：exp( (j*beta2/2 * ω^2 - alpha/2) * dz/2 )
    # - beta2: dispersion
    # - alpha: loss
    linear_half = torch.exp((0.5j * beta2 * omega**2 - 0.5 * alpha) * (dz / 2))

    # 非线性相位旋转系数：exp( j*gamma*|A|^2*dz )
    nonlinear_coef = 1j * gamma * dz

    # SSFM: 线性半步 -> 非线性整步 -> 线性半步
    for _ in range(n_steps):
        # 线性：频域乘算子，再回时域
        A = torch.fft.ifft(torch.fft.fft(A) * linear_half)
        # 非线性：时域幅度相关相位旋转
        A = A * torch.exp(nonlinear_coef * torch.abs(A)**2)
        # 线性：再做一次半步
        A = torch.fft.ifft(torch.fft.fft(A) * linear_half)

    return A


# ============================================================
# 3) 生成 QAM + RRC 初始波形 A(0,t) & 构建数据集
# ============================================================

def _rrc_taps_torch(beta: float, sps: int, span: int, device):
    """
    生成 Root-Raised-Cosine (RRC) FIR taps（能量归一化）。
    输出:
      taps: shape [span*sps + 1], float tensor on device
    """
    if beta < 0 or beta > 1:
        raise ValueError("beta must be in [0, 1].")
    if sps <= 0 or span <= 0:
        raise ValueError("sps and span must be positive integers.")

    N = span * sps
    # t单位是“符号周期”（除以 sps）
    t = torch.arange(-N/2, N/2 + 1, device=device, dtype=torch.float32) / float(sps)
    taps = torch.zeros_like(t)

    eps = 1e-12  # 防止除0
    for i in range(t.numel()):
        ti = t[i].item()

        # 特殊点：t=0
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + (4 * beta / np.pi)

        # 特殊点：t = ±1/(4β)
        elif beta > 0 and abs(abs(ti) - 1/(4*beta)) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )

        # 通用公式
        else:
            ti_t = t[i]
            numerator = torch.sin(np.pi * ti_t * (1 - beta)) + 4 * beta * ti_t * torch.cos(np.pi * ti_t * (1 + beta))
            denominator = (np.pi * ti_t * (1 - (4 * beta * ti_t) ** 2))
            taps[i] = numerator / (denominator + eps)

    # 能量归一化：sum(taps^2)=1
    taps = taps / torch.sqrt(torch.sum(taps**2) + eps)
    return taps


def _qam64_symbols_torch(num_symbols: int, device):
    """
    随机生成 64-QAM 复符号，并把平均功率归一到 1。
    输出:
      syms: shape [num_symbols], complex64 tensor
    """
    # 64QAM: I/Q 取 8 个电平
    levels = torch.tensor([-7., -5., -3., -1., 1., 3., 5., 7.], device=device)
    I_idx = torch.randint(0, 8, (num_symbols,), device=device)
    Q_idx = torch.randint(0, 8, (num_symbols,), device=device)

    I = levels[I_idx]
    Q = levels[Q_idx]
    syms = (I + 1j * Q).to(torch.complex64)

    # 平均功率归一化到 1：E{|syms|^2}=1
    syms = syms / torch.sqrt(torch.mean(torch.abs(syms) ** 2) + 1e-12)
    return syms


def generate_qam_waveform(
    sps: int = 8,
    beta: float = 0.2,
    span: int = 10,
    normalize_power: bool = True,
    center_crop: bool = True,
):
    """
    生成 realistic 64QAM + RRC 成形波形 A0(t) = A(0,t)
    返回:
      - A0:           [N_t] complex tensor on device
      - symbols_np:   numpy complex array（用于画星座）
      - centers_np:   numpy array（符号中心对应时间坐标，仅用于可视化/调试）
    """
    # (1) 生成 64QAM 符号序列
    syms = _qam64_symbols_torch(num_symbols, device=device)  # [num_symbols]
    symbols_np = syms.detach().cpu().numpy()

    # (2) 上采样：把符号插入到离散序列中，每个符号间隔 sps 点
    up = torch.zeros(num_symbols * sps, dtype=torch.complex64, device=device)
    up[::sps] = syms  # 其它点为0（相当于脉冲串）

    # (3) RRC 成形：对 up 做 FIR 卷积（实部虚部分开 conv1d）
    h = _rrc_taps_torch(beta=beta, sps=sps, span=span, device=device)  # real taps
    x = up.view(1, 1, -1)   # conv1d 需要 [N] -> [B=1,C=1,N]
    h1 = h.view(1, 1, -1)

    # same padding，让输出长度≈输入长度
    pad = (h.numel() - 1) // 2

    y_re = torch.nn.functional.conv1d(x.real, h1, padding=pad)
    y_im = torch.nn.functional.conv1d(x.imag, h1, padding=pad)

    shaped = (y_re + 1j * y_im).view(-1)  # [num_symbols*sps]

    # (4) 把 shaped 裁剪/补零到固定长度 N_t（保证模型输入维度固定）
    if shaped.numel() < N_t:
        pad_len = N_t - shaped.numel()
        left = pad_len // 2
        right = pad_len - left
        shaped = torch.nn.functional.pad(shaped, (left, right))
    elif shaped.numel() > N_t:
        start = (shaped.numel() - N_t) // 2 if center_crop else 0
        shaped = shaped[start:start + N_t]

    A0 = shaped[:N_t].clone()

    # (5) 再做一次平均功率归一化（让训练尺度稳定）
    if normalize_power:
        power = torch.sqrt(torch.mean(torch.abs(A0) ** 2) + 1e-12)
        A0 = A0 / (power + 1e-12)

    # (6) 返回符号中心（只是辅助可视化，不参与训练）
    centers = torch.linspace(t_grid.min(), t_grid.max(), num_symbols, device=device)
    centers_np = centers.detach().cpu().numpy()

    return A0, symbols_np, centers_np


def add_awgn(signal: torch.Tensor, snr_db: float):
    """
    给复信号添加 proper complex AWGN.
    输入:
      - signal: complex tensor [...], 例如 [N_t]
      - snr_db: SNR(dB)
    输出:
      - signal + noise
    """
    # 信号平均功率 E{|x|^2}
    power = torch.mean(torch.abs(signal) ** 2)

    # SNR(dB) -> 线性
    snr_linear = 10 ** (snr_db / 10)

    # 噪声功率 = 信号功率 / SNR
    noise_power = power / snr_linear

    # proper complex noise：Re/Im 方差各占一半
    noise = torch.sqrt(noise_power / 2) * (
        torch.randn_like(signal.real) +
        1j * torch.randn_like(signal.real)
    )

    return signal + noise


def build_dataset(n_samples, snr_db):
    """
    构建训练/测试数据集：
      - A0: 输入 A(0,t)
      - AL_clean: SSFM 得到的“无噪声真值” A(L,t)
      - AL_noisy: 加 AWGN 后的“观测” A_obs(L,t)

    你现在的训练策略是：
      - PDE/IC/clean anchor 主要贴合 AL_clean（学习物理+真值）
      - obs loss 让模型知道观测噪声存在，但不要硬去拟合噪声纹理
    """
    A0_all = torch.zeros(n_samples, N_t, dtype=torch.complex64, device=device)
    AL_clean_all = torch.zeros_like(A0_all)
    AL_noisy_all = torch.zeros_like(A0_all)

    for n in range(n_samples):
        # 生成输入波形 A(0,t)
        A0, _, _ = generate_qam_waveform()

        # SSFM 得到真值 A(L,t)
        AL_clean = ssfm_propagate(A0, L)

        # 永远加噪声：得到观测 A_obs(L,t)
        AL_noisy = add_awgn(AL_clean, snr_db=float(snr_db))

        # 写入 batch
        A0_all[n] = A0
        AL_clean_all[n] = AL_clean
        AL_noisy_all[n] = AL_noisy

    return A0_all, AL_clean_all, AL_noisy_all


def make_branch_input(A0_batch):
    """
    DeepONet branch 输入必须是实向量：
    把 complex A0(t) 拆成 [Re, Im] 拼接
    输入:
      A0_batch: [B, N_t] complex
    输出:
      [B, 2*N_t] float
    """
    return torch.cat([A0_batch.real, A0_batch.imag], dim=1)


# ============================================================
# 4) DeepONet 结构 + Fourier trunk
# ============================================================

class ResidualBlock(nn.Module):
    """最简单的两层 MLP residual block：x -> x + f(x)"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return x + h


class DeepONet(nn.Module):
    """
    DeepONet 思路：
      - Branch 网络输入：整条 A0(t)（一个“函数”），输出一组系数 B (latent)
      - Trunk 网络输入：(t,z) 坐标，输出一组基函数值 T (latent)
      - 输出 u(t,z), v(t,z) = 复内积 <B, T> 的实部/虚部
    """
    def __init__(self, n_sensors, p_dim, fourier_dim):
        super().__init__()

        # -------- Branch: 输入维度 2*N_t（Re/Im 拼起来）--------
        self.branch_in = nn.Linear(2 * n_sensors, branch_hidden)
        self.branch_blocks = nn.Sequential(
            ResidualBlock(branch_hidden),
            ResidualBlock(branch_hidden),
        )
        # 输出 2*p_dim：前 p_dim 表示复系数实部，后 p_dim 表示虚部
        self.branch_out = nn.Linear(branch_hidden, 2 * p_dim)

        # -------- Trunk: 输入为 (t,z) 的 Fourier features --------
        self.fourier_dim = fourier_dim
        # B_matrix: 用来把 (t,z) 映射到 Fourier feature 空间的随机投影矩阵
        # register_buffer: 不训练，但随模型保存/搬迁device
        B_matrix = torch.randn(2, fourier_dim) * 3.0
        self.register_buffer("B", B_matrix)

        self.trunk_in = nn.Linear(2 * fourier_dim, trunk_hidden)
        self.trunk_blocks = nn.Sequential(
            ResidualBlock(trunk_hidden),
            ResidualBlock(trunk_hidden),
        )
        self.trunk_out = nn.Linear(trunk_hidden, 2 * p_dim)

        # 输出缩放（可训练）：有时能稳定训练尺度
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, u_in, t, z):
        """
        输入:
          u_in: [B, 2*N_t]   (branch 输入：一条函数/波形)
          t,z:  [B, 1]       (trunk 输入：坐标)
        输出:
          u,v:  [B, 1]       (预测复场 A(t,z)=u+jv)
        """
        # ---------------- Branch forward ----------------
        bx = torch.relu(self.branch_in(u_in))
        bx = self.branch_blocks(bx)
        bx = self.branch_out(bx)  # [B, 2*P_dim]

        # 拆成复系数：B_r + j B_i
        B_r, B_i = bx[:, :P_dim], bx[:, P_dim:]

        # ---------------- Trunk forward (Fourier features) ----------------
        # 归一化坐标，避免数值尺度太大影响训练
        t_norm = t / T_max
        z_norm = z / L
        coords = torch.cat([t_norm, z_norm], dim=1)  # [B,2]

        # 随机投影 -> sin/cos 拼接
        proj = coords @ self.B  # [B, fourier_dim]
        fourier_features = torch.cat(
            [torch.sin(2 * np.pi * proj), torch.cos(2 * np.pi * proj)], dim=1
        )  # [B, 2*fourier_dim]

        tx = torch.relu(self.trunk_in(fourier_features))
        tx = self.trunk_blocks(tx)
        tx = self.trunk_out(tx)  # [B, 2*P_dim]

        T_r, T_i = tx[:, :P_dim], tx[:, P_dim:]

        # ---------------- Complex inner product ----------------
        # (B_r + jB_i) · (T_r + jT_i)
        # real = sum(B_r*T_r - B_i*T_i)
        # imag = sum(B_r*T_i + B_i*T_r)
        real_part = torch.sum(B_r * T_r - B_i * T_i, dim=1, keepdim=True)
        imag_part = torch.sum(B_r * T_i + B_i * T_r, dim=1, keepdim=True)

        return self.output_scale * real_part, self.output_scale * imag_part


def create_model():
    """创建模型 + 优化器 + 学习率调度器"""
    model = DeepONet(N_t, P_dim, fourier_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    return model, optimizer, scheduler


# ============================================================
# 5) PDE 物理残差（NLSE in real/imag form）
# ============================================================

def physics_loss(model, u_in_batch, n_samples, z_max=1.0, chunk=1024):
    """
    在域内随机采样 (t,z) 点，计算 NLSE 残差 f_u, f_v 的均方。
    这里用 chunk 是为了避免一次性采样太多导致显存爆掉。

    输入:
      - u_in_batch: [B_fns, 2*N_t]  (B_fns 条“函数/波形”作为 batch)
      - n_samples:  物理采样点总数
    输出:
      - scalar loss
    """
    B_fns = u_in_batch.shape[0]  # 当前 batch 中有多少条“函数”
    total = 0.0
    count = 0

    for start in range(0, n_samples, chunk):
        m = min(chunk, n_samples - start)

        # 对每个物理点，随机选一条函数作为 branch 输入
        idx_fn = torch.randint(0, B_fns, (m,), device=device)
        u_in_pde = u_in_batch[idx_fn]  # [m, 2*N_t]

        # 随机采样 (t,z)
        t_f = (torch.rand(m, 1, device=device) * 2 - 1) * T_max  # [-T_max, +T_max]
        z_f = torch.rand(m, 1, device=device) * z_max            # [0, z_max]

        # 必须开梯度，才能求 u_t, u_tt, u_z 等
        t_f.requires_grad_(True)
        z_f.requires_grad_(True)

        # 网络输出 u(t,z), v(t,z)
        u, v = model(u_in_pde, t_f, z_f)
        ones = torch.ones_like(u)

        # 一阶导：对 z
        u_z = torch.autograd.grad(u, z_f, ones, create_graph=True)[0]
        v_z = torch.autograd.grad(v, z_f, ones, create_graph=True)[0]

        # 一阶、二阶导：对 t
        u_t  = torch.autograd.grad(u, t_f, ones, create_graph=True)[0]
        v_t  = torch.autograd.grad(v, t_f, ones, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_f, ones, create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t, t_f, ones, create_graph=True)[0]

        # 强度 |A|^2 = u^2 + v^2
        intensity = u**2 + v**2

        # NLSE 残差（把复方程拆成实部/虚部）
        # 你现在采用的是一种常见写法：
        f_u = u_z + 0.5 * alpha * u + 0.5 * beta2 * v_tt + gamma * intensity * v
        f_v = v_z + 0.5 * alpha * v - 0.5 * beta2 * u_tt - gamma * intensity * u

        # 残差平方的均值
        total = total + torch.mean(f_u**2 + f_v**2)
        count += 1

    return total / count


# ============================================================
# 6) 训练：Phase 1（PDE+IC） + Phase 2（PDE+IC+anchors）
# ============================================================

def train_model(model, optimizer, scheduler, A0_train, AL_clean_train, AL_noisy_train):
    # ---------------- Phase 1: physics-first ----------------
    print("\n=== Phase 1: Physics-first training (PDE + IC) ===")

    # 权重：让模型先“学会符合物理 + 初值”
    w_ic = 10.0
    w_pde = 1.0

    for ep in range(1, epochs_pretrain + 1):
        model.train()
        optimizer.zero_grad()

        # (1) 从训练集中抽一批“函数” (A0(t))
        idx_fns = torch.randint(0, N_train, (batch_size_fns,), device=device)
        A0_batch = A0_train[idx_fns]             # [Bf, N_t] complex
        u_in_batch = make_branch_input(A0_batch) # [Bf, 2*N_t] real

        # (2) IC loss: 在 z=0，强制预测 A(t,0)=A0(t)
        idx_time_ic = torch.randint(0, N_t, (N_ic,), device=device)               # 采样哪些时间点
        idx_fn_ic   = torch.randint(0, batch_size_fns, (N_ic,), device=device)    # 每个点对应哪条函数

        t_ic = t_grid[idx_time_ic].view(-1, 1)     # [N_ic,1]
        z_ic = torch.zeros_like(t_ic)              # z=0

        # 真值来自 A0_batch
        A0_ic = A0_batch[idx_fn_ic, idx_time_ic]   # [N_ic] complex
        u_true_ic = A0_ic.real.view(-1, 1)
        v_true_ic = A0_ic.imag.view(-1, 1)

        # 预测：输入对应那条函数的 branch 向量 + 坐标 (t,0)
        u_in_ic = u_in_batch[idx_fn_ic]            # [N_ic, 2*N_t]
        u_pred_ic, v_pred_ic = model(u_in_ic, t_ic, z_ic)

        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2 +
                             (v_pred_ic - v_true_ic) ** 2)

        # (3) PDE loss: 域内随机采样 (t,z)，逼残差为 0
        loss_pde_raw = physics_loss(model, u_in_batch, N_pde, z_max=L)
        # clamp：避免偶发梯度爆炸把训练打崩
        loss_pde = torch.clamp(loss_pde_raw, max=1e2)

        # (4) total loss
        loss = w_ic * loss_ic + w_pde * loss_pde

        # (5) backprop + 梯度裁剪 + 更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ep % 200 == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs_pretrain} | "
                  f"Loss={loss.item():.4e} | IC={loss_ic.item():.4e} | PDE={loss_pde_raw.item():.4e}")

    # ---------------- Phase 2: physics + data anchors ----------------
    print("\n=== Phase 2: Physics-driven + sparse data anchors (z=L) ===")

    # 思路：物理仍主导，但在端点 z=L 用少量数据“拉”到 SSFM
    w_ic    = 10.0
    w_pde   = 1.0
    w_clean = 2.0    # 端点贴 clean（追求极致拟合 SSFM clean）
    w_obs   = 0.5    # 端点贴 noisy（让模型“理解噪声统计”，但不要拟合噪声纹理）

    for k in range(1, epochs_hybrid + 1):
        ep = epochs_pretrain + k
        model.train()
        optimizer.zero_grad()

        # (1) 抽 batch_size_fns 条函数
        idx_fns = torch.randint(0, N_train, (batch_size_fns,), device=device)
        A0_batch  = A0_train[idx_fns]        # A(0,t)
        ALc_batch = AL_clean_train[idx_fns]  # A_clean(L,t)
        ALn_batch = AL_noisy_train[idx_fns]  # A_obs(L,t)

        # 注意：Phase2 也要重新构建 branch input（保证图和 batch 一致）
        u_in_batch = make_branch_input(A0_batch)

        # (2) IC loss: z=0
        idx_time_ic = torch.randint(0, N_t, (N_ic,), device=device)
        idx_fn_ic   = torch.randint(0, batch_size_fns, (N_ic,), device=device)
        t_ic = t_grid[idx_time_ic].view(-1, 1)
        z_ic = torch.zeros_like(t_ic)

        A0_ic = A0_batch[idx_fn_ic, idx_time_ic]
        u_true_ic = A0_ic.real.view(-1, 1)
        v_true_ic = A0_ic.imag.view(-1, 1)

        u_in_ic = u_in_batch[idx_fn_ic]
        u_pred_ic, v_pred_ic = model(u_in_ic, t_ic, z_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2 + (v_pred_ic - v_true_ic) ** 2)

        # (3) PDE loss: 域内残差
        loss_pde_raw = physics_loss(model, u_in_batch, N_pde, z_max=L)
        loss_pde = torch.clamp(loss_pde_raw, max=1e2)

        # (4) Data anchor at z=L: 随机抽一些 t 点，在 z=L 处对齐 SSFM
        idx_time_data = torch.randint(0, N_t, (N_data,), device=device)
        idx_fn_data   = torch.randint(0, batch_size_fns, (N_data,), device=device)
        t_data = t_grid[idx_time_data].view(-1, 1)
        z_data = torch.full_like(t_data, L)  # z=L

        # clean 真值
        ALc = ALc_batch[idx_fn_data, idx_time_data]
        u_true_clean = ALc.real.view(-1, 1)
        v_true_clean = ALc.imag.view(-1, 1)

        # noisy 观测
        ALn = ALn_batch[idx_fn_data, idx_time_data]
        u_obs = ALn.real.view(-1, 1)
        v_obs = ALn.imag.view(-1, 1)

        u_in_data = u_in_batch[idx_fn_data]
        u_pred_L, v_pred_L = model(u_in_data, t_data, z_data)

        # clean anchor（逼近 SSFM clean）
        loss_clean = torch.mean((u_pred_L - u_true_clean) ** 2 + (v_pred_L - v_true_clean) ** 2)

        # obs likelihood：相当于 (y - pred)^2 / sigma^2
        # sigma^2 从 snr_db_train 推出来（用 no_grad 不影响训练图）
        with torch.no_grad():
            obs_power = torch.mean(torch.abs(ALn) ** 2) + 1e-12
            snr_lin = 10 ** (snr_db_train / 10)
            sigma2 = obs_power / snr_lin
            sigma2_r = sigma2 / 2  # real/imag 各一半

        loss_obs = torch.mean((u_obs - u_pred_L) ** 2 + (v_obs - v_pred_L) ** 2) / (sigma2_r + 1e-12)

        # (5) total loss
        loss = w_ic * loss_ic + w_pde * loss_pde + w_clean * loss_clean + w_obs * loss_obs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if k % 200 == 0 or k == 1:
            print(f"Epoch {ep}/{epochs_pretrain + epochs_hybrid} | "
                  f"Loss={loss.item():.4e} | IC={loss_ic.item():.4e} | "
                  f"PDE={loss_pde_raw.item():.4e} | CLEAN={loss_clean.item():.4e} | OBS={loss_obs.item():.4e}")


# ============================================================
# 7) 评估 & 作图（看 amplitude、constellation、error）
# ============================================================

def evaluate_and_plot(model, A0_test, AL_clean_test, AL_noisy_test, title_suffix="Hybrid PINN DeepONet"):
    model.eval()
    output_dir = make_figure_output_dir(__file__)
    print(f"Saving evaluation figures to {output_dir}")

    # 随机挑一条测试样本画图
    idx = np.random.randint(0, len(A0_test))
    A0 = A0_test[idx]
    AL_clean = AL_clean_test[idx]
    AL_noisy = AL_noisy_test[idx]

    # 用 DeepONet 预测整条 A_pred(L,t)
    with torch.no_grad():
        # 这里 repeat(N_t,1) 的含义：
        #   trunk 需要对每个 t 点各算一次输出，但 branch 输入（同一条 A0）是固定的
        u_in = make_branch_input(A0.unsqueeze(0)).repeat(N_t, 1)  # [N_t, 2*N_t]
        t_plot = t_grid.view(-1, 1)                               # [N_t, 1]
        z_plot = torch.full_like(t_plot, L)                       # [N_t, 1]
        u_pred, v_pred = model(u_in, t_plot, z_plot)
        A_pred = (u_pred + 1j * v_pred).cpu().numpy().flatten()

    # 转 numpy 方便画图
    t_np = t_grid.cpu().numpy()
    AL_clean_np = AL_clean.cpu().numpy()
    AL_noisy_np = AL_noisy.cpu().numpy()

    # --- 图1：幅度对比 ---
    fig = plt.figure(figsize=(12, 4))
    plt.plot(t_np, np.abs(AL_noisy_np), label="SSFM+AWGN |A_obs(L)|", alpha=0.7)
    plt.plot(t_np, np.abs(AL_clean_np), label="SSFM clean |A_clean(L)|", alpha=0.7)
    plt.plot(t_np, np.abs(A_pred), '--', label="PINN pred (clean)")
    plt.xlabel("Time")
    plt.ylabel("|A|")
    plt.title(f"Amplitude at z = L ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, "amplitude_at_z_L")
    plt.show()

    # --- 图2：星座图（I/Q） ---
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(AL_noisy_np.real, AL_noisy_np.imag, s=10, alpha=0.4, label="SSFM+AWGN (obs)")
    plt.scatter(A_pred.real, A_pred.imag, s=10, alpha=0.7, label="PINN pred (clean)")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"Constellation at z = L ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    save_figure(fig, output_dir, "constellation_at_z_L")
    plt.show()

    # --- 图3：幅度误差 ---
    amp_err_clean = np.abs(A_pred) - np.abs(AL_clean_np)
    amp_err_obs   = np.abs(A_pred) - np.abs(AL_noisy_np)

    fig = plt.figure(figsize=(12, 3))
    plt.plot(t_np, amp_err_clean, label="Error vs clean")
    plt.plot(t_np, amp_err_obs,   label="Error vs obs", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Amplitude Error at z = L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, "amplitude_error_at_z_L")
    plt.show()


# ============================================================
# 8) 主程序（数据生成 -> 训练 -> 评估 -> 保存）
# ============================================================

def main():
    print("Building training dataset with SSFM...")
    A0_train, ALc_train, ALn_train = build_dataset(N_train, snr_db_train)

    print("Building test dataset with SSFM...")
    A0_test,  ALc_test,  ALn_test  = build_dataset(N_test, snr_db_eval)

    model, optimizer, scheduler = create_model()

    train_model(model, optimizer, scheduler, A0_train, ALc_train, ALn_train)
    evaluate_and_plot(model, A0_test, ALc_test, ALn_test)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
