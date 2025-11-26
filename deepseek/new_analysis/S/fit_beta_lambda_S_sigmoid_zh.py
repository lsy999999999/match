# -*- coding: utf-8 -*-
"""
fit_beta_lambda_S_sigmoid.py
- 用 sigmoid + numpy 直接拟合：
  [A] taken 场景：p = σ(beta0 + lambda * (Sa - Sb))  → 极大似然拟合 beta0, lambda
  [B] single 场景：p = σ(beta0 + lambda * (A - S))   → 固定 lambda 与 beta0，用一维求解拟合 S（阈值）
- 读取你工程里的 load_data.py / config.py，不依赖 statsmodels。
- 会把结果存成 JSON：unified_params_<model_key>.json
"""

import os, sys, json
import numpy as np
import pandas as pd

# ========= 按实际修改的常量（3 行） =========
REPO_ROOT  = "/home/lsy/match"
ANALYSIS_DIR = "/home/lsy/match/deepseek/new_analysis"
MODEL_KEY  = "deepseek_zh_fitting"   # 对应 0627_gpt4_eng
# =========================================

# 让仓库模块可 import（优先 1009analysis 下的版本）
for p in [ANALYSIS_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from load_data import load_all_group_data_for_model, load_source_scores  # noqa: E402
from config import config as CONFIG  # noqa: E402

# ---------- 数值稳定版 sigmoid ----------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

# ---------- 数据准备 ----------
def prepare_taken_and_single(model_key: str):
    """
    返回：
      x_taken = Sa - Sb, y_taken
      A_single = A(追求者总分), y_single
    """
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log is None or df_log.empty:
        raise RuntimeError("没有加载到任何日志数据（CSV）。")

    score_dict = load_source_scores(CONFIG["source_data_path"])
    if score_dict is None:
        raise RuntimeError("无法从 Excel 计算 (iid,pid)->总分。")

    # taken：有 current_partner
    df_taken = df_log[df_log['current_partner'].notna()].copy()

    def get_Sa_Sb(row):
        try:
            d, ns, cp = int(row['target']), int(row['proposer']), int(row['current_partner'])
            return score_dict.get((d, ns)), score_dict.get((d, cp))
        except Exception:
            return None, None

    df_taken[['Sa', 'Sb']] = df_taken.apply(get_Sa_Sb, axis=1, result_type='expand')
    df_taken.dropna(subset=['Sa', 'Sb', 'result'], inplace=True)
    df_taken['x'] = df_taken['Sa'].astype(float) - df_taken['Sb'].astype(float)
    x_taken = df_taken['x'].to_numpy(dtype=float)
    y_taken = df_taken['result'].to_numpy(dtype=float)

    # single：current_partner 为空（A = S_proposer）
    df_single = df_log[df_log['current_partner'].isna()].copy()

    def get_A(row):
        try:
            d, p = int(row['target']), int(row['proposer'])
            return score_dict.get((d, p))
        except Exception:
            return None

    df_single['A'] = df_single.apply(get_A, axis=1)
    df_single.dropna(subset=['A', 'result'], inplace=True)
    A_single = df_single['A'].to_numpy(dtype=float)
    y_single = df_single['result'].to_numpy(dtype=float)

    return x_taken, y_taken, A_single, y_single

# ---------- [A] 同时拟合 beta0, lambda（taken） ----------
def fit_logistic_intercept_slope(x: np.ndarray, y: np.ndarray,
                                 max_iter: int = 100, tol: float = 1e-8,
                                 ridge_eps: float = 1e-6):
    """
    用牛顿法极大似然拟合：
      p = σ(beta0 + lambda * x)
    返回 beta0, lambda
    """
    x = x.astype(float)
    y = y.astype(float)
    n = len(x)
    if n == 0:
        raise ValueError("taken 样本为空。")

    # 初值：beta0 = logit(均值)，lambda=0
    p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta0 = float(np.log(p0 / (1 - p0)))
    lam = 0.0

    for _ in range(max_iter):
        z = beta0 + lam * x
        p = sigmoid(z)

        # 梯度：g = [sum(p - y), sum( (p - y) * x )]
        r = (p - y)
        g0 = r.sum()
        g1 = (r * x).sum()
        g = np.array([g0, g1])

        # Hessian：H = Σ p(1-p) [[1, x],[x, x^2]]  + ridge
        w = p * (1 - p)
        H00 = w.sum() + ridge_eps
        H01 = (w * x).sum()
        H11 = (w * (x ** 2)).sum() + ridge_eps
        H = np.array([[H00, H01], [H01, H11]])

        # 牛顿更新
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # 回退：小步长梯度下降
            step = g * 1e-3

        beta0_new = beta0 - step[0]
        lam_new   = lam   - step[1]

        if abs(beta0_new - beta0) < tol and abs(lam_new - lam) < tol:
            beta0, lam = beta0_new, lam_new
            break
        beta0, lam = beta0_new, lam_new

    return float(beta0), float(lam)

# ---------- [B] 固定 beta0, lambda，拟合 single 的阈值 S ----------
def grad_single(A: np.ndarray, y: np.ndarray, beta0: float, lam: float, S: float) -> float:
    """
    g(S) = Σ ( y - σ(beta0 + lam*(A - S)) )
    解 g(S*)=0 为极值点（MLE）。
    """
    z = beta0 + lam * (A - S)
    p = sigmoid(z)
    return float((y - p).sum())

def fit_single_threshold(A: np.ndarray, y: np.ndarray, beta0: float, lam: float,
                         left: float = -200.0, right: float = 200.0,
                         tol: float = 1e-7, max_iter: int = 200) -> float:
    """
    二分法（必要时牛顿兜底）求解 single 阈值 S。
    """
    gl = grad_single(A, y, beta0, lam, left)
    gr = grad_single(A, y, beta0, lam, right)

    # 若端点同号，扩大区间
    expand = 0
    while gl * gr > 0 and expand < 6:
        span = right - left if right > left else 400.0
        left  -= span
        right += span
        gl = grad_single(A, y, beta0, lam, left)
        gr = grad_single(A, y, beta0, lam, right)
        expand += 1

    # 牛顿兜底
    if gl * gr > 0:
        S = 0.5 * (left + right)
        for _ in range(max_iter):
            z = beta0 + lam * (A - S)
            p = sigmoid(z)
            g = (y - p).sum()
            H = -(lam * (p * (1 - p))).sum()  # d/dS g'(S) = -lam sum p(1-p)
            if abs(H) < 1e-9:
                break
            step = g / H
            step = np.clip(step, -5.0, 5.0)
            S_new = S - step
            if abs(S_new - S) < tol:
                return float(S_new)
            S = S_new
        return float(S)

    # 正常二分
    l, r = left, right
    for _ in range(max_iter):
        m = 0.5 * (l + r)
        gm = grad_single(A, y, beta0, lam, m)
        if abs(gm) < tol:
            return float(m)
        if gl * gm > 0:
            l, gl = m, gm
        else:
            r, gr = m, gm
        if abs(r - l) < tol:
            return float(0.5 * (l + r))
    return float(0.5 * (l + r))

# ---------- 主程序 ----------
def main():
    # 1) 准备数据
    x_taken, y_taken, A_single, y_single = prepare_taken_and_single(MODEL_KEY)
    print(f"[Data] taken: {len(x_taken)}  | single: {len(A_single)}")

    # 2) 拟合 beta0, lambda（taken）
    beta0, lam = fit_logistic_intercept_slope(x_taken, y_taken)
    print("\n[A] Taken fit (sigmoid MLE)")
    print(f"beta0 = {beta0:.6f}")
    print(f"lambda = {lam:.6f}")

    # 3) 固定 beta0, lambda，拟合 single 的阈值 S
    S = fit_single_threshold(A_single, y_single, beta0, lam)
    alpha_single = beta0 - lam * S
    A50 = -alpha_single / lam  # 50% 概率的 A 阈值

    # 4) 打印 & 保存
    print("\n[B] Single threshold (fixed lambda, sigmoid MLE)")
    print(f"S (aka C) = {S:.6f}")
    print(f"alpha_single = beta0 - lambda*S = {alpha_single:.6f}")
    print(f"A@50% = -alpha_single/lambda = {A50:.6f}")

    # 训练集 NLL（可选）
    z_taken = beta0 + lam * x_taken
    p_taken = sigmoid(z_taken)
    z_single = beta0 + lam * (A_single - S)
    p_single = sigmoid(z_single)
    eps = 1e-12
    nll_taken = -(y_taken * np.log(np.clip(p_taken, eps, 1 - eps)) +
                  (1 - y_taken) * np.log(np.clip(1 - p_taken, eps, 1 - eps))).sum()
    nll_single = -(y_single * np.log(np.clip(p_single, eps, 1 - eps)) +
                   (1 - y_single) * np.log(np.clip(1 - p_single, eps, 1 - eps))).sum()
    print(f"NLL taken  = {float(nll_taken):.4f}")
    print(f"NLL single = {float(nll_single):.4f}")

    # 写 JSON（放在 1009analysis 目录，后续脚本可直接读取）
    out = {
        "beta0_taken": float(beta0),
        "lambda_taken": float(lam),
        "s0": float(S),
        "alpha_single": float(alpha_single),
        "a50_single": float(A50),
        "nll_taken": float(nll_taken),
        "nll_single": float(nll_single),
        "model_key": MODEL_KEY,
    }
    save_path = os.path.join(ANALYSIS_DIR, f"unified_params_{MODEL_KEY}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {save_path}")

if __name__ == "__main__":
    main()
