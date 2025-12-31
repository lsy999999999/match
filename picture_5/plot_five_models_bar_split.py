# -*- coding: utf-8 -*-
"""
plot_five_models_bar_split.py

功能：
将 5 个大模型的 "Random vs AI" 分组对比图拆分为 5 个独立的 PDF 文件。
- 每个文件包含一个模型的双柱图 (Double Bar Chart)。
- 字体放大，图例置于右上角。

输入：
- 自动调用计算逻辑：针对每个模型读取参数，计算 AI E[BP]，并现场生成 Random 匹配计算对应的 Random E[BP]。
"""

import os
import sys
import glob
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ================= USER CONFIGURATION =================
REPO_ROOT = "/home/lsy/match"
SOURCE_DATA_PATH = "/home/lsy/match/dataset/save_merge_select_null_3.xlsx"
N_RANDOM_ITER = 50  # 计算 Random 基准时的平均次数

MODELS_CONFIG = {
    "GPT-4-Turbo": {
        "params_json": "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json",
        "match_dir":   "/home/lsy/match/bahavior_simul/0627_gpt4_eng" 
    },
    "Claude-Sonnet": {
        "params_json": "/home/lsy/match/claude/analysis/unified_params_claude_en_fitting.json",
        "match_dir":   "/home/lsy/match/claude/0725_claude_eng"
    },
    "DeepSeek-R1": {
        "params_json": "/home/lsy/match/deepseek/new_analysis/unified_params_deepseek_en_fitting.json",
        "match_dir":   "/home/lsy/match/deepseek/0703_ds_eng"
    },
    "Gemini-Flash": {
        "params_json": "/home/lsy/match/gemini/analysis/unified_params_gemini_en_fitting.json",
        "match_dir":   "/home/lsy/match/gemini/0713_gemini_eng"
    },
    "Qwen-72B": {
        "params_json": "/home/lsy/match/qwen/new_analysis/unified_params_qwen_en_fitting.json",
        "match_dir":   "/home/lsy/match/qwen/0707_qw_eng"
    },
}

# 输出目录 (独立的PDF将保存在这里)
OUTPUT_DIR = "/home/lsy/match/picture_5/bar_plots_split"
# ============================================================

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from load_data import load_source_scores
except ImportError:
    print(f"[Error] 无法从 {REPO_ROOT} 导入 load_data。")
    sys.exit(1)

# ---------- 核心计算逻辑 ----------
def _sigmoid(x):
    if x > 30: return 1.0
    if x < -30: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(s_new, s_cur, beta0, lam, s0, truncate=False):
    if s_new is None: return 0.0
    val_cur = s0 if s_cur is None else s_cur
    if truncate and (s_new < s0): return 0.0
    return float(_sigmoid(beta0 + lam * (s_new - val_cur)))

def _expected_numbp_for_matching(matching, men_ids, women_ids, score_dict, beta0, lam, s0, truncate=False):
    total = 0.0
    for m in men_ids:
        m_partner = matching.get(m)
        if m_partner in [None, "rejected", m]:
            m_is_single = True
            val_m_cur = s0
        else:
            m_is_single = False
            val_m_cur = score_dict.get((m, int(m_partner)), s0)

        for w in women_ids:
            if not m_is_single and m_partner == w: continue
            
            s_mw = score_dict.get((m, w))
            p_m = _p_switch(s_mw, val_m_cur, beta0, lam, s0, truncate)

            w_partner = matching.get(w)
            if w_partner in [None, "rejected", w]:
                val_w_cur = s0
            else:
                val_w_cur = score_dict.get((w, int(w_partner)), s0)
            
            s_wm = score_dict.get((w, m))
            p_w = _p_switch(s_wm, val_w_cur, beta0, lam, s0, truncate)

            total += p_m * p_w
    return float(total)

def _symmetrize_matching(m_dict):
    out = {}
    for k, v in m_dict.items():
        try:
            ki = int(k)
        except: continue
        if v in ["rejected", None]: out[ki] = ki
        else:
            try:
                vi = int(v)
                if vi == ki: out[ki] = ki
                else:
                    out[ki] = vi
                    out[vi] = ki
            except: out[ki] = ki
    return out

def _extract_group_id(fname):
    import re
    base = os.path.basename(fname)
    m = re.search(r"group\D*(\d+)", base, re.IGNORECASE)
    if m: return int(m.group(1))
    return None

def generate_random_matching(men_ids, women_ids):
    m_pool = list(men_ids)
    w_pool = list(women_ids)
    random.shuffle(m_pool)
    random.shuffle(w_pool)
    matching = {}
    n_pairs = min(len(m_pool), len(w_pool))
    for i in range(n_pairs):
        m, w = m_pool[i], w_pool[i]
        matching[m] = w
        matching[w] = m
    for i in range(n_pairs, len(m_pool)): matching[m_pool[i]] = m_pool[i]
    for i in range(n_pairs, len(w_pool)): matching[w_pool[i]] = w_pool[i]
    return matching

# ---------- 数据计算处理 ----------

def process_model_data(model_name, config, score_dict, df_source):
    """计算单个模型的数据"""
    print(f"Processing {model_name}...")
    if not os.path.exists(config["params_json"]): return None
    with open(config["params_json"], 'r') as f: p = json.load(f)
    beta0, lam, s0 = float(p["beta0_taken"]), float(p["lambda_taken"]), float(p["s0"])
    
    match_files = sorted(glob.glob(os.path.join(config["match_dir"], "*.json")))
    processed_groups = set()
    results = []
    
    for fp in match_files:
        gid = _extract_group_id(fp)
        if gid is None or not (1 <= gid <= 21): continue
        if gid in processed_groups: continue
        processed_groups.add(gid)
        
        g_data = df_source[df_source['group'] == gid]
        if g_data.empty: continue
        
        if 'gender' in g_data.columns:
            men = sorted(g_data[g_data['gender']==1]['iid'].dropna().unique().astype(int).tolist())
            women = sorted(g_data[g_data['gender']==0]['iid'].dropna().unique().astype(int).tolist())
        else:
            men = sorted(g_data['iid'].dropna().unique().astype(int).tolist())
            women = sorted(g_data['pid'].dropna().unique().astype(int).tolist())
            women = [x for x in women if x not in men]
        
        # AI EBP
        try:
            with open(fp, 'r') as f: raw = json.load(f)
            ebp_ai = _expected_numbp_for_matching(_symmetrize_matching(raw), men, women, score_dict, beta0, lam, s0, truncate=False)
        except: ebp_ai = None
        
        # Random EBP
        rand_sum = 0
        valid = 0
        for _ in range(N_RANDOM_ITER):
            try:
                val = _expected_numbp_for_matching(generate_random_matching(men, women), men, women, score_dict, beta0, lam, s0, truncate=True)
                rand_sum += val
                valid += 1
            except: pass
        ebp_rand = (rand_sum / valid) if valid > 0 else None
        
        if ebp_ai is not None and ebp_rand is not None:
            results.append({'Group': gid, 'Ebp_AI': ebp_ai, 'Ebp_Random': ebp_rand})
            
    return pd.DataFrame(results).sort_values('Group')

# ---------- 绘图与保存 ----------

def save_individual_plots():
    print("Loading source scores...")
    score_dict = load_source_scores(SOURCE_DATA_PATH)
    df_source = pd.read_excel(SOURCE_DATA_PATH)
    if not score_dict: return

    # 设置绘图风格 (增大字号)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_names = list(MODELS_CONFIG.keys())
    # 颜色列表 (对应 5 个模型，与之前保持一致)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, name in enumerate(model_names):
        df = process_model_data(name, MODELS_CONFIG[name], score_dict, df_source)
        if df is None or df.empty:
            print(f"[Warning] No data for {name}")
            continue

        color = colors[i]
        
        # 创建单独的 Figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 自适应 Y 轴上限
        local_max = max(df['Ebp_AI'].max(), df['Ebp_Random'].max())
        if local_max < 1: local_max = 1
        y_limit = local_max * 1.15

        groups = df['Group']
        bar_width = 0.35 
        
        # 1. 绘制 Random Bar (Left, Gray)
        ax.bar(groups - bar_width/2, df['Ebp_Random'], 
               color='gray', alpha=0.5, width=bar_width, label='Random')
        
        # 2. 绘制 AI Bar (Right, Color)
        ax.bar(groups + bar_width/2, df['Ebp_AI'], 
               color=color, alpha=0.9, width=bar_width, label='AI Agent')
        
        # 标题 (模型名称)
        ax.set_title(name, y=1.02, fontweight='bold')
        
        # 坐标轴标签
        ax.set_ylabel("Expected Blocking Pairs (EBP)")
        ax.set_xlabel("Group ID")
            
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0, 22)
        ax.set_ylim(0, y_limit)
        
        # 图例 (右上角)
        # 显式定义 Handles 以确保颜色正确
        legend_handles = [
            mpatches.Patch(color='gray', alpha=0.5, label='Random'),
            mpatches.Patch(color=color, alpha=0.9, label='AI Agent')
        ]
        ax.legend(handles=legend_handles, loc='upper right', frameon=True, framealpha=0.9)

        sns.despine()
        plt.tight_layout()

        # 保存
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(OUTPUT_DIR, f"barplot_{safe_name}.pdf")
        
        plt.savefig(out_path, format='pdf', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

    print(f"\n[Success] All 5 individual bar plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    if "/path/to/your/" in MODELS_CONFIG["GPT-4-Turbo"]["params_json"]:
        print("请先在脚本开头的 MODELS_CONFIG 中填入您实际的文件路径，然后再运行。")
    else:
        save_individual_plots()