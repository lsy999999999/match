# -*- coding: utf-8 -*-
"""
plot_five_models_group_bar_subplots.py

功能：
生成 5 个大模型的 "Random vs AI" 分组对比图 (双条形图版)。
- 布局：2行3列 (前5个为子图，第6个为图例)。
- 内容：每个子图中，X轴为 Group ID (1-21)，Y轴为 E[BP]。
- 逻辑：
    1. 灰色柱子 (Left Bar): 代表该组的 Random Baseline E[BP]。
    2. 彩色柱子 (Right Bar): 代表该组的 AI E[BP]。
    3. 并排展示，直观对比高度差。

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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

# ================= 用户配置区域 =================
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

OUTPUT_PDF = "/home/lsy/match/draw_picture/pictures/five_models_double_bar.pdf"
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
    """计算单个模型的数据：返回 DataFrame [Group, Ebp_AI, Ebp_Random]"""
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
            # AI 不截断
            ebp_ai = _expected_numbp_for_matching(_symmetrize_matching(raw), men, women, score_dict, beta0, lam, s0, truncate=False)
        except: ebp_ai = None
        
        # Random EBP (使用当前模型参数)
        rand_sum = 0
        valid = 0
        for _ in range(N_RANDOM_ITER):
            try:
                # Random 截断
                val = _expected_numbp_for_matching(generate_random_matching(men, women), men, women, score_dict, beta0, lam, s0, truncate=True)
                rand_sum += val
                valid += 1
            except: pass
        ebp_rand = (rand_sum / valid) if valid > 0 else None
        
        if ebp_ai is not None and ebp_rand is not None:
            results.append({'Group': gid, 'Ebp_AI': ebp_ai, 'Ebp_Random': ebp_rand})
            
    return pd.DataFrame(results).sort_values('Group')

# ---------- 主函数 (绘图) ----------

def main():
    print("Loading data...")
    score_dict = load_source_scores(SOURCE_DATA_PATH)
    df_source = pd.read_excel(SOURCE_DATA_PATH)
    if not score_dict: return
    
    # 1. 计算所有模型数据
    all_data = {}
    model_names = list(MODELS_CONFIG.keys())
    
    for name in model_names:
        df = process_model_data(name, MODELS_CONFIG[name], score_dict, df_source)
        if df is not None and not df.empty:
            all_data[name] = df
            
    if not all_data:
        print("No valid data calculated.")
        return

    # 2. 绘图设置
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({"font.family": "serif"})
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # --- 循环绘制 5 个子图 ---
    for i in range(5):
        if i >= len(model_names): break
        name = model_names[i]
        ax = plt.subplot(gs[i // 3, i % 3])
        
        if name in all_data:
            df = all_data[name]
            color = colors[i]
            
            # 计算当前模型的 Y 轴最大值，并留出 15% 余量 (自适应Y轴)
            local_max = max(df['Ebp_AI'].max(), df['Ebp_Random'].max())
            if local_max < 1: local_max = 1
            y_limit = local_max * 1.15

            # X 轴坐标 (1..21)
            groups = df['Group']
            bar_width = 0.35 # 柱宽
            
            # 1. 绘制 Random Bar (左侧柱)
            # 位置偏移 -width/2
            ax.bar(groups - bar_width/2, df['Ebp_Random'], color='gray', alpha=0.6, width=bar_width, label='_nolegend_')
            
            # 2. 绘制 AI Bar (右侧柱)
            # 位置偏移 +width/2
            ax.bar(groups + bar_width/2, df['Ebp_AI'], color=color, alpha=0.9, width=bar_width, label='_nolegend_')
            
            # 标题 (放在下方)
            ax.set_title(f"{identifiers[i]} {name}", y=-0.25, fontsize=14, fontweight='bold')
            
            # 坐标轴标签
            if i % 3 == 0:
                ax.set_ylabel("Expected Blocking Pairs (EBP)", fontsize=12)
            else:
                ax.set_ylabel("")
                
            if i >= 3:
                ax.set_xlabel("Group ID", fontsize=12)
            elif i == 2:
                ax.set_xlabel("")
                
            # 设置刻度
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlim(0, 22)
            ax.set_ylim(0, y_limit)
            
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center')
            ax.set_title(f"{identifiers[i]} {name} (N/A)", y=-0.25)
            
        sns.despine(ax=ax)

    # --- 右下角图例 ---
    legend_ax = plt.subplot(gs[1, 2])
    legend_ax.axis('off')
    
    # 构造自定义图例
    legend_elements = [
        mpatches.Patch(color='gray', alpha=0.6, label='Random Baseline'),
        mpatches.Patch(color='black', alpha=0.9, label='AI Agent Performance'),
    ]
    
    # 底部标题
    plt.figtext(0.5, 0.02, "Expected Blocking Pairs across Groups", 
                ha="center", fontsize=16, fontweight='bold', color='black')
    
    # 添加解释文字
    legend_ax.text(0.5, 0.65, "Interpretation:", ha='center', fontsize=12, fontweight='bold')
    legend_ax.text(0.5, 0.45, "Comparing the height of bars:\nLower AI bars indicate better stability.", 
                   ha='center', fontsize=11, style='italic', color='#333333')
    
    legend_ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.35), 
                     fontsize=12, frameon=False)

    # --- 保存 ---
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.35)
    
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n[Success] Double bar comparison chart saved to: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()