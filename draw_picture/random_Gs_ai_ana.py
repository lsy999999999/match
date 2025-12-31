# -*- coding: utf-8 -*-
"""
run_five_models_correct_random_comparison.py

【核心逻辑修正】
不再读取固定的 Random CSV。而是针对每一个模型 (GPT-4, Claude 等)：
1. 读取该模型的特定参数 (beta0, lambda, s0)。
2. 计算该模型的 AI 匹配结果的 E[BP]。
3. 现场生成 N 次随机匹配，并使用 *该模型的参数* 计算 Random E[BP] (取平均)。
   -> 这样保证了 Random Baseline 是在该模型特定的"价值观"下的基准。

输出：
- 2行3列 PDF 对比图 (无阴影，点+细线)。
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
import seaborn as sns

# ================= 用户配置区域 =================

# 1. 项目根目录
REPO_ROOT = "/home/lsy/match"

# 2. 原始分数 Excel 路径
SOURCE_DATA_PATH = "/home/lsy/match/dataset/save_merge_select_null_3.xlsx"

# 3. 随机实验次数 (计算 Random Baseline 时的平均次数)
N_RANDOM_ITER = 50 

# 4. 五个模型的配置信息
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

# 5. 输出 PDF 路径
OUTPUT_PDF = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/five_models_correct_random_comparison.pdf"

# ============================================================

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from load_data import load_source_scores
except ImportError:
    print(f"[Error] 无法从 {REPO_ROOT} 导入 load_data。")
    sys.exit(1)

# ---------- 核心数学逻辑 (Sigmoid & EBP) ----------

def _sigmoid(x):
    # 数值稳定 Sigmoid
    if x > 30: return 1.0
    if x < -30: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

def _p_switch(s_new, s_cur, beta0, lam, s0, truncate=False):
    if s_new is None: return 0.0
    val_cur = s0 if s_cur is None else s_cur
    
    # 截断逻辑: 如果开启，且新分数 < 单身分数，则概率为 0
    if truncate and (s_new < s0):
        return 0.0
        
    return float(_sigmoid(beta0 + lam * (s_new - val_cur)))

def _expected_numbp_for_matching(matching, men_ids, women_ids, score_dict, beta0, lam, s0, truncate=False):
    """计算给定匹配的期望阻塞对数量"""
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
            if not m_is_single and m_partner == w:
                continue
            
            # P(m wants w)
            s_mw = score_dict.get((m, w))
            p_m = _p_switch(s_mw, val_m_cur, beta0, lam, s0, truncate)

            # P(w wants m)
            w_partner = matching.get(w)
            if w_partner in [None, "rejected", w]:
                val_w_cur = s0
            else:
                val_w_cur = score_dict.get((w, int(w_partner)), s0)
            
            s_wm = score_dict.get((w, m))
            p_w = _p_switch(s_wm, val_w_cur, beta0, lam, s0, truncate)

            total += p_m * p_w
            
    return float(total)

# ---------- 辅助功能 ----------

def _symmetrize_matching(m_dict):
    out = {}
    for k, v in m_dict.items():
        try:
            ki = int(k)
        except: continue
        if v in ["rejected", None]:
            out[ki] = ki
        else:
            try:
                vi = int(v)
                if vi == ki: out[ki] = ki
                else:
                    out[ki] = vi
                    out[vi] = ki
            except:
                out[ki] = ki
    return out

def _extract_group_id(fname):
    import re
    base = os.path.basename(fname)
    m = re.search(r"group\D*(\d+)", base, re.IGNORECASE)
    if m: return int(m.group(1))
    return None

def generate_random_matching(men_ids, women_ids):
    """生成随机匹配"""
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

# ---------- 单个模型的数据处理 ----------

def process_model_data(model_name, config, score_dict, df_source):
    """
    针对单个模型：
    1. 加载其参数 (beta, lambda, s0)
    2. 计算 AI 匹配的 E[BP] (truncate=False)
    3. 生成 Random 匹配并计算 E[BP] (truncate=True, N次平均) - 使用该模型的参数
    """
    print(f"Processing {model_name}...")
    
    # 1. 加载参数
    if not os.path.exists(config["params_json"]):
        print(f"  [Error] Params not found: {config['params_json']}")
        return None
    with open(config["params_json"], 'r') as f:
        p = json.load(f)
    beta0 = float(p["beta0_taken"])
    lam = float(p["lambda_taken"])
    s0 = float(p["s0"])
    
    print(f"  Params: beta0={beta0:.2f}, lambda={lam:.2f}, s0={s0:.2f}")

    # 2. 找到所有匹配文件
    match_files = sorted(glob.glob(os.path.join(config["match_dir"], "*.json")))
    
    results = []
    
    # 遍历组
    processed_groups = set()
    
    for fp in match_files:
        gid = _extract_group_id(fp)
        if gid is None or not (1 <= gid <= 21): continue
        if gid in processed_groups: continue
        processed_groups.add(gid)
        
        # 获取人员 ID
        g_data = df_source[df_source['group'] == gid]
        if g_data.empty: continue
        
        if 'gender' in g_data.columns:
            men = sorted(g_data[g_data['gender']==1]['iid'].dropna().unique().astype(int).tolist())
            women = sorted(g_data[g_data['gender']==0]['iid'].dropna().unique().astype(int).tolist())
        else:
            men = sorted(g_data['iid'].dropna().unique().astype(int).tolist())
            women = sorted(g_data['pid'].dropna().unique().astype(int).tolist())
            women = [x for x in women if x not in men]
            
        # --- A. 计算 AI E[BP] ---
        try:
            with open(fp, 'r') as f:
                raw = json.load(f)
            ai_matching = _symmetrize_matching(raw)
            # AI 不截断
            ebp_ai = _expected_numbp_for_matching(ai_matching, men, women, score_dict, beta0, lam, s0, truncate=False)
        except Exception as e:
            print(f"  Error AI calc group {gid}: {e}")
            ebp_ai = None

        # --- B. 计算 Random E[BP] (使用当前模型的 beta0, lam, s0) ---
        # 截断 (truncate=True) 用于 Random
        rand_ebp_sum = 0
        valid_runs = 0
        for _ in range(N_RANDOM_ITER):
            try:
                rand_match = generate_random_matching(men, women)
                val = _expected_numbp_for_matching(rand_match, men, women, score_dict, beta0, lam, s0, truncate=True)
                rand_ebp_sum += val
                valid_runs += 1
            except: pass
        
        ebp_random = (rand_ebp_sum / valid_runs) if valid_runs > 0 else None
        
        if ebp_ai is not None and ebp_random is not None:
            results.append({
                'group': gid,
                'Ebp_AI': ebp_ai,
                'Ebp_Random': ebp_random
            })
            
    if not results: return None
    return pd.DataFrame(results).sort_values('group')

# ---------- 主函数 ----------

def main():
    print("Loading score dict...")
    score_dict = load_source_scores(SOURCE_DATA_PATH)
    if not score_dict: return
    df_source = pd.read_excel(SOURCE_DATA_PATH)
    
    # 1. 计算所有模型数据
    all_data = {}
    model_names = list(MODELS_CONFIG.keys())
    
    for name in model_names:
        df = process_model_data(name, MODELS_CONFIG[name], score_dict, df_source)
        if df is not None:
            all_data[name] = df
            
    if not all_data:
        print("No data generated.")
        return

    # 2. 绘图
    print("Generating Combined Plot...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({"font.family": "serif"})
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制 5 个子图
    for i in range(5):
        if i >= len(model_names): break
        name = model_names[i]
        ax = plt.subplot(gs[i // 3, i % 3])
        
        if name in all_data:
            df = all_data[name]
            
            # Plot Random (灰色虚线) - 注意：这里的 Random 曲线对每个模型都是不一样的！
            ax.plot(df['group'], df['Ebp_Random'], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                    marker='x', markersize=6, label='Random Matching')
            
            # Plot AI (彩色实线)
            ax.plot(df['group'], df['Ebp_AI'], color=colors[i], linestyle='-', linewidth=2, alpha=0.9,
                    marker='o', markersize=6, label='Hybrid GS-LLM Matching')
            
            # 标题
            ax.set_title(f"{identifiers[i]} {name}", y=-0.25, fontsize=14, fontweight='bold')
            
            # 坐标轴
            if i % 3 == 0: ax.set_ylabel("Expected Blocking Pairs (EBP)")
            if i >= 2: ax.set_xlabel("Group ID")
            
            ax.set_xticks([1, 5, 10, 15, 20])
            
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center')
            
        sns.despine(ax=ax)

    # 3. 绘制右下角图例
    legend_ax = plt.subplot(gs[1, 2])
    legend_ax.axis('off')

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, marker='x', markersize=8,
               label='Random Matching'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, marker='o', markersize=8,
               label='Hybrid GS-LLM Matching')
    ]
    
    # 添加注释说明 Random 是针对每个模型计算的
    legend_ax.text(0.5, 0.2, "*Random baseline calculated using\neach model's specific parameters", 
                   ha='center', fontsize=10, style='italic', color='gray')

    legend_ax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=False, labelspacing=1.5)

    # 4. 保存
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.4)
    
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n[Success] Correct random comparison saved to: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()