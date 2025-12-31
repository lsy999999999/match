# -*- coding: utf-8 -*-
"""
plot_five_models_gs_vs_hybrid_split_color.py

功能：
将 5 个大模型的 "Gale-Shapley vs Hybrid GS-LLM" 对比图拆分为 5 个独立的 PDF。
- 风格：彩色双柱图 (Double Bar Chart)。
- 配色：GS=灰色, Hybrid=各模型特定颜色 (与 Random 对比图保持一致)。
- 字体：大号字体，适合论文排版。

输入：
- 直接读取已生成的 CSV 结果文件 (Ebp_from_unified_json_*.csv)。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ================= USER CONFIGURATION =================

# 请确认 CSV 路径是否正确
MODELS_CONFIG = {
    "GPT-4-Turbo": {
        # 你提供的确切路径
        "csv_path": "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_from_unified_json_0627_en_21.csv"
    },
    "Claude-Sonnet": {
        "csv_path": "/home/lsy/match/claude/analysis/different_analysis/Ebp_from_unified_json_0725_en_21.csv"
    },
    "DeepSeek-R1": {
        # 推测路径，请核对
        "csv_path": "/home/lsy/match/deepseek/new_analysis/different_analysis/Ebp_from_unified_json_0703_en_21.csv"
    },
    "Gemini-Flash": {
        # 推测路径，请核对
        "csv_path": "/home/lsy/match/gemini/analysis/different_analysis/Ebp_from_unified_json_0725_en_21.csv"
    },
    "Qwen-72B": {
        # 推测路径，请核对
        "csv_path": "/home/lsy/match/qwen/new_analysis/different_analysis/Ebp_from_unified_json_0707_en_21.csv"
    },
}

# 输出目录
OUTPUT_DIR = "/home/lsy/match/picture_5/gs_vs_hybrid_plots_split"
# ======================================================

def load_data(model_name, config):
    """读取单个模型的 CSV 数据"""
    csv_path = config.get("csv_path")
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"[Warning] CSV not found for {model_name}: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # 标准化列名
        df.columns = [c.strip() for c in df.columns]
        
        # 检查必需列
        required = ['group', 'Ebp_AI', 'Ebp_Human']
        if not all(col in df.columns for col in required):
            print(f"[Error] CSV for {model_name} missing columns. Found: {df.columns}")
            return None
            
        # 过滤 1-21 组并排序
        df = df[df['group'].between(1, 21)].sort_values('group')
        return df
    except Exception as e:
        print(f"[Error] Failed to read {csv_path}: {e}")
        return None

def save_individual_plots():
    # 设置大字体风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 26,     # 标题更大
        "axes.labelsize": 24,     # 轴标签更大
        "xtick.labelsize": 20,    # 刻度更大
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5
    })

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_names = list(MODELS_CONFIG.keys())
    # 颜色列表 (与 Random 对比图保持一致)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, name in enumerate(model_names):
        df = load_data(name, MODELS_CONFIG[name])
        
        if df is None or df.empty:
            print(f"Skipping {name} due to missing data.")
            continue

        color = colors[i]
        
        # 创建单独的 Figure
        plt.figure(figsize=(12, 7)) #稍微宽一点适应大字体
        ax = plt.gca()

        # 自适应 Y 轴
        # Ebp_Human 是 GS, Ebp_AI 是 Hybrid
        local_max = max(df['Ebp_AI'].max(), df['Ebp_Human'].max())
        if local_max < 1: local_max = 1
        y_limit = local_max * 1.2 # 多留一点空间给图例

        groups = df['group']
        bar_width = 0.35 
        
        # 1. 绘制 Gale-Shapley Bar (Left, Gray) - Baseline
        ax.bar(groups - bar_width/2, df['Ebp_Human'], 
               color='gray', alpha=0.5, width=bar_width, label='Gale-Shapley')
        
        # 2. 绘制 Hybrid GS-LLM Bar (Right, Color) - Our Method
        ax.bar(groups + bar_width/2, df['Ebp_AI'], 
               color=color, alpha=0.9, width=bar_width, label='Hybrid GS-LLM')
        
        # 标题 (模型名称)
        ax.set_title(name, y=1.02, fontweight='bold')
        
        # 坐标轴标签
        ax.set_ylabel("Expected Blocking Pairs (EBP)")
        ax.set_xlabel("Group ID")
            
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0, 22)
        ax.set_ylim(0, y_limit)
        
        # 图例 (右上角)
        legend_handles = [
            mpatches.Patch(color='gray', alpha=0.5, label='Gale-Shapley'),
            mpatches.Patch(color=color, alpha=0.9, label='Hybrid GS-LLM')
        ]
        ax.legend(handles=legend_handles, loc='upper right', frameon=True, framealpha=0.95)

        sns.despine()
        plt.tight_layout()

        # 保存
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(OUTPUT_DIR, f"gs_hybrid_barplot_{safe_name}.pdf")
        
        plt.savefig(out_path, format='pdf', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

    print(f"\n[Success] All 5 individual GS vs Hybrid plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    save_individual_plots()