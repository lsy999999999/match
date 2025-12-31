# -*- coding: utf-8 -*-
"""
plot_five_models_gs_vs_hybrid_grayscale.py

功能：
生成 5 个大模型的 "Gale-Shapley vs Hybrid GS-LLM" 分组对比图 (黑白打印安全版)。

【核心修正】：
直接读取已生成的 CSV 结果文件，而不是在脚本内重新计算。
这保证了绘图数据与你现有的 Ebp_from_unified_json_*.csv 完全一致。

图表设计：
- 布局: 2行3列 (前5个子图，第6个图例)。
- 配色: 黑白灰阶 (GS=浅灰, Hybrid=深灰/黑)。
- Y轴: 自适应高度。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

# ================= 用户配置区域 (请确认 CSV 路径) =================

# 定义 5 个模型的配置信息
# 请将 'csv_path' 修改为你实际生成的 CSV 文件路径
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

# 输出 PDF 路径
OUTPUT_PDF = "/home/lsy/match/draw_picture/pictures/five_models_gs_vs_hybrid_grayscale_from_csv.pdf"
# ============================================================

def load_data(model_name, config):
    """读取单个模型的 CSV 数据"""
    csv_path = config.get("csv_path")
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"[Warning] CSV not found for {model_name}: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # 确保列名标准化 (处理可能的大小写差异)
        df.columns = [c.strip() for c in df.columns]
        
        # 检查必需列
        required = ['group', 'Ebp_AI', 'Ebp_Human']
        if not all(col in df.columns for col in required):
            print(f"[Error] CSV for {model_name} missing columns. Found: {df.columns}")
            return None
            
        # 过滤 1-21 组
        df = df[df['group'].between(1, 21)].sort_values('group')
        return df
    except Exception as e:
        print(f"[Error] Failed to read {csv_path}: {e}")
        return None

def main():
    print("Starting visualization from CSVs...")
    
    # 1. 读取所有数据
    all_data = {}
    model_names = list(MODELS_CONFIG.keys())
    
    for name in model_names:
        df = load_data(name, MODELS_CONFIG[name])
        if df is not None and not df.empty:
            all_data[name] = df
            
    if not all_data:
        print("No valid data loaded. Please check CSV paths in MODELS_CONFIG.")
        return

    # 2. 设置绘图风格 (纯白底 + 黑色文字)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        "font.family": "serif", 
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5
    })
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    # [配色方案: 黑白灰]
    COLOR_GS = '#E0E0E0'      # 浅灰 (Gale-Shapley)
    COLOR_HYBRID = '#333333'  # 深灰/黑 (AI/Hybrid)
    BAR_EDGE_COLOR = 'black'
    
    # --- 循环绘制 5 个子图 ---
    for i in range(5):
        if i >= len(model_names): break
        name = model_names[i]
        ax = plt.subplot(gs[i // 3, i % 3])
        
        if name in all_data:
            df = all_data[name]
            
            # 准备绘图数据
            groups = df['group']
            # 注意: CSV中的 'Ebp_Human' 对应图表中的 'Gale-Shapley'
            #       CSV中的 'Ebp_AI'    对应图表中的 'Hybrid GS-LLM'
            
            # 自适应 Y 轴
            local_max = max(df['Ebp_AI'].max(), df['Ebp_Human'].max())
            if local_max < 1: local_max = 1
            y_limit = local_max * 1.15

            bar_width = 0.35 
            
            # 1. 绘制 GS Bar (Left, Light Gray)
            ax.bar(groups - bar_width/2, df['Ebp_Human'], 
                   color=COLOR_GS, edgecolor=BAR_EDGE_COLOR, linewidth=0.8,
                   width=bar_width, label='_nolegend_')
            
            # 2. 绘制 Hybrid Bar (Right, Dark Gray)
            ax.bar(groups + bar_width/2, df['Ebp_AI'], 
                   color=COLOR_HYBRID, edgecolor=BAR_EDGE_COLOR, linewidth=0.8,
                   width=bar_width, label='_nolegend_')
            
            # 标题
            ax.set_title(f"{identifiers[i]} {name}", y=-0.25, fontsize=14, fontweight='bold', color='black')
            
            # 坐标轴标签
            if i % 3 == 0:
                ax.set_ylabel("Expected Blocking Pairs (EBP)", fontsize=12)
            else:
                ax.set_ylabel("")
                
            if i >= 3:
                ax.set_xlabel("Group ID", fontsize=12)
            elif i == 2:
                ax.set_xlabel("")
                
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlim(0, 22)
            ax.set_ylim(0, y_limit)
            
        else:
            # 数据缺失时的占位
            ax.text(0.5, 0.5, "Data Not Found\nCheck Path", ha='center', fontsize=12)
            ax.set_title(f"{identifiers[i]} {name}", y=-0.25)
            
        sns.despine(ax=ax)

    # --- 右下角图例 ---
    legend_ax = plt.subplot(gs[1, 2])
    legend_ax.axis('off')
    
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_GS, edgecolor=BAR_EDGE_COLOR, label='Gale-shapley'),
        mpatches.Patch(facecolor=COLOR_HYBRID, edgecolor=BAR_EDGE_COLOR, label='Hybrid GS-LLM'),
    ]
    
    # 底部标题 (居中放置于图例区域或整个画布下方，这里放在图例上方作为子图f的替代)
    legend_ax.text(0.5, 0.7, "Comparison Legend", ha='center', fontsize=14, fontweight='bold', color='black')
    
    legend_ax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=False)
    
    # 底部总标题
    plt.figtext(0.5, 0.02, "Expected Blocking Pairs: GS vs Hybrid", 
                ha="center", fontsize=16, fontweight='bold', color='black')

    # --- 保存 ---
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.45)
    
    output_dir = os.path.dirname(OUTPUT_PDF)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n[Success] Chart saved to: {OUTPUT_PDF}")
    print("请检查生成的 PDF 是否与您的 CSV 数据一致。")

if __name__ == "__main__":
    main()