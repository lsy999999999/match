# -*- coding: utf-8 -*-
"""
plot_stability_comparison.py

功能：
1. 读取 "Random Baseline" 的 CSV 结果。
2. 读取 "AI" 的 CSV 结果。
3. 将两者 (Random, AI) 的 E[BP] (期望阻塞对数量) 绘制在同一张折线图上。
   - 移除了 Human (GS) 的绘制。
   - X轴: Group ID (1-21)
   - Y轴: Expected Blocking Pairs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置路径 (请核对) =================
# 1. 刚刚生成的 Random 基准 CSV
RANDOM_CSV = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_Random_Baseline_21.csv"

# 2. 之前计算的 AI 结果 CSV
MAIN_CSV   = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/Ebp_from_unified_json_0627_en_21_weighted.csv"

# 3. 图片输出路径
OUTPUT_IMG = "/home/lsy/match/bahavior_simul/1009analysis/different_analysis/stability_comparison_plot_AI_vs_Random.png"
# ===================================================

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(RANDOM_CSV):
        print(f"[Error] 找不到 Random CSV: {RANDOM_CSV}")
        return
    if not os.path.exists(MAIN_CSV):
        print(f"[Error] 找不到 Main CSV: {MAIN_CSV}")
        return

    # 2. 读取数据
    print("Loading data...")
    df_rand = pd.read_csv(RANDOM_CSV) # Columns: group, Ebp_Random
    df_main = pd.read_csv(MAIN_CSV)   # Columns: group, Ebp_AI, Ebp_Human

    # 3. 合并数据 (按 group)
    # inner join 确保只画出两个文件都有的组
    df = pd.merge(df_main, df_rand, on='group', how='inner')
    
    # 排序
    df = df.sort_values('group')
    
    print(f"Merged data for {len(df)} groups.")
    print(df[['group', 'Ebp_AI', 'Ebp_Random']].head())

    # 4. 绘图
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    # A. Random (基准线 - 也就是"烂"的上限)
    plt.plot(df['group'], df['Ebp_Random'], 
             marker='x', color='gray', linestyle='--', linewidth=1.5, alpha=0.8,
             label='Random Baseline (Lower Bound)')

    # B. AI (大模型表现)
    plt.plot(df['group'], df['Ebp_AI'], 
             marker='o', color='#1f77b4', linewidth=2.5, markersize=8,
             label='AI Decision (LLM)')

    # [已移除 Human 绘制部分]

    # 5. 美化图表
    plt.title("Stability Analysis: AI vs Random Baseline\n(Lower E[BP] means more stable)", fontsize=16, fontweight='bold')
    plt.xlabel("Group ID", fontsize=14)
    plt.ylabel("Expected Number of Blocking Pairs (E[BP])", fontsize=14)
    
    # 设置 X 轴刻度为整数
    plt.xticks(df['group'].unique())
    
    # 添加图例
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    
    # 在 AI 和 Random 之间填充颜色，强调差距 (Improvement)
    plt.fill_between(df['group'], df['Ebp_AI'], df['Ebp_Random'], 
                     color='skyblue', alpha=0.2, label='Stability Improvement')

    plt.tight_layout()

    # 6. 保存与显示
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"\n[Success] 对比图已保存至: {OUTPUT_IMG}")
    # plt.show() # 如在本地运行可取消注释

if __name__ == "__main__":
    main()