# -*- coding: utf-8 -*-
"""
verify_sigmoid_fitting.py

功能：
1. 读取所有决策日志 (CSV)。
2. 计算每个决策的分数差 Delta = Score_New - Score_Current (单身时 Score_Current = S0)。
3. 将 Delta 分箱(Binning)，计算每个箱内 AI 选择 "Yes" 的实际概率。
4. 绘制 "实际概率散点图" vs "拟合 Sigmoid 曲线"，以验证拟合效果。

使用前请确保:
1. 已运行拟合脚本生成了 json 参数文件。
2. config.py 和 load_data.py 在 Python 路径下。
"""

import os
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 引入项目中的模块
try:
    from load_data import load_source_scores
    from config import config
except ImportError:
    print("错误: 无法导入 load_data 或 config，请确保在项目根目录下运行。")
    exit(1)

# ================= 配置区域 =================
# 拟合生成的参数文件 (根据你的文件名调整)
PARAMS_JSON_PATH = "/home/lsy/match/deepseek/new_analysis/unified_params_deepseek_zh_fitting.json"

# 原始决策数据目录 (即 csv 所在文件夹)
# 根据你提供的文件名推测
DECISION_DIR = "/home/lsy/match/deepseek/0704_ds_Chinese" 
# ===========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # 1. 加载拟合参数 (beta0, lambda, s0)
    print(f"正在加载参数: {PARAMS_JSON_PATH}")
    if not os.path.exists(PARAMS_JSON_PATH):
        print(f"找不到参数文件: {PARAMS_JSON_PATH}")
        print("请先运行 fit_beta_lambda_S_sigmoid.py 生成参数。")
        return

    with open(PARAMS_JSON_PATH, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    beta0 = float(params['beta0_taken'])
    lam = float(params['lambda_taken'])
    s0 = float(params['s0'])
    
    print(f"参数已加载: beta0={beta0:.4f}, lambda={lam:.4f}, S0={s0:.4f}")

    # 2. 加载源分数 (加权分 0-100)
    print("正在加载源分数 (Weighted Scores)...")
    score_dict = load_source_scores(config["source_data_path"])
    if not score_dict:
        print("分数加载失败。")
        return

    # 3. 遍历 CSV 收集数据点 (Delta, Result)
    x_data = [] # Delta = S_new - S_cur
    y_data = [] # Decision (0 or 1)
    
    # 查找所有 group 的 CSV
    csv_pattern = os.path.join(DECISION_DIR, "*group*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"警告: 在 {DECISION_DIR} 下没有找到 CSV 文件，尝试在当前目录查找...")
        csv_files = glob.glob("*group*.csv")
    
    print(f"找到 {len(csv_files)} 个决策文件，开始处理...")

    count_single = 0
    count_taken = 0

    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    # load_data.py 中的标准是 6 列
                    if len(row) < 6: continue
                    
                    try:
                        # 解析 CSV 行
                        # 格式: Prompt, Reason, Target(Chooser), Proposer(New), Current, Result
                        target_id = int(row[2])
                        proposer_id = int(row[3])
                        current_str = str(row[4]).strip()
                        decision = int(row[5]) # 1=Yes, 0=No
                        
                        # 获取新对象分数
                        s_new = score_dict.get((target_id, proposer_id))
                        if s_new is None: continue # 找不到分数的跳过
                        
                        # 获取当前对象分数
                        if not current_str or current_str.lower() in ['none', 'nan', '', 'null']:
                            # 单身情况
                            s_current = s0
                            count_single += 1
                        else:
                            # 有对象情况
                            try:
                                current_id = int(float(current_str))
                                s_current = score_dict.get((target_id, current_id))
                                if s_current is None: 
                                    # 如果找不到现任分数，通常用 s0 兜底或跳过，这里选择跳过以保准确性
                                    continue 
                                count_taken += 1
                            except ValueError:
                                continue
                        
                        # 计算 Diff
                        delta = s_new - s_current
                        
                        x_data.append(delta)
                        y_data.append(decision)
                        
                    except ValueError:
                        continue
        except Exception as e:
            print(f"读取文件 {filepath} 出错: {e}")

    print(f"数据处理完成: 总样本数 {len(x_data)} (Single场景: {count_single}, Taken场景: {count_taken})")

    # 4. 数据可视化
    if not x_data:
        print("没有提取到有效数据，无法绘图。")
        return

    df = pd.DataFrame({'delta': x_data, 'decision': y_data})
    
    # --- 核心逻辑: 分箱计算实际概率 ---
    # 将 Delta 进行切分 (例如每 5 分一个箱)
    # 范围根据数据自动调整，通常在 -50 到 +50 之间
    min_x, max_x = df['delta'].min(), df['delta'].max()
    bins = np.arange(np.floor(min_x), np.ceil(max_x) + 5, 5) # 步长为5
    
    df['bin'] = pd.cut(df['delta'], bins=bins)
    
    # 计算每个箱的平均值 (即 Yes 的概率) 和样本数
    binned_stats = df.groupby('bin', observed=True)['decision'].agg(['mean', 'count']).reset_index()
    # 计算箱中心点用于绘图
    binned_stats['center'] = binned_stats['bin'].apply(lambda b: b.mid).astype(float)
    
    # 过滤掉样本太少的箱子 (避免噪点)
    valid_bins = binned_stats[binned_stats['count'] > 10]

    # --- 绘图 ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # A. 绘制实际数据的散点 (Empirical Data)
    plt.scatter(valid_bins['center'], valid_bins['mean'], 
                color='blue', s=80, alpha=0.7, edgecolors='k', zorder=5,
                label='Empirical Probability (Actual Data)')
    
    # B. 绘制拟合的 Sigmoid 曲线 (Theoretical Curve)
    x_smooth = np.linspace(min_x, max_x, 300)
    # 公式: P = sigmoid(beta0 + lambda * x)
    z_smooth = beta0 + lam * x_smooth
    y_smooth = sigmoid(z_smooth)
    
    plt.plot(x_smooth, y_smooth, color='red', linewidth=3, alpha=0.8,
             label=f'Fitted Sigmoid\n$\\beta_0={beta0:.2f}, \\lambda={lam:.2f}$')
    
    # C. 辅助线和标注
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    decision_boundary = -beta0 / lam # P=0.5 时的 x
    plt.axvline(decision_boundary, color='green', linestyle=':', linewidth=2,
                label=f'Decision Boundary (x={decision_boundary:.1f})')
    
    plt.title(f"Validation of Sigmoid Fit (S0 = {s0:.2f})\nDoes AI behavior match the curve?", fontsize=14)
    plt.xlabel("Score Difference (New Option - Current Option)", fontsize=12)
    plt.ylabel("Probability of Switching (Yes)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=10, loc='best')
    
    # 保存图片
    output_img = "/home/lsy/match/deepseek/new_analysis/S/sigmoid_verification_plot_zh.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 验证图已保存为: {output_img}")
    print("如果蓝点紧密围绕红线分布，说明 Sigmoid 拟合是合理的。")
    # plt.show() # 如果在支持显示的终端可取消注释

if __name__ == "__main__":
    main()