# -*- coding: utf-8 -*-
"""
fitting_five_models_individual.py

功能：
分别生成 5 个独立的 PDF 文件，每个文件包含一个模型的拟合散点图。
- 去除图例 (No Legend)。
- 增大坐标轴标签字号 (Larger Axis Labels)。
- 保持统一的坐标轴范围 (Global Axis Limits)。
"""

import os
import sys
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================USER CONFIGURATION START=================
# 请在此处填写您的实际路径
REPO_ROOT = "/home/lsy/match"  # 请修改为您项目的根目录

# 源分数文件路径 (Excel)
SOURCE_DATA_PATH = "/home/lsy/match/dataset/save_merge_select_null_3.xlsx"

# 定义 5 个模型的配置信息
MODELS_CONFIG = {
    "GPT-4-Turbo": {
        "params_json": "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json",
        "decision_dir": "/home/lsy/match/bahavior_simul/0627_gpt4_eng"
    },
    "Claude-Sonnet": {
        "params_json": "/home/lsy/match/claude/analysis/unified_params_claude_en_fitting.json",
        "decision_dir": "/home/lsy/match/claude/0725_claude_eng"
    },
    "DeepSeek-R1": {
        "params_json": "/home/lsy/match/deepseek/new_analysis/unified_params_deepseek_en_fitting.json",
        "decision_dir": "/home/lsy/match/deepseek/0703_ds_eng"
    },
    "Gemini-Flash": {
        "params_json": "/home/lsy/match/gemini/analysis/unified_params_gemini_en_fitting.json",
        "decision_dir": "/home/lsy/match/gemini/0713_gemini_eng"
    },
    "Qwen-72B": {
        "params_json": "/home/lsy/match/qwen/new_analysis/unified_params_qwen_en_fitting.json",
        "decision_dir": "/home/lsy/match/qwen/0707_qw_eng"
    },
}

# 输出文件夹 (生成的5个PDF将保存在这里)
OUTPUT_DIR = "/home/lsy/match/picture_5/fitting_plots_split"
# =================USER CONFIGURATION END=================

# 添加项目路径以导入模块
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from load_data import load_source_scores
except ImportError:
    print(f"[Error] 无法从 {REPO_ROOT} 导入 load_data。请检查路径。")
    sys.exit(1)


def sigmoid(x):
    """数值稳定的 Sigmoid 函数"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def process_model_data(model_name, config, score_dict):
    """
    处理单个模型的数据
    """
    params_json = config["params_json"]
    decision_dir = config["decision_dir"]

    print(f"Processing model: {model_name}...")

    if not os.path.exists(params_json):
        print(f"  [Warning] Params file not found: {params_json}. Skipping.")
        return None
    try:
        with open(params_json, 'r', encoding='utf-8') as f:
            params = json.load(f)
        beta0 = float(params['beta0_taken'])
        lam = float(params['lambda_taken'])
        s0 = float(params['s0'])
    except Exception as e:
        print(f"  [Error] Failed to load params: {e}")
        return None

    x_data = []
    y_data = []

    csv_pattern = os.path.join(decision_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        return None

    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6: continue
                    try:
                        target_id = int(row[2])
                        proposer_id = int(row[3])
                        current_str = str(row[4]).strip()
                        decision = int(row[5])

                        s_new = score_dict.get((target_id, proposer_id))
                        if s_new is None: continue

                        if not current_str or current_str.lower() in ['none', 'nan', '', 'null']:
                            s_current = s0
                        else:
                            current_id = int(float(current_str))
                            s_current = score_dict.get((target_id, current_id))
                            if s_current is None: continue

                        delta = s_new - s_current
                        x_data.append(delta)
                        y_data.append(decision)

                    except (ValueError, IndexError):
                        continue
        except Exception:
            continue

    if not x_data:
        return None

    # 分箱逻辑
    df = pd.DataFrame({'delta': x_data, 'decision': y_data})
    min_x_raw, max_x_raw = df['delta'].min(), df['delta'].max()

    bin_step = 5.0
    bins = np.arange(np.floor(min_x_raw / bin_step) * bin_step - bin_step,
                     np.ceil(max_x_raw / bin_step) * bin_step + bin_step,
                     bin_step)

    df['bin'] = pd.cut(df['delta'], bins=bins)
    bin_stats = df.groupby('bin', observed=True)['decision'].agg(['mean', 'count']).reset_index()
    bin_stats['center'] = bin_stats['bin'].apply(lambda x: x.mid).astype(float)
    valid_bins = bin_stats[bin_stats['count'] >= 15]

    if valid_bins.empty:
         return None

    return {
        'beta0': beta0,
        'lam': lam,
        's0': s0,
        'emp_x': valid_bins['center'].values,
        'emp_y': valid_bins['mean'].values,
        'min_delta': min_x_raw,
        'max_delta': max_x_raw
    }


def save_individual_plots():
    # 1. 加载分数
    print("Loading source scores...")
    score_dict = load_source_scores(SOURCE_DATA_PATH)
    if not score_dict:
        print("[Error] Failed to load scores.")
        return

    # 2. 处理所有数据并计算全局范围 (Global Limits)
    # 为了让5张图的横坐标看起来比例一致，我们先计算出所有模型的最大最小值
    processed_data = {}
    global_min_x = 0
    global_max_x = 0
    first_valid = True

    model_names = list(MODELS_CONFIG.keys())
    
    for name in model_names:
        res = process_model_data(name, MODELS_CONFIG[name], score_dict)
        if res:
            processed_data[name] = res
            if first_valid:
                global_min_x = res['min_delta']
                global_max_x = res['max_delta']
                first_valid = False
            else:
                global_min_x = min(global_min_x, res['min_delta'])
                global_max_x = max(global_max_x, res['max_delta'])

    if not processed_data:
        print("[Error] No valid data generated.")
        return

    # 设置统一的 X 轴范围
    x_pad = (global_max_x - global_min_x) * 0.05
    xlim_global = (global_min_x - x_pad, global_max_x + x_pad)
    x_smooth = np.linspace(xlim_global[0], xlim_global[1], 500)

    # 3. 设置绘图风格 (增大字号)
    # font_scale=1.5 会让整体字体变大
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        "font.family": "serif",
        #"axes.titlesize": 18,     # 标题字号
        "axes.labelsize": 20,     # 【关键】横纵轴标签字号 (调大)
        "xtick.labelsize": 16,    # 刻度字号
        "ytick.labelsize": 16,
        "text.usetex": False
    })

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)']

    # 4. 循环生成并保存每一张图
    for i, model_name in enumerate(model_names):
        if model_name not in processed_data:
            continue
            
        data = processed_data[model_name]
        identifier = identifiers[i] if i < len(identifiers) else ""

        # 创建单独的 Figure
        # figsize 可以根据单张图的需求调整，这里设为正方形偏宽一点
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # A. 绘制拟合曲线 (红线)
        z_smooth = data['beta0'] + data['lam'] * x_smooth
        y_smooth = sigmoid(z_smooth)
        ax.plot(x_smooth, y_smooth, color='#d62728', linewidth=3, alpha=0.9, zorder=2)

        # B. 绘制经验概率点 (蓝色实心圆)
        ax.scatter(data['emp_x'], data['emp_y'], s=120, color='#1f77b4', linewidth=0, alpha=0.9, zorder=3)

        # C. 绘制无差异点辅助线 (p=0.5)
        if abs(data['lam']) > 1e-9:
            boundary_x = -data['beta0'] / data['lam']
            if xlim_global[0] <= boundary_x <= xlim_global[1]:
                ax.axvline(boundary_x, color='#2ca02c', linestyle=':', linewidth=3, alpha=0.8, zorder=1)
        
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=0)

        # 设置坐标轴范围
        ax.set_xlim(xlim_global)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # 设置标签 (字号在 rcParams 中已统一调大)
        ax.set_xlabel(r"$\Delta(\text{score diff})$")
        ax.set_ylabel(r"$P(\text{probability of switching})$")

        # 标题 (放到下方，模拟 LaTeX caption 风格，或者直接用子图标题)
        # 这里的标题保留模型名称和参数，方便你在 Overleaf 里面配对
        #title_str = f"{identifier} {model_name} ($\\beta={data['beta0']:.2f}, \\lambda={data['lam']:.2f}$)"
        # y=-0.25 将标题放置在 X 轴下方
        #ax.set_title(title_str, y=-0.25, fontsize=18, fontweight='bold')

        # 去除边框
        sns.despine(ax=ax)

        # 紧凑布局
        plt.tight_layout()
        # 如果标题被切掉了，可以手动调整底部边距
        plt.subplots_adjust(bottom=0.22)

        # 保存为单独 PDF
        # 文件名例如: fitting_GPT-4-Turbo.pdf
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(OUTPUT_DIR, f"fitting_{safe_name}.pdf")
        
        plt.savefig(out_path, format='pdf', dpi=300)
        print(f"Saved: {out_path}")
        
        # 关闭画布，释放内存
        plt.close()

    print(f"\n[Success] All 5 individual plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if "/path/to/your/" in MODELS_CONFIG["GPT-4-Turbo"]["params_json"]:
        print("请先在脚本开头的 MODELS_CONFIG 中填入您实际的文件路径，然后再运行。")
    else:
        save_individual_plots()