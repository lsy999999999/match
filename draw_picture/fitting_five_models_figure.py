# -*- coding: utf-8 -*-
"""
fitting_five_models_paper_figure.py

功能：
生成一篇学术论文所需的 5 个大模型拟合结果对比组合图 (PDF格式)。
布局为 2行3列，前5个格子放子图，第6个格子（右下角）放全局图例。

需要用户配置：
在代码开头的 `MODELS_CONFIG` 字典中，为 5 个模型分别指定：
1. 拟合参数文件路径 (json)
2. 决策 CSV 文件夹路径
"""

import os
import sys
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

# =================USER CONFIGURATION START=================
# 请在此处填写您的实际路径
# 确保 load_data.py 和 config.py 所在的根目录在 sys.path 中
REPO_ROOT = "/home/lsy/match"  # 请修改为您项目的根目录

# 源分数文件路径 (Excel)
SOURCE_DATA_PATH = "/home/lsy/match/dataset/save_merge_select_null_3.xlsx"

# 定义 5 个模型的配置信息
# key 将作为图标题的一部分显示
MODELS_CONFIG = {
    "GPT-4-Turbo": {
        # 填入该模型拟合出的 json 文件路径
        "params_json": "/home/lsy/match/bahavior_simul/1009analysis/unified_params_gpt4_en_fitting.json",
        # 填入该模型决策 csv 所在的文件夹路径
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

# 输出 PDF 路径
OUTPUT_PDF = "/home/lsy/match/draw_picture/pictures/five_models_fitting_comparison.pdf"
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
    处理单个模型的数据：读取参数，读取决策，计算 Diff，分箱计算经验概率。
    返回用于绘图的数据字典，如果出错返回 None。
    """
    params_json = config["params_json"]
    decision_dir = config["decision_dir"]

    print(f"Processing model: {model_name}...")

    # 1. 加载拟合参数
    if not os.path.exists(params_json):
        print(f"  [Warning] Params file not found: {params_json}. Skipping {model_name}.")
        return None
    try:
        with open(params_json, 'r', encoding='utf-8') as f:
            params = json.load(f)
        beta0 = float(params['beta0_taken'])
        lam = float(params['lambda_taken'])
        s0 = float(params['s0'])
    except Exception as e:
        print(f"  [Error] Failed to load params for {model_name}: {e}")
        return None

    # 2. 遍历 CSV 收集数据点
    x_data = []  # Delta (Score Diff)
    y_data = []  # Decision (0/1)

    csv_pattern = os.path.join(decision_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"  [Warning] No CSV files found in {decision_dir}. Skipping {model_name}.")
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

                        # 获取新对象分数
                        s_new = score_dict.get((target_id, proposer_id))
                        if s_new is None: continue

                        # 获取当前状态分数
                        if not current_str or current_str.lower() in ['none', 'nan', '', 'null']:
                            s_current = s0  # 单身
                        else:
                            current_id = int(float(current_str))
                            s_current = score_dict.get((target_id, current_id))
                            if s_current is None: continue

                        delta = s_new - s_current
                        x_data.append(delta)
                        y_data.append(decision)

                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"  [Warning] Error reading {filepath}: {e}")

    if not x_data:
        print(f"  [Warning] No valid data points for {model_name}.")
        return None

    # 3. 数据分箱 (Binning) 计算经验概率
    df = pd.DataFrame({'delta': x_data, 'decision': y_data})
    min_x_raw, max_x_raw = df['delta'].min(), df['delta'].max()

    # 设置分箱步长，可以根据数据密度调整
    bin_step = 5.0
    # 确保覆盖所有数据，并向两侧扩展一点以便绘图
    bins = np.arange(np.floor(min_x_raw / bin_step) * bin_step - bin_step,
                     np.ceil(max_x_raw / bin_step) * bin_step + bin_step,
                     bin_step)

    df['bin'] = pd.cut(df['delta'], bins=bins)
    # 统计每个箱的均值(概率)和计数，使用 observed=True 处理 categorical 数据
    bin_stats = df.groupby('bin', observed=True)['decision'].agg(['mean', 'count']).reset_index()
    # 计算箱中心
    bin_stats['center'] = bin_stats['bin'].apply(lambda x: x.mid).astype(float)
    # 过滤掉样本太少的箱子
    valid_bins = bin_stats[bin_stats['count'] >= 15]

    if valid_bins.empty:
         print(f"  [Warning] Not enough data for binning for {model_name}.")
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


def plot_combined_figure():
    # 1. 加载源分数 (只需加载一次)
    print("Loading source scores...")
    score_dict = load_source_scores(SOURCE_DATA_PATH)
    if not score_dict:
        print("[Error] Failed to load scores.")
        return

    # 2. 处理所有模型数据
    processed_data = {}
    global_min_x = 0
    global_max_x = 0

    model_names = list(MODELS_CONFIG.keys())
    
    first_valid = True
    for name in model_names:
        res = process_model_data(name, MODELS_CONFIG[name], score_dict)
        if res:
            processed_data[name] = res
            # 更新全局坐标轴范围
            if first_valid:
                global_min_x = res['min_delta']
                global_max_x = res['max_delta']
                first_valid = False
            else:
                global_min_x = min(global_min_x, res['min_delta'])
                global_max_x = max(global_max_x, res['max_delta'])

    if not processed_data:
        print("[Error] No models generated valid data for plotting.")
        return

    # 向两侧扩展一点范围使图更好看
    x_pad = (global_max_x - global_min_x) * 0.05
    xlim_global = (global_min_x - x_pad, global_max_x + x_pad)
    # 创建平滑的曲线 X 轴数据
    x_smooth = np.linspace(xlim_global[0], xlim_global[1], 500)

    # 3. 设置绘图风格和布局
    # 使用 seaborn 的 paper 风格，字体稍大适合阅读
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    # 设置 LaTeX 字体渲染 (如果系统支持，效果更好；不支持会自动回退)
    plt.rcParams.update({
        "text.usetex": False, # 设为 True 需要系统安装 LaTeX，为兼容性暂设 False
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    fig = plt.figure(figsize=(18, 12)) # A4横向比例大致调整
    # 创建 2行3列 的网格
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    axes_list = []
    # 前两行的前两个是图
    axes_list.append(plt.subplot(gs[0, 0]))
    axes_list.append(plt.subplot(gs[0, 1]))
    axes_list.append(plt.subplot(gs[0, 2]))
    axes_list.append(plt.subplot(gs[1, 0]))
    axes_list.append(plt.subplot(gs[1, 1]))
    # 第六个位置 (gs[1, 2]) 留空给图例

    # 子图编号标签
    identifiers = ['(a)', '(b)', '(c)', '(d)', '(e)']

    # 4. 循环绘制子图
    for i, ax in enumerate(axes_list):
        if i >= len(model_names): break
        
        model_name = model_names[i]
        data = processed_data.get(model_name)
        identifier = identifiers[i]

        if data:
            # A. 绘制拟合曲线 (红线)
            z_smooth = data['beta0'] + data['lam'] * x_smooth
            y_smooth = sigmoid(z_smooth)
            ax.plot(x_smooth, y_smooth, color='#d62728', linewidth=3, alpha=0.9, zorder=2)

            # B. 绘制经验概率点 (蓝圈散点)
            # s=大小, facecolors='none'为空心圆, edgecolors设置边框色
            ax.scatter(data['emp_x'], data['emp_y'], s=80, color='#1f77b4', linewidth=0, alpha=0.9, zorder=3)

            # C. 绘制无差异点辅助线 (p=0.5, 绿虚线)
            if abs(data['lam']) > 1e-9:
                boundary_x = -data['beta0'] / data['lam']
                # 只在显示范围内绘制
                if xlim_global[0] <= boundary_x <= xlim_global[1]:
                    ax.axvline(boundary_x, color='#2ca02c', linestyle=':', linewidth=2.5, alpha=0.8, zorder=1)
            
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder=0)

            # 设置标题 (放在子图下方)
            # 使用 LaTeX 格式化数学符号
           #title_str = f"{identifier} {model_name} ($\\beta={data['beta0']:.2f}, \\lambda={data['lam']:.2f}$)"
            #ax.set_title(title_str, y=-0.25, fontsize=14)

        else:
            # 如果该模型数据缺失，显示占位符
            ax.text(0.5, 0.5, f"{model_name}\nData Not Found", ha='center', va='center', transform=ax.transAxes)
            #ax.set_title(f"{identifier} {model_name} (N/A)", y=-0.25)

        # 统一坐标轴
        ax.set_xlim(xlim_global)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # 设置轴标签 (使用数学符号)
        ax.set_xlabel(r"$\Delta(\text{score diff})$")
        ax.set_ylabel(r"$P(\text{probability of switching})$")
        
        # 移除顶部和右侧边框，更学术
        sns.despine(ax=ax)


    # 5. 在右下角空白区域绘制全局图例
    legend_ax = plt.subplot(gs[1, 2])
    legend_ax.axis('off') # 关闭坐标轴显示

    # 创建自定义图例句柄，精确匹配要求的样式
    legend_elements = [
        # 蓝色空心圆圈
        Line2D([0], [0], marker='o', color='w', label='Empirical probability (LLM sampling)',
               markerfacecolor='#1f77b4', markersize=12),
        # 红色实线
        Line2D([0], [0], color='#d62728', lw=3, label='Quantal response (logistic fit)'),
        # 绿色点状虚线 (使用垂直线符号模拟图例中的垂直虚线效果)
        Line2D([0], [0], color='#2ca02c', linestyle=':', linewidth=2.5, marker='|', markersize=15, markeredgewidth=2,
               label=r'Indifference point ($p = 0.5$)')
    ]

    # 放置图例，调整位置使其居中好看
    legend_ax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=False, labelspacing=1.5, borderpad=1.5)

    # 调整整体布局防止重叠
    plt.tight_layout()
    # 额外调整底部空间以容纳下方的标题
    plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.35)

    # 6. 保存为 PDF
    # 创建目录
    output_dir = os.path.dirname(OUTPUT_PDF)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n[Success] Combined figure saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    # 在运行前，请务必在代码开头配置好 MODELS_CONFIG 和路径
    # 检查是否已配置演示路径
    if "/path/to/your/" in MODELS_CONFIG["GPT-4-Turbo"]["params_json"]:
        print("请先在脚本开头的 MODELS_CONFIG 中填入您实际的文件路径，然后再运行。")
    else:
        plot_combined_figure()