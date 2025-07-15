import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from load_data import load_all_group_data_for_model, load_source_scores
from config import config


# Sa 和 Sb 的准确含义:
# Sa: 新追求者(a)的总分。这个分数是决策者（被求偶的人）给新追求者打的。
# 它是对新追求者在六个维度（attractive, sincere, etc.）上的评分的总和。
# Sb: 现任伴侣(b)的总分。这个分数也是决策者给自己的现任伴侣打的。它是对现任伴侣在六个维度上的评分的总和。
# 一句话总结: Sa 和 Sb 是同一个决策者，基于同一套标准，对两个不同的人（新来的和现在的）给出的综合评价分数。
# λ (Lambda) 的计算方式和含义 (挖墙脚场景):
# 自变量: x = Sa - Sb (新追求者总分 - 现任伴侣总分)。这个值可以是正数（新来的更好）、负数（新来的更差）或零。
# 因变量: y = result (1代表接受换人, 0代表拒绝)。
# 模型: 我们假设接受的概率 P 服从 Sigmoid 函数: P(accept) = 1 / (1 + exp(-(β₀ + λ * (Sa - Sb))))。
# 计算: 我们使用 scipy.optimize.curve_fit，将大量的 (Sa - Sb, result) 数据点喂给它，[直接]
# 让它找到最优的 β₀ 和 λ，使得Sigmoid曲线能最好地穿过这些数据点。
# λ 的含义: λ 是一个标量（单个数值），它代表了模型对分数差异的敏感度。
# λ > 0: 意味着 Sa - Sb 越大，接受的概率越高。这是我们期望的理性行为。
# λ 的值越大，S型曲线越陡峭，说明模型对分数差异的反应越剧烈，决策越“果断”。
# λ 接近0，S型曲线越平缓，说明模型对分数差异不敏感，决策更随机。
# S_threshold (单身场景下的接受阈值) 的计算方式和含义:
# 自变量: x = S (单身追求者的总分)。
# 因变量: y = result (1代表接受, 0代表拒绝)。
# 模型: 我们同样假设接受的概率 P 服从 Sigmoid 函数: P(accept) = 1 / (1 + exp(-(β₀ + λₛ * S)))。
# 这里的 λₛ 是总分 S 的系数。
# 计算:
# 第一步，我们用 curve_fit 找到最优的 β₀ 和 λₛ。
# 第二步，我们定义阈值 S_threshold 为**接受概率恰好为50%**的那个分数点。
# 在Sigmoid函数中，当 P = 0.5 时，其线性部分 β₀ + λₛ * S 必须等于0。
# 因此，我们通过求解方程 β₀ + λₛ * S_threshold = 0 来得到阈值：S_threshold = -β₀ / λₛ。
# S_threshold 的含义: S_threshold 是一个标量（单个数值），它代表了模型对一个全新的追求者的**“心理及格线”**。
# 当追求者的总分 S 低于这个阈值时，被接受的概率低于50%；当分数高于这个阈值时，被接受的概率高于50%。

def sigmoid_model(x, beta0, lambda_param):
    """标准的二维Sigmoid函数"""
    # 增加一个很小的数到分母，防止除以零的警告
    return 1.0 / (1.0 + np.exp(-(beta0 + lambda_param * x)))

# --- 2. 数据准备与分箱 ---

def prepare_scalar_data(model_key):
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None: return None, None
    
    # a. "挖墙脚"场景
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_scores(row):
        try:
            decision_maker, new_suitor, current_partner = int(row['target']), int(row['proposer']), int(row['current_partner'])
            Sa = source_scores.get((decision_maker, new_suitor))
            Sb = source_scores.get((decision_maker, current_partner))
            return Sa, Sb
        except (ValueError, TypeError): return None, None
    df_taken[['Sa', 'Sb']] = df_taken.apply(get_scores, axis=1, result_type='expand')
    df_taken.dropna(subset=['Sa', 'Sb'], inplace=True)
    df_taken['score_diff'] = df_taken['Sa'] - df_taken['Sb']

    # b. "向单身求偶"场景
    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_score(row):
        try:
            decision_maker, proposer = int(row['target']), int(row['proposer'])
            return source_scores.get((decision_maker, proposer))
        except (ValueError, TypeError): return None
    df_single['S'] = df_single.apply(get_single_score, axis=1)
    df_single.dropna(subset=['S'], inplace=True)

    # --- 数据分箱 (Binning) ---
    binned_taken, binned_single = None, None
    if not df_taken.empty:
        df_taken['diff_bin'] = pd.cut(df_taken['score_diff'], bins=20)
        binned_taken = df_taken.groupby('diff_bin', observed=False)['result'].mean().reset_index()
        binned_taken['diff_mid'] = binned_taken['diff_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
        binned_taken.dropna(subset=['diff_mid', 'result'], inplace=True)
    if not df_single.empty:
        df_single['s_bin'] = pd.cut(df_single['S'], bins=15)
        binned_single = df_single.groupby('s_bin', observed=False)['result'].mean().reset_index()
        binned_single['s_mid'] = binned_single['s_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
        binned_single.dropna(subset=['s_mid', 'result'], inplace=True)

    return binned_taken, binned_single


# --- 3. 拟合与绘图 ---

def fit_and_plot_scalar(model_key):
    binned_taken, binned_single = prepare_scalar_data(model_key)
    model_label = config[model_key].get('label', model_key)
    
    plt.figure(figsize=(16, 7))
    plt.suptitle(f'Sigmoid Decision Model Fitting for {model_label}', fontsize=18)

    # --- a. 拟合与绘制 Lambda ---
    ax1 = plt.subplot(1, 2, 1)
    if binned_taken is not None and not binned_taken.empty:
        # 【关键修正】将Pandas Series转换为纯粹的NumPy数组再进行拟合
        x_data = binned_taken['diff_mid'].to_numpy()
        y_data = binned_taken['result'].to_numpy()
        try:
            popt, _ = curve_fit(sigmoid_model, x_data, y_data, p0=[0, 0.1], maxfev=5000)
            beta0_fit, lambda_fit = popt
            print(f"\n--- Sigmoid Fitting for λ ('Taken' Scenario) ---")
            print(f"Fitted Function: P(accept) = 1 / (1 + exp(-({beta0_fit:.4f} + {lambda_fit:.4f} * (Sa-Sb))))")
            print(f"Estimated λ (Sensitivity to score difference): {lambda_fit:.4f}")

            sns.scatterplot(x='diff_mid', y='result', data=binned_taken, ax=ax1, label='Binned Acceptance Rate')
            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = sigmoid_model(x_fit, beta0_fit, lambda_fit)
            ax1.plot(x_fit, y_fit, 'r-', label=f'Sigmoid Fit (λ={lambda_fit:.4f})')
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit Sigmoid model for lambda: {e}")
            sns.scatterplot(x='diff_mid', y='result', data=binned_taken, ax=ax1, label='Binned Acceptance Rate')
    ax1.set_title('P(Accept) vs. Score Difference (Sa - Sb)')
    ax1.set_xlabel('Score Difference (New Suitor - Current Partner)')
    ax1.set_ylabel('Average Acceptance Rate')
    ax1.axvline(0, color='grey', linestyle='--', lw=1)
    ax1.axhline(0.5, color='grey', linestyle=':', lw=1)
    ax1.legend()

    # --- b. 拟合与绘制 阈值 S ---
    ax2 = plt.subplot(1, 2, 2)
    if binned_single is not None and not binned_single.empty:
        # 【关键修正】转换为NumPy数组
        x_data = binned_single['s_mid'].to_numpy()
        y_data = binned_single['result'].to_numpy()
        try:
            popt, _ = curve_fit(sigmoid_model, x_data, y_data, p0=[0, 0.1], maxfev=5000)
            beta0_fit_s, lambda_s_fit = popt
            s_threshold_fit = -beta0_fit_s / lambda_s_fit if lambda_s_fit != 0 else np.inf
            
            print(f"\n--- Sigmoid Fitting for Threshold S ('Single' Scenario) ---")
            print(f"Fitted Function: P(accept) = 1 / (1 + exp(-({beta0_fit_s:.4f} + {lambda_s_fit:.4f} * S)))")
            print(f"Estimated Acceptance Threshold S (at 50% probability): {s_threshold_fit:.2f}")

            sns.scatterplot(x='s_mid', y='result', data=binned_single, ax=ax2, label='Binned Acceptance Rate')
            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = sigmoid_model(x_fit, beta0_fit_s, lambda_s_fit)
            ax2.plot(x_fit, y_fit, 'g-', label='Sigmoid Fit')
            if np.isfinite(s_threshold_fit):
                ax2.axvline(s_threshold_fit, color='r', linestyle='--', label=f'50% Threshold S = {s_threshold_fit:.2f}')
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit Sigmoid model for threshold S: {e}")
            sns.scatterplot(x='s_mid', y='result', data=binned_single, ax=ax2, label='Binned Acceptance Rate')
    ax2.set_title('P(Accept) vs. Suitor\'s Total Score (S)')
    ax2.set_xlabel('Suitor\'s Total Score (S)')
    ax2.set_ylabel('Average Acceptance Rate')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = f"fit_scalar_sigmoid_results_{model_key}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plots saved to {output_filename}")
    plt.show()

# --- 主程序 ---
if __name__ == '__main__':
    model_key_to_analyze = 'deepseek_zh_fitting' 
    print(f"--- Starting Scalar Sigmoid Fitting Analysis for: {model_key_to_analyze} ---")
    fit_and_plot_scalar(model_key_to_analyze)


# --- Starting Scalar Sigmoid Fitting Analysis for: gpt4_zh_fitting ---
# 源数据Excel评分已成功加载并处理。

# --- Sigmoid Fitting for λ ('Taken' Scenario) ---
# Fitted Function: P(accept) = 1 / (1 + exp(-(0.0429 + 0.0229 * (Sa-Sb))))
# Estimated λ (Sensitivity to score difference): 0.0229

# --- Sigmoid Fitting for Threshold S ('Single' Scenario) ---
# Fitted Function: P(accept) = 1 / (1 + exp(-(1.0555 + 0.0322 * S)))
# Estimated Acceptance Threshold S (at 50% probability): -32.75

# Analysis plots saved to fit_scalar_sigmoid_results_gpt4_zh_fitting.png