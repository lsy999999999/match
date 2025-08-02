# 文件名: fit_unified_logit_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from load_data import load_all_group_data_for_model, load_source_scores
from config import config

# --- 1. 数据准备函数 (保持不变) ---
def prepare_scalar_data(model_key):
    # (此函数无需修改)
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None: return None, None
    
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_scores(row):
        try:
            d, ns, cp = int(row['target']), int(row['proposer']), int(row['current_partner'])
            return source_scores.get((d, ns)), source_scores.get((d, cp))
        except: return None, None
    df_taken[['Sa', 'Sb']] = df_taken.apply(get_scores, axis=1, result_type='expand')
    df_taken.dropna(subset=['Sa', 'Sb'], inplace=True)
    df_taken['score_diff'] = df_taken['Sa'] - df_taken['Sb']

    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_score(row):
        try:
            d, p = int(row['target']), int(row['proposer'])
            return source_scores.get((d, p))
        except: return None
    df_single['S'] = df_single.apply(get_single_score, axis=1)
    df_single.dropna(subset=['S'], inplace=True)
    return df_taken, df_single

# --- 2. 拟合与分析主函数 ---
def analyze_unified_model_with_logit(model_key):
    df_taken, df_single = prepare_scalar_data(model_key)
    model_label = config[model_key].get('label', model_key)
    
    # --- 步骤 A: 从“挖墙脚”场景学习通用的决策参数 beta0 和 lambda ---
    beta0_taken, lambda_taken = None, None
    if df_taken is not None and not df_taken.empty:
        X = sm.add_constant(df_taken['score_diff'])
        y = df_taken['result']
        try:
            result_taken = sm.Logit(y, X.astype(float)).fit(disp=0)
            beta0_taken = result_taken.params['const']
            lambda_taken = result_taken.params['score_diff']
            print("\n--- Unified Decision Parameters (Learned from 'Taken' Scenario) ---")
            print(result_taken.summary())
        except Exception as e:
            print(f"Could not fit model for 'Taken' scenario: {e}")
    else:
        print("Not enough 'Taken' data to learn decision parameters.")
        return

    if beta0_taken is None or lambda_taken is None:
        print("Aborting analysis as decision parameters could not be learned.")
        return

    # --- 步骤 B: 从“单身”场景学习其独立的截距 beta0_single ---
    beta0_single = None
    if df_single is not None and not df_single.empty:
        X = sm.add_constant(df_single['S'])
        y = df_single['result']
        try:
            result_single = sm.Logit(y, X.astype(float)).fit(disp=0)
            beta0_single = result_single.params['const']
            lambda_single = result_single.params['S'] # 我们也获取这个lambda以供比较
            print(f"\n--- Intercept Parameter (Learned from 'Single' Scenario) ---")
            print(result_single.summary())
        except Exception as e:
            print(f"Could not fit model for 'Single' scenario: {e}")
    else:
        print("Not enough 'Single' data to calculate S₀.")
        return

    # --- 步骤 C: 使用统一模型假设，反向求解 S₀ ---
    s0_calculated = None
    if beta0_single is not None and lambda_taken != 0:
        # 核心公式: S₀ = (β₀_taken - β₀_single) / λ_taken
        s0_calculated = (beta0_taken - beta0_single) / lambda_taken
        print(f"\n--- S₀ Calculation (Based on Unified Model Assumption) ---")
        print(f"  β₀ from 'Taken' model: {beta0_taken:.4f}")
        print(f"  β₀ from 'Single' model: {beta0_single:.4f}")
        print(f"  λ from 'Taken' model: {lambda_taken:.4f}")
        print(f"  Calculated S₀ = ({beta0_taken:.4f} - {beta0_single:.4f}) / {lambda_taken:.4f} = {s0_calculated:.2f}")

    # --- 步骤 D: 可视化 ---
    plt.figure(figsize=(16, 7))
    plt.suptitle(f'Unified Decision Model Fitting for {model_label}', fontsize=18)

    # 左图: 展示在“挖墙脚”场景下学习到的决策函数
    ax1 = plt.subplot(1, 2, 1)
    sns.regplot(x='score_diff', y='result', data=df_taken, logistic=True, ci=None,
                ax=ax1, line_kws={'color': 'red', 'label': f'Learned Rule (λ={lambda_taken:.4f})'},
                scatter_kws={'alpha': 0.2, 'color': 'blue'})
    ax1.set_title('P(Accept) vs. Score Difference (Sa - Sb)')
    ax1.set_xlabel('Score Difference (New Suitor - Current Partner)')
    ax1.set_ylabel('Probability of Acceptance')
    ax1.legend()

    # 右图: 展示如何用这个决策函数来解释“单身”场景的数据
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(x='S', y='result', data=df_single, ax=ax2, label='Raw Single Decisions (0/1)', alpha=0.2, color='purple')
    if s0_calculated is not None:
        x_fit = np.linspace(df_single['S'].min(), df_single['S'].max(), 100)
        # 使用统一的beta0_taken, lambda_taken 和计算出的s0来画出预测曲线
        z_fit = beta0_taken + lambda_taken * (x_fit - s0_calculated)
        y_fit = 1.0 / (1.0 + np.exp(-z_fit))
        ax2.plot(x_fit, y_fit, 'g-', label=f'Unified Model Fit (S₀={s0_calculated:.2f})')
        
        s_threshold_50_pct = s0_calculated - (beta0_taken / lambda_taken)
        if np.isfinite(s_threshold_50_pct):
            ax2.axvline(s_threshold_50_pct, color='r', linestyle='--', label=f'50% Threshold S₁ = {s_threshold_50_pct:.2f}')
    
    ax2.set_title('P(Accept) vs. Suitor\'s Total Score (S₁)')
    ax2.set_xlabel('Suitor\'s Total Score (S₁)')
    ax2.set_ylabel('Probability of Acceptance')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = f"fit_unified_logit_results_{model_key}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plots saved to {output_filename}")
    plt.show()

# --- 主程序 ---
if __name__ == '__main__':
    model_key_to_analyze = 'claude_zh_fitting' 
    print(f"--- Starting Unified Model Fitting (Logit-only) for: {model_key_to_analyze} ---")
    analyze_unified_model_with_logit(model_key_to_analyze)