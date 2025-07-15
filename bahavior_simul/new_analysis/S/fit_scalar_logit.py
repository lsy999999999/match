# 文件名: fit_scalar_logit.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # 引入statsmodels

from load_data import load_all_group_data_for_model, load_source_scores
from config import config

# --- 1. 数据准备 (不再需要分箱) ---

def prepare_scalar_data_for_logit(model_key):
    """准备用于逻辑回归的原始0/1数据"""
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None: return None, None
    
    # a. "挖墙脚"场景
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_scores(row):
        try:
            d, ns, cp = int(row['target']), int(row['proposer']), int(row['current_partner'])
            return source_scores.get((d, ns)), source_scores.get((d, cp))
        except (ValueError, TypeError): return None, None
    df_taken[['Sa', 'Sb']] = df_taken.apply(get_scores, axis=1, result_type='expand')
    df_taken.dropna(subset=['Sa', 'Sb'], inplace=True)
    df_taken['score_diff'] = df_taken['Sa'] - df_taken['Sb']

    # b. "向单身求偶"场景
    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_score(row):
        try:
            d, p = int(row['target']), int(row['proposer'])
            return source_scores.get((d, p))
        except (ValueError, TypeError): return None
    df_single['S'] = df_single.apply(get_single_score, axis=1)
    df_single.dropna(subset=['S'], inplace=True)

    return df_taken, df_single

# --- 2. 拟合与绘图 (使用Logit) ---

def fit_and_plot_logit(model_key):
    # 【修改】调用新的数据准备函数
    df_taken, df_single = prepare_scalar_data_for_logit(model_key)
    model_label = config[model_key].get('label', model_key)
    
    plt.figure(figsize=(16, 7))
    plt.suptitle(f'Logistic Regression (Sigmoid) Fitting for {model_label}', fontsize=18)

    # --- a. 拟合与绘制 Lambda ---
    ax1 = plt.subplot(1, 2, 1)
    if df_taken is not None and not df_taken.empty:
        # 准备Logit模型的数据
        X = df_taken['score_diff']
        y = df_taken['result']
        X = sm.add_constant(X) # 添加截距项
        
        # 拟合Logit模型
        logit_model_taken = sm.Logit(y, X.astype(float))
        result_taken = logit_model_taken.fit(disp=0)
        
        # 提取参数
        beta0_fit = result_taken.params['const']
        lambda_fit = result_taken.params['score_diff']
        
        print("\n--- Logistic Regression for λ ('Taken' Scenario) ---")
        print(result_taken.summary())
        print(f"Estimated λ (Sensitivity to score difference): {lambda_fit:.4f}")

        # 可视化
        sns.regplot(x='score_diff', y='result', data=df_taken, logistic=True, ci=None,
                    ax=ax1, line_kws={'color': 'red', 'label': f'Logit Fit (λ={lambda_fit:.4f})'},
                    scatter_kws={'alpha': 0.2, 'color': 'blue', 'label': 'Raw Decisions (0/1)'})
    
    ax1.set_title('P(Accept) vs. Score Difference (Sa - Sb)')
    ax1.set_xlabel('Score Difference (New Suitor - Current Partner)')
    ax1.set_ylabel('Probability of Acceptance')
    ax1.axvline(0, color='grey', linestyle='--', lw=1)
    ax1.axhline(0.5, color='grey', linestyle=':', lw=1)
    ax1.legend()

    # --- b. 拟合与绘制 阈值 S ---
    ax2 = plt.subplot(1, 2, 2)
    if df_single is not None and not df_single.empty:
        X = df_single['S']
        y = df_single['result']
        X = sm.add_constant(X)
        
        logit_model_single = sm.Logit(y, X.astype(float))
        result_single = logit_model_single.fit(disp=0)
        
        beta0_fit_s = result_single.params['const']
        lambda_s_fit = result_single.params['S']
        
        s_threshold_fit = -beta0_fit_s / lambda_s_fit if lambda_s_fit != 0 else np.inf
            
        print(f"\n--- Logistic Regression for Threshold S ('Single' Scenario) ---")
        print(result_single.summary())
        print(f"Estimated Acceptance Threshold S (at 50% probability): {s_threshold_fit:.2f}")

        sns.regplot(x='S', y='result', data=df_single, logistic=True, ci=None,
                    ax=ax2, line_kws={'color': 'green', 'label': 'Logit Fit'},
                    scatter_kws={'alpha': 0.2, 'color': 'purple', 'label': 'Raw Decisions (0/1)'})
        
        if np.isfinite(s_threshold_fit):
            ax2.axvline(s_threshold_fit, color='r', linestyle='--', label=f'50% Threshold S = {s_threshold_fit:.2f}')
    
    ax2.set_title('P(Accept) vs. Suitor\'s Total Score (S)')
    ax2.set_xlabel('Suitor\'s Total Score (S)')
    ax2.set_ylabel('Probability of Acceptance')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = f"fit_scalar_logit_results_{model_key}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plots saved to {output_filename}")
    plt.show()

# --- 主程序 ---
if __name__ == '__main__':
    # 在这里切换要分析的模型键
    model_key_to_analyze = 'gpt4_en_fitting' 
    
    print(f"--- Starting Scalar Logistic Regression Fitting for: {model_key_to_analyze} ---")
    fit_and_plot_logit(model_key_to_analyze)