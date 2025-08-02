import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # 使用statsmodels进行逻辑回归

from config import config
from load_data import load_all_group_data_for_model, load_source_scores_vectorized

# --- 1. 数据准备 (与之前的向量化脚本相同) ---
def prepare_vector_data(model_key):
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None, None

    source_scores_vec, score_cols = load_source_scores_vectorized(config["source_data_path"])
    if source_scores_vec is None: return None, None, None
    
    # a. "挖墙脚"场景
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_score_vectors(row):
        try:
            decision_maker, new_suitor, current_partner = int(row['target']), int(row['proposer']), int(row['current_partner'])
            Xa = np.array(source_scores_vec.get((decision_maker, new_suitor), [np.nan]*6))
            Xb = np.array(source_scores_vec.get((decision_maker, current_partner), [np.nan]*6))
            return (Xa - Xb).tolist()
        except (ValueError, TypeError): return [np.nan]*6
    diff_cols = [f'diff_{col}' for col in score_cols]
    df_taken[diff_cols] = df_taken.apply(get_score_vectors, axis=1, result_type='expand')
    df_taken.dropna(subset=diff_cols, inplace=True)
    
    # b. "向单身求偶"场景
    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_vector(row):
        try:
            decision_maker, proposer = int(row['target']), int(row['proposer'])
            return source_scores_vec.get((decision_maker, proposer), [np.nan]*6)
        except (ValueError, TypeError): return [np.nan]*6
    df_single[score_cols] = df_single.apply(get_single_vector, axis=1, result_type='expand')
    df_single.dropna(subset=score_cols, inplace=True)

    return df_taken, df_single, score_cols


# --- 2. 模型拟合与参数求解 (使用statsmodels) ---
def fit_vector_lambda(df_taken, diff_cols):
    """使用多元逻辑回归拟合lambda向量"""
    if df_taken.empty or len(df_taken) < (len(diff_cols) + 1):
        print("Not enough 'Proposing to Taken' data to fit vector lambda.")
        return None

    X = df_taken[diff_cols]
    y = df_taken['result']
    
    # statsmodels需要手动添加截距项 (const)
    X = sm.add_constant(X) 
    
    # 创建并拟合模型
    logit_model = sm.Logit(y, X.astype(float))
    result = logit_model.fit(disp=0) # disp=0禁止打印收敛信息
    
    print("\n--- Vector λ Fitting Results (Multivariate Logistic Regression) ---")
    print(result.summary())
    
    # 提取系数向量(λ)，排除截距项'const'
    lambda_vector = result.params.drop('const')
    
    return lambda_vector

def fit_vector_x_reserve(df_single, score_cols):
    """使用多元逻辑回归求解X_reserve向量"""
    if df_single.empty or len(df_single) < (len(score_cols) + 1):
        print("Not enough 'Proposing to Single' data to fit vector X_reserve.")
        return None
    
    X = df_single[score_cols]
    y = df_single['result']

    X = sm.add_constant(X)
    logit_model = sm.Logit(y, X.astype(float))
    result = logit_model.fit(disp=0)

    print("\n--- Vector X_reserve Fitting Results ---")
    print(result.summary())
    
    b0 = result.params['const']
    lambda_from_single = result.params.drop('const')
    
    # 计算X_reserve
    if (lambda_from_single**2).sum() == 0:
        print("Warning: Lambda vector from single's data is zero, cannot calculate X_reserve.")
        return None
    
    x_reserve_vector = -b0 * lambda_from_single / (lambda_from_single**2).sum()

    return x_reserve_vector


# --- 3. 可视化 (与之前的向量化脚本完全相同) ---
def plot_vector_results(lambda_vec, x_reserve_vec, model_key):
    if lambda_vec is None and x_reserve_vec is None:
        print("Cannot plot results due to fitting errors for both scenarios.")
        return

    model_label = config[model_key].get('label', model_key)
    
    plot_data = {}
    if lambda_vec is not None:
        lambda_vec.index = lambda_vec.index.str.replace('diff_', '')
        plot_data['λ (Dimension Weights)'] = lambda_vec
    
    if x_reserve_vec is not None:
        plot_data['X_reserve (Acceptance Thresholds)'] = x_reserve_vec
        
    df_plot = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    fig.suptitle(f'Fitted Decision Parameters for {model_label}', fontsize=18)

    # 图1： Lambda 向量可视化
    if 'λ (Dimension Weights)' in df_plot.columns:
        sns.barplot(x=df_plot.index, y='λ (Dimension Weights)', data=df_plot, 
                    ax=axes[0], palette='viridis', hue=df_plot.index, legend=False)
        axes[0].set_title('λ Vector: Importance of Each Dimension in "Taken" Scenario')
        axes[0].set_ylabel('Weight (Log-Odds Ratio)')
        axes[0].set_xlabel('Dimension')
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        axes[0].text(0.5, 0.5, 'Lambda fitting was not successful.', ha='center', va='center', fontsize=12, color='red')
        axes[0].set_title('λ Vector: Importance of Each Dimension in "Taken" Scenario')

    # 图2： X_reserve 向量可视化
    if 'X_reserve (Acceptance Thresholds)' in df_plot.columns:
        sns.barplot(x=df_plot.index, y='X_reserve (Acceptance Thresholds)', data=df_plot, 
                    ax=axes[1], palette='plasma', hue=df_plot.index, legend=False)
        axes[1].set_title('X_reserve Vector: Threshold Score for Each Dimension in "Single" Scenario')
        axes[1].set_ylabel('Score Threshold')
        axes[1].set_xlabel('Dimension')
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        axes[1].text(0.5, 0.5, 'X_reserve fitting was not successful.', ha='center', va='center', fontsize=12, color='red')
        axes[1].set_title('X_reserve Vector: Threshold Score for Each Dimension in "Single" Scenario')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = f"fit_vector_sigmoid_results_{model_key}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plots saved to {output_filename}")
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':
    # 在这里切换要分析的模型键, 例如 'gpt4_en_fitting', 'gpt4_zh_fitting', 'deepseek_en_fitting', etc.
    model_key_to_analyze = 'gemini_zh_fitting' 
    
    print(f"--- Starting Vector Sigmoid Fitting for: {model_key_to_analyze} ---")
    df_taken, df_single, score_cols = prepare_vector_data(model_key_to_analyze)
    
    lambda_vector = None
    if df_taken is not None:
        lambda_vector = fit_vector_lambda(df_taken, [f'diff_{col}' for col in score_cols])
    
    x_reserve_vector = None
    if df_single is not None:
        x_reserve_vector = fit_vector_x_reserve(df_single, score_cols)
    
    plot_vector_results(lambda_vector, x_reserve_vector, model_key_to_analyze)
