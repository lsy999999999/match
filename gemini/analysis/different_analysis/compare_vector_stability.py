# 文件名: compare_vector_stability.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from config import config
from load_data import load_all_group_data_for_model, load_matchings_from_json, load_source_scores_vectorized
from gale_shapley_classic import classic_gale_shapley_matcher

# --- 1. 核心函数 ---

def get_vector_logit_params(model_key):
    """加载数据并为两个场景分别拟合六维逻辑回归模型，返回两套参数。"""
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None

    source_scores_vec, score_cols = load_source_scores_vectorized(config["source_data_path"])
    if source_scores_vec is None: return None

    params = {'taken': None, 'single': None}
    
    df_taken, df_single, _ = prepare_vector_data(df_log, source_scores_vec, score_cols)
    
    if df_taken is not None and not df_taken.empty:
        diff_cols = [f'diff_{col}' for col in score_cols]
        X = sm.add_constant(df_taken[diff_cols]); y = df_taken['result']
        try:
            result = sm.Logit(y, X.astype(float)).fit(disp=0)
            params['taken'] = {'beta0': result.params['const'], 'lambda': result.params.drop('const')}
            print(f"--- 'Taken' Vector Params for '{config[model_key]['label']}': Fitted successfully.")
        except Exception as e:
             print(f"Warning: Failed to fit 'Taken' vector model for {model_key}: {e}")

    if df_single is not None and not df_single.empty:
        X = sm.add_constant(df_single[score_cols]); y = df_single['result']
        try:
            result = sm.Logit(y, X.astype(float)).fit(disp=0)
            params['single'] = {'beta0': result.params['const'], 'lambda': result.params.drop('const')}
            print(f"--- 'Single' Vector Params for '{config[model_key]['label']}': Fitted successfully.")
        except Exception as e:
            print(f"Warning: Failed to fit 'Single' vector model for {model_key}: {e}")
            
    return params

def prepare_vector_data(df_log, source_scores_vec, score_cols):
    """辅助函数，准备向量化数据"""
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_score_vectors(row):
        try:
            d, ns, cp = int(row['target']), int(row['proposer']), int(row['current_partner'])
            Xa = np.array(source_scores_vec.get((d, ns), [np.nan]*6)); Xb = np.array(source_scores_vec.get((d, cp), [np.nan]*6))
            return (Xa - Xb).tolist()
        except: return [np.nan]*6
    diff_cols = [f'diff_{col}' for col in score_cols]
    df_taken[diff_cols] = df_taken.apply(get_score_vectors, axis=1, result_type='expand')
    df_taken.dropna(subset=diff_cols, inplace=True)
    
    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_vector(row):
        try:
            d, p = int(row['target']), int(row['proposer'])
            return source_scores_vec.get((d, p), [np.nan]*6)
        except: return [np.nan]*6
    df_single[score_cols] = df_single.apply(get_single_vector, axis=1, result_type='expand')
    df_single.dropna(subset=score_cols, inplace=True)
    return df_taken, df_single, score_cols

def predict_prob_vector_sigmoid(X_vector, beta0, lambda_vector):
    """通用的多维Sigmoid预测函数"""
    if X_vector is None or lambda_vector is None or beta0 is None: return 0
    z = beta0 + np.dot(lambda_vector, X_vector)
    return 1.0 / (1.0 + np.exp(-z))

# 【关键修正】在这里添加缺失的函数定义
def get_men_women_ids_from_group_df(group_df):
    """
    从源数据DataFrame中获取男性和女性的ID列表。
    """
    if 'gender' not in group_df.columns or 'iid' not in group_df.columns:
        return [], []
    # 假设 gender=1 是男性, gender=0 是女性
    men_ids = sorted(group_df[group_df['gender'] == 1]['iid'].unique())
    women_ids = sorted(group_df[group_df['gender'] == 0]['iid'].unique())
    return men_ids, women_ids

def calculate_expected_blocking_pairs_vector(matchings, source_scores_vec, params, df_source, model_label_for_debug=""):
    """使用六维向量模型计算E[numbp]"""
    if not params or params.get('taken') is None or params.get('single') is None:
        print(f"[{model_label_for_debug}] 警告: 缺少拟合参数，无法计算稳定性。")
        return []
        
    params_taken = params['taken']
    params_single = params['single']
    expected_bp_counts = []
    
    for group_idx, matching in enumerate(matchings):
        if not matching: continue
        current_group_id = group_idx + 1 # 假设组号从1开始
        group_df = df_source[df_source['group'] == current_group_id]
        if group_df.empty: continue
        men, women = get_men_women_ids_from_group_df(group_df)
        if not men or not women: continue

        total_prob_sum = 0
        for m in men:
            for w in women:
                m_partner, w_partner = matching.get(m), matching.get(w)
                if m_partner == w: continue

                # prob1: P(m prefers w > m_p)
                if m_partner is None or isinstance(m_partner, str):
                    X_w_for_m = source_scores_vec.get((m, w))
                    prob1 = predict_prob_vector_sigmoid(X_w_for_m, params_single['beta0'], params_single['lambda'])
                else:
                    Xa = source_scores_vec.get((m, w)); Xb = source_scores_vec.get((m, m_partner))
                    if Xa is None or Xb is None: continue
                    X_diff = np.array(Xa) - np.array(Xb)
                    prob1 = predict_prob_vector_sigmoid(X_diff, params_taken['beta0'], params_taken['lambda'])
                
                # prob2: P(w prefers m > w_p)
                if w_partner is None or isinstance(w_partner, str):
                    X_m_for_w = source_scores_vec.get((w, m))
                    prob2 = predict_prob_vector_sigmoid(X_m_for_w, params_single['beta0'], params_single['lambda'])
                else:
                    Xa = source_scores_vec.get((w, m)); Xb = source_scores_vec.get((w, w_partner))
                    if Xa is None or Xb is None: continue
                    X_diff = np.array(Xa) - np.array(Xb)
                    prob2 = predict_prob_vector_sigmoid(X_diff, params_taken['beta0'], params_taken['lambda'])

                total_prob_sum += prob1 * prob2
        
        expected_bp_counts.append(total_prob_sum)
    return expected_bp_counts

# --- 主程序 ---
if __name__ == '__main__':
    print("--- Vector Stability Comparison (End-to-End) ---")
    
    # 1. 拟合两套六维模型参数
    print("\n[Step 1] Fitting vector decision models...")
    params_en = get_vector_logit_params('gpt4_en_fitting')
    params_zh = get_vector_logit_params('gpt4_zh_fitting')

    # 2. 加载数据
    print("\n[Step 2] Loading all necessary data...")
    df_source = pd.read_excel(config["source_data_path"])
    source_scores_vec, _ = load_source_scores_vectorized(config["source_data_path"])
    gpt4_en_matchings_run = load_matchings_from_json('gpt4_en_fitting')
    gpt4_zh_matchings_run = load_matchings_from_json('gpt4_zh_fitting')
    
    # 3. 为经典Gale-Shapley算法生成匹配结果
    print("\n[Step 3] Generating Classic GS matchings (based on GPT score)...")
    classic_gs_matchings = []
    num_groups = config['gpt4_en_fitting']['num_groups']
    for i in range(1, num_groups + 1):
        group_df = df_source[df_source['group'] == i]
        if group_df.empty: continue
        men_ids, women_ids = get_men_women_ids_from_group_df(group_df)
        if not men_ids or not women_ids: continue
        men_prefs = {m: group_df[group_df['iid'] == m].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs = {w: group_df[group_df['pid'] == w].sort_values(by='gpt_score', ascending=False)['iid'].tolist() for w in women_ids}
        classic_gs_matchings.append(classic_gale_shapley_matcher(men_prefs, women_prefs))

    # 4. 计算稳定性
    results = {}
    print("\n[Step 4] Calculating E[numbp] using vector models...")
    
    results['GPT-4 (Eng) Run'] = calculate_expected_blocking_pairs_vector(gpt4_en_matchings_run, source_scores_vec, params_en, df_source, "GPT-4 (Eng) Run")
    results['GPT-4 (Chinese) Run'] = calculate_expected_blocking_pairs_vector(gpt4_zh_matchings_run, source_scores_vec, params_zh, df_source, "GPT-4 (Chi) Run")
    results['Classic GS (judged by Eng AI)'] = calculate_expected_blocking_pairs_vector(classic_gs_matchings, source_scores_vec, params_en, df_source, "Classic GS (Eng Eval)")
    results['Classic GS (judged by Chi AI)'] = calculate_expected_blocking_pairs_vector(classic_gs_matchings, source_scores_vec, params_zh, df_source, "Classic GS (Chi Eval)")
    
    # 5. 打印和可视化
    print("\n--- Final Stability Results: Expected Number of Blocking Pairs (E[numbp]) ---")
    for strategy, values in results.items():
        if values:
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"Strategy: {strategy:<40} | Mean E[numbp]: {mean_val:>7.4f} | Std Dev: {std_val:>7.4f} | N_groups: {len(values)}")
    
    plot_data = []
    for strategy, values in results.items():
        if values:
            for v in values:
                plot_data.append({'Strategy': strategy, 'E[numbp]': v})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(16, 10))
    sns.boxplot(y='Strategy', x='E[numbp]', data=df_plot, orient='h')
    plt.title('Stability Comparison of Matching Strategies (Vector Model)', fontsize=18)
    plt.xlabel('Expected # of Blocking Pairs (Lower is More Stable)')
    plt.ylabel('Strategy & Evaluation Model')
    plt.tight_layout()
    output_filename = "stability_comparison_vector.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nFinal comparison plot saved to {output_filename}")
    plt.show()