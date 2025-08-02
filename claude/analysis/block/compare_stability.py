import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 导入你项目中的模块
from config import config
from load_data import load_all_group_data_for_model, load_matchings_from_json, load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# --- 1. 核心函数 ---

def get_unified_model_params(model_key):
    """
    通过两次独立的Logit拟合，计算出统一模型的所有参数 (β₀_taken, λ_taken, S₀)。
    """
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None, None
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None: return None, None, None

    beta0_taken, lambda_taken, s0_calculated = None, None, None
    
    # --- a. 从 'Taken' 场景学习通用参数 ---
    df_taken = df_log[df_log['current_partner'].notna()].copy()
    def get_scores(row):
        try:
            d, ns, cp = int(row['target']), int(row['proposer']), int(row['current_partner'])
            return source_scores.get((d, ns)), source_scores.get((d, cp))
        except: return None, None
    df_taken[['Sa', 'Sb']] = df_taken.apply(get_scores, axis=1, result_type='expand')
    df_taken.dropna(subset=['Sa', 'Sb'], inplace=True)
    df_taken['score_diff'] = df_taken['Sa'] - df_taken['Sb']
    
    if not df_taken.empty:
        X = sm.add_constant(df_taken['score_diff'])
        y = df_taken['result']
        try:
            result_taken = sm.Logit(y, X.astype(float)).fit(disp=0)
            beta0_taken = result_taken.params['const']
            lambda_taken = result_taken.params['score_diff']
            print(f"--- Unified Params (from 'Taken' model) for '{config[model_key]['label']}':")
            print(f"    β₀_taken = {beta0_taken:.4f}, λ_taken = {lambda_taken:.4f}")
        except Exception as e:
             print(f"Warning: Failed to fit 'Taken' model for {model_key}: {e}")

    # --- b. 从 'Single' 场景学习其截距 ---
    df_single = df_log[df_log['current_partner'].isna()].copy()
    def get_single_score(row):
        try:
            d, p = int(row['target']), int(row['proposer'])
            return source_scores.get((d, p))
        except: return None
    df_single['S'] = df_single.apply(get_single_score, axis=1)
    df_single.dropna(subset=['S'], inplace=True)
    
    if not df_single.empty:
        X = sm.add_constant(df_single['S'])
        y = df_single['result']
        try:
            result_single = sm.Logit(y, X.astype(float)).fit(disp=0)
            beta0_single = result_single.params['const']
            print(f"--- Intercept (from 'Single' model) for '{config[model_key]['label']}':")
            print(f"    β₀_single = {beta0_single:.4f}")
            
            # --- c. 反向求解 S₀ ---
            if beta0_taken is not None and lambda_taken is not None and lambda_taken != 0:
                s0_calculated = (beta0_taken - beta0_single) / lambda_taken
                print(f"--- Calculated S₀ for '{config[model_key]['label']}': {s0_calculated:.2f} ---")

        except Exception as e:
            print(f"Warning: Failed to fit 'Single' model for {model_key}: {e}")
            
    return beta0_taken, lambda_taken, s0_calculated

def predict_prob_unified(score_new, score_current, beta0, lambda_param):
    """通用的Sigmoid预测函数，用于统一模型"""
    if score_new is None or score_current is None or beta0 is None or lambda_param is None:
        return 0
    score_diff = score_new - score_current
    z = beta0 + lambda_param * score_diff
    return 1.0 / (1.0 + np.exp(-z))

def get_men_women_ids_from_group_df(group_df):
    """从源数据DataFrame中获取男性和女性的ID列表。"""
    if 'gender' not in group_df.columns or 'iid' not in group_df.columns: return [], []
    men_ids = sorted(group_df[group_df['gender'] == 1]['iid'].unique())
    women_ids = sorted(group_df[group_df['gender'] == 0]['iid'].unique())
    return men_ids, women_ids

def calculate_expected_blocking_pairs_unified(matchings, source_scores, beta0, lambda_param, s0_param, df_source, model_label_for_debug=""):
    """【新】使用统一模型的参数计算E[numbp]"""
    if beta0 is None or lambda_param is None or s0_param is None:
        print(f"[{model_label_for_debug}] 警告: 缺少统一模型参数，无法计算稳定性。")
        return []
        
    expected_bp_counts = []
    for group_idx, matching_unclean in enumerate(matchings):
        if not matching_unclean: continue
        matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in matching_unclean.items()}
        current_group_id = group_idx + 1
        group_df = df_source[df_source['group'] == current_group_id]
        if group_df.empty: continue
        men, women = get_men_women_ids_from_group_df(group_df)
        if not men or not women: continue

        total_prob_sum = 0
        for m in men:
            for w in women:
                m_partner, w_partner = matching.get(m), matching.get(w)
                if m_partner == w: continue

                # prob1: P(m switches)
                score_m_w = source_scores.get((m, w))
                if m_partner is None or isinstance(m_partner, str):
                    prob1 = predict_prob_unified(score_m_w, s0_param, beta0, lambda_param)
                else:
                    score_m_mp = source_scores.get((m, m_partner))
                    prob1 = predict_prob_unified(score_m_w, score_m_mp, beta0, lambda_param)
                
                # prob2: P(w switches)
                score_w_m = source_scores.get((w, m))
                if w_partner is None or isinstance(w_partner, str):
                    prob2 = predict_prob_unified(score_w_m, s0_param, beta0, lambda_param)
                else:
                    score_w_wp = source_scores.get((w, w_partner))
                    prob2 = predict_prob_unified(score_w_m, score_w_wp, beta0, lambda_param)

                total_prob_sum += prob1 * prob2
        
        expected_bp_counts.append(total_prob_sum)
    return expected_bp_counts

# --- 主程序 ---
if __name__ == '__main__':
    print("--- Unified Model Stability Comparison (Logit-only) ---")
    
    # 1. 拟合统一模型参数
    print("\n[Step 1] Fitting unified decision models...")
    beta0_en, lambda_en, s0_en = get_unified_model_params('claude_en_fitting')
    beta0_zh, lambda_zh, s0_zh = get_unified_model_params('claude_zh_fitting')

    params = {
        'eng': {'beta0': beta0_en, 'lambda': lambda_en, 's0': s0_en},
        'zh': {'beta0': beta0_zh, 'lambda': lambda_zh, 's0': s0_zh}
    }

    # 2. 加载数据
    print("\n[Step 2] Loading all necessary data...")
    df_source = pd.read_excel(config["source_data_path"])
    if df_source is None: exit("Fatal: Could not load source Excel data.")
        
    source_scores_total = load_source_scores(config["source_data_path"])
    if source_scores_total is None: exit("Fatal: Could not process source scores.")

    claude_en_matchings_run = load_matchings_from_json('claude_en_fitting')
    claude_zh_matchings_run = load_matchings_from_json('claude_zh_fitting')
    
    # 3. 为经典Gale-Shapley算法生成匹配结果
    print("\n[Step 3] Generating matchings for Classic Gale-Shapley (based on 'objective' and 'gpt_score')...")
    objective_matchings = []
    gpt_guided_matchings = []
    
    dims = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
    score_cols = [f'{d}_partner' for d in dims]
    importance_cols = [f'{d}_important' for d in dims]
    weighted_score = np.zeros(len(df_source))
    for i in range(len(dims)):
        weighted_score += df_source[score_cols[i]].fillna(0) * df_source[importance_cols[i]].fillna(0)
    df_source['weighted_score'] = weighted_score / 100

    num_groups = config['claude_en_fitting']['num_groups']
    
    for i in range(1, num_groups + 1):
        group_df = df_source[df_source['group'] == i]
        if group_df.empty: continue
        men_ids, women_ids = get_men_women_ids_from_group_df(group_df)
        if not men_ids or not women_ids: continue

        men_prefs_obj = {m: group_df[group_df['iid'] == m].sort_values(by='weighted_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs_obj = {w: group_df[group_df['pid'] == w].sort_values(by='weighted_score', ascending=False)['iid'].tolist() for w in women_ids}
        objective_matchings.append(classic_gale_shapley_matcher(men_prefs_obj, women_prefs_obj))

        men_prefs_claude = {m: group_df[group_df['iid'] == m].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs_claude = {w: group_df[group_df['pid'] == w].sort_values(by='gpt_score', ascending=False)['iid'].tolist() for w in women_ids}
        gpt_guided_matchings.append(classic_gale_shapley_matcher(men_prefs_claude, women_prefs_claude))

    # 4. 计算稳定性
    results = {}
    print("\n[Step 4] Calculating E[numbp] using Unified Models...")
    
    # 评估由AI实际运行得到的匹配 (AI-driven Random Proposer)
    if all(p is not None for p in params['eng'].values()):
        results['AI Run (judged by Eng AI)'] = calculate_expected_blocking_pairs_unified(claude_en_matchings_run, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], params['eng']['s0'], df_source, "AI Run (Eng Eval)")
    if all(p is not None for p in params['zh'].values()):
        results['AI Run (judged by Chi AI)'] = calculate_expected_blocking_pairs_unified(claude_zh_matchings_run, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], params['zh']['s0'], df_source, "AI Run (Chi Eval)")
    
    # 评估基于客观分数的经典GS匹配 (Objective GS)
    if all(p is not None for p in params['eng'].values()):
        results['Objective GS (judged by Eng AI)'] = calculate_expected_blocking_pairs_unified(objective_matchings, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], params['eng']['s0'], df_source, "Objective GS (Eng Eval)")
    if all(p is not None for p in params['zh'].values()):
        results['Objective GS (judged by Chi AI)'] = calculate_expected_blocking_pairs_unified(objective_matchings, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], params['zh']['s0'], df_source, "Objective GS (Chi Eval)")
        
    # 评估基于GPT分数的经典GS匹配 (GPT-score GS)
    if all(p is not None for p in params['eng'].values()):
        results['GPT-score GS (judged by Eng AI)'] = calculate_expected_blocking_pairs_unified(gpt_guided_matchings, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], params['eng']['s0'], df_source, "GPT-score GS (Eng Eval)")
    if all(p is not None for p in params['zh'].values()):
        results['GPT-score GS (judged by Chi AI)'] = calculate_expected_blocking_pairs_unified(gpt_guided_matchings, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], params['zh']['s0'], df_source, "GPT-guided (Chi Eval)")


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

    if not df_plot.empty:
        plt.figure(figsize=(16, 10))
        sns.boxplot(y='Strategy', x='E[numbp]', data=df_plot, orient='h')
        plt.title('Stability Comparison of Different Matching Strategies', fontsize=18)
        plt.xlabel('Expected # of Blocking Pairs (Lower is More Stable)')
        plt.ylabel('Strategy & Evaluation Model')
        plt.tight_layout()
        output_filename = "stability_comparison_unified_model.png"
        plt.savefig(output_filename, dpi=300)
        print(f"\nFinal comparison plot saved to {output_filename}")
        plt.show()
    else:
        print("\nNo stability results to plot.")