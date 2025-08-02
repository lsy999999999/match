import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from config import config
from load_data import load_all_group_data_for_model, load_matchings_from_json, load_source_scores
from gale_shapley_classic import classic_gale_shapley_matcher

# --- 1. 核心函数 ---

def get_logit_params(model_key):
    # (此函数不变，它只关心分数差和结果，与性别区分无关)
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
    if df_taken.empty: return None, None
    X = sm.add_constant(df_taken['score_diff']); y = df_taken['result']
    try:
        result = sm.Logit(y, X.astype(float)).fit(disp=0)
        print(f"--- Model Parameters Fitted for '{config[model_key]['label']}' ---")
        print(f"  Fitted beta0: {result.params['const']:.4f}, Fitted lambda: {result.params['score_diff']:.4f}")
        return result.params['const'], result.params['score_diff']
    except: return None, None

def predict_prob_switch_sigmoid(Sa, Sb, beta0, lambda_param):
    # (此函数不变)
    if Sa is None or Sb is None: return 0
    return 1.0 / (1.0 + np.exp(-(beta0 + lambda_param * (Sa - Sb))))

def get_men_women_ids_from_group_df(group_df):
    """
    【新辅助函数】从源数据DataFrame中获取男性和女性的ID列表。
    """
    # 确保'gender'和'iid'列存在
    if 'gender' not in group_df.columns or 'iid' not in group_df.columns:
        return [], []
    # 假设 gender=1 是男性, gender=0 是女性
    men_ids = sorted(group_df[group_df['gender'] == 1]['iid'].unique())
    women_ids = sorted(group_df[group_df['gender'] == 0]['iid'].unique())
    return men_ids, women_ids

def calculate_expected_blocking_pairs(matchings, source_scores, beta0, lambda_param, df_source, model_label_for_debug=""):
    """
    核心计算函数，使用新的性别区分逻辑。
    """
    if beta0 is None or lambda_param is None: return []
    
    expected_bp_counts = []
    
    for group_idx, matching_unclean in enumerate(matchings):
        if not matching_unclean: continue
        
        # 清理从JSON加载的ID类型
        matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in matching_unclean.items()}
        
        # 【关键修正】从源数据中为当前组动态获取男性和女性ID
        # 我们假设matchings列表的顺序与1到N的组号对应
        current_group_id = group_idx + 1 # 假设第一组是group 1
        group_df = df_source[df_source['group'] == current_group_id]
        if group_df.empty:
            print(f"  警告: 在源数据中找不到组 {current_group_id} 的信息，无法区分性别。跳过该组稳定性计算。")
            continue
            
        men, women = get_men_women_ids_from_group_df(group_df)
        
        if not men or not women:
            print(f"  警告: 组 {current_group_id} 中缺少男性或女性ID。跳过。")
            continue

        total_prob_sum = 0
        non_zero_prob_count = 0

        for m in men:
            for w in women:
                m_partner = matching.get(m)
                if m_partner == w: continue
                w_partner = matching.get(w)
                
                # prob1: P(m prefers w > m_p)
                prob1 = 1.0
                if m_partner is not None and not isinstance(m_partner, str):
                    score_m_w, score_m_mp = source_scores.get((m, w)), source_scores.get((m, m_partner))
                    if score_m_w is None or score_m_mp is None: continue
                    prob1 = predict_prob_switch_sigmoid(score_m_w, score_m_mp, beta0, lambda_param)
                
                # prob2: P(w prefers m > w_p)
                prob2 = 1.0
                if w_partner is not None and not isinstance(w_partner, str):
                    score_w_m, score_w_wp = source_scores.get((w, m)), source_scores.get((w, w_partner))
                    if score_w_m is None or score_w_wp is None: continue
                    prob2 = predict_prob_switch_sigmoid(score_w_m, score_w_wp, beta0, lambda_param)

                p_bp = prob1 * prob2
                if p_bp > 1e-9: non_zero_prob_count += 1
                total_prob_sum += p_bp
        
        print(f"  Group {current_group_id:02d} ({model_label_for_debug:^30}): E[numbp] = {total_prob_sum:7.4f}, Found {non_zero_prob_count:4d} potential pairs with P_bp > 0.")
        expected_bp_counts.append(total_prob_sum)
        
    return expected_bp_counts


# --- 主程序 ---
if __name__ == '__main__':
    print("--- Stability Comparison using Logistic Regression Parameters (End-to-End) ---")
    
    # 1. 动态拟合模型参数
    params = {}
    print("\n[Step 1] Fitting decision model for deepseek (Eng)...")
    beta0_en, lambda_en = get_logit_params('deepseek_en_fitting')
    params['eng'] = {'beta0': beta0_en, 'lambda': lambda_en}
    
    print("\n[Step 2] Fitting decision model for deepseek (Chinese)...")
    beta0_zh, lambda_zh = get_logit_params('deepseek_zh_fitting')
    params['zh'] = {'beta0': beta0_zh, 'lambda': lambda_zh}

    # 2. 加载数据
    print("\n[Step 3] Loading all necessary data...")
    df_source = pd.read_excel(config["source_data_path"])
    if df_source is None: exit("Fatal: Could not load source Excel data.")
        
    source_scores_total = load_source_scores(config["source_data_path"])
    if source_scores_total is None: exit("Fatal: Could not process source scores.")

    deepseek_en_matchings_run = load_matchings_from_json('deepseek_en_fitting')
    deepseek_zh_matchings_run = load_matchings_from_json('deepseek_zh_fitting')
    
    # 3. 为经典Gale-Shapley算法生成匹配结果
    print("\n[Step 4] Generating matchings for Classic Gale-Shapley (based on 'objective' and 'gpt_score')...")
    objective_matchings = []
    deepseek_guided_matchings = []
    
    # 使用你之前截图确认的列名来计算客观加权分数
    dims = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
    score_cols = [f'{d}_partner' for d in dims]
    importance_cols = [f'{d}_important' for d in dims]
    weighted_score = np.zeros(len(df_source))
    for i in range(len(dims)):
        weighted_score += df_source[score_cols[i]].fillna(0) * df_source[importance_cols[i]].fillna(0)
    df_source['weighted_score'] = weighted_score / 100

    num_groups = config['deepseek_en_fitting']['num_groups']
    
    for i in range(1, num_groups + 1):
        group_df = df_source[df_source['group'] == i]
        if group_df.empty: continue
        
        men_ids, women_ids = get_men_women_ids_from_group_df(group_df)
        if not men_ids or not women_ids: continue

        # a. 客观匹配
        men_prefs_obj = {m: group_df[group_df['iid'] == m].sort_values(by='weighted_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs_obj = {w: group_df[group_df['pid'] == w].sort_values(by='weighted_score', ascending=False)['iid'].tolist() for w in women_ids}
        objective_matchings.append(classic_gale_shapley_matcher(men_prefs_obj, women_prefs_obj))

        # b. GPT指导的匹配
        men_prefs_deepseek = {m: group_df[group_df['iid'] == m].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs_deepseek = {w: group_df[group_df['pid'] == w].sort_values(by='gpt_score', ascending=False)['iid'].tolist() for w in women_ids}
        deepseek_guided_matchings.append(classic_gale_shapley_matcher(men_prefs_deepseek, women_prefs_deepseek))

    # 4. 计算所有策略下的E[numbp]
    results = {}
    print("\n[Step 5] Calculating Expected Number of Blocking Pairs for each strategy...")
    
    # 评估由AI实际运行得到的匹配
    if params['eng']['beta0'] is not None:
        results['deepseek (Eng) Actual Run'] = calculate_expected_blocking_pairs(deepseek_en_matchings_run, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], df_source, "deepseek (Eng) Run")
    if params['zh']['beta0'] is not None:
        results['deepseek (Chinese) Actual Run'] = calculate_expected_blocking_pairs(deepseek_zh_matchings_run, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], df_source, "deepseek (Chi) Run")
    
    # 评估基于客观分数的经典GS匹配
    if params['eng']['beta0'] is not None:
        results['Objective GS (judged by Eng AI)'] = calculate_expected_blocking_pairs(objective_matchings, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], df_source, "Objective (Eng Eval)")
    if params['zh']['beta0'] is not None:
        results['Objective GS (judged by Chi AI)'] = calculate_expected_blocking_pairs(objective_matchings, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], df_source, "Objective (Chi Eval)")

    # 评估基于GPT分数的经典GS匹配
    if params['eng']['beta0'] is not None:
        results['gpt 4-score GS (judged by Eng AI)'] = calculate_expected_blocking_pairs(deepseek_guided_matchings, source_scores_total, params['eng']['beta0'], params['eng']['lambda'], df_source, "deepseek-guided (Eng Eval)")
    if params['zh']['beta0'] is not None:
        results['gpt 4-score GS (judged by Chi AI)'] = calculate_expected_blocking_pairs(deepseek_guided_matchings, source_scores_total, params['zh']['beta0'], params['zh']['lambda'], df_source, "deepseek-guided (Chi Eval)")

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
    plt.title('Stability Comparison of Different Matching Strategies', fontsize=18)
    plt.xlabel('Expected # of Blocking Pairs (Lower is More Stable)')
    plt.ylabel('Strategy & Evaluation Model')
    plt.tight_layout()
    output_filename = "stability_comparison_final.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nFinal comparison plot saved to {output_filename}")
    plt.show()