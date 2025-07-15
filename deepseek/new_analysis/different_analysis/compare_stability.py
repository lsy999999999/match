import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from load_data import load_all_group_data_for_model, load_source_scores, load_gender_map
from gale_shapley_classic import classic_gale_shapley_matcher


# --- 1. 定义和拟合决策模型的函数 ---

def sigmoid_model(x, beta0, lambda_param):
    return 1.0 / (1.0 + np.exp(-(beta0 + lambda_param * x)))

def predict_prob_switch_sigmoid(Sa, Sb, beta0, lambda_param):
    if any(v is None for v in [Sa, Sb, beta0, lambda_param]): return 0
    score_diff = Sa - Sb
    z = beta0 + lambda_param * score_diff
    return 1.0 / (1.0 + np.exp(-z))

# --- 2. 计算期望阻塞对数量的核心函数 ---

def calculate_expected_blocking_pairs(matchings, source_scores, gender_map, beta0, lambda_param, model_label=""):
    if beta0 is None or lambda_param is None: return []
    
    expected_bp_counts = []
    
    for group_idx, matching_str_keys in enumerate(matchings):
        if not matching_str_keys: continue
        
        try:
            matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in matching_str_keys.items()}
        except (ValueError, TypeError): continue

        all_participants = list(matching.keys())
        # 【关键修正】使用gender_map来正确区分男女
        men = {p for p in all_participants if gender_map.get(p) == 1}
        women = {p for p in all_participants if gender_map.get(p) == 0}

        if not men or not women:
            print(f"[{model_label}] 警告: Group {group_idx + 1} 缺少男性或女性参与者，跳过。")
            continue

        total_prob_sum = 0
        
        for m in men:
            for w in women:
                m_partner = matching.get(m)
                if m_partner == w: continue

                w_partner = matching.get(w)

                is_m_single = m_partner is None or isinstance(m_partner, str)
                is_w_single = w_partner is None or isinstance(w_partner, str)

                # prob1: P(m prefers w > m_p)
                if is_m_single: prob1 = 1.0
                else:
                    m_p = m_partner
                    score_m_w, score_m_mp = source_scores.get((m, w)), source_scores.get((m, m_p))
                    prob1 = predict_prob_switch_sigmoid(score_m_w, score_m_mp, beta0, lambda_param)
                
                # prob2: P(w prefers m > w_p)
                if is_w_single: prob2 = 1.0
                else:
                    w_p = w_partner
                    score_w_m, score_w_wp = source_scores.get((w, m)), source_scores.get((w, w_p))
                    prob2 = predict_prob_switch_sigmoid(score_w_m, score_w_wp, beta0, lambda_param)

                total_prob_sum += prob1 * prob2
        
        expected_bp_counts.append(total_prob_sum)
    
    return expected_bp_counts

# --- 主程序 ---
if __name__ == '__main__':
    print("--- Starting Stability Comparison Analysis (Sigmoid-based) ---")


    # 1. 使用你提供的硬编码参数
    params = {
        'eng': {'beta0': -12.7006, 'lambda': 0.6782},
        'zh':  {'beta0': 0.0429,   'lambda': 0.0229}
    }
    print("Using pre-fitted parameters for Eng and Chinese models.")

    # 2. 加载所有需要的数据
    source_scores = load_source_scores(config["source_data_path"])
    gender_map = load_gender_map(config["source_data_path"])
    _, deepseek_en_matchings = load_all_group_data_for_model('deepseek_en_fitting')
    _, deepseek_zh_matchings = load_all_group_data_for_model('deepseek_zh_fitting')

    if not source_scores or not gender_map:
        print("无法加载源数据或性别映射，分析中止。")
        exit()

    # 3. 为经典Gale-Shapley算法生成匹配结果
    print("\nGenerating matchings for Classic Gale-Shapley...")
    df_source = pd.read_excel(config["source_data_path"])
    classic_gs_matchings = []
    num_groups = config['deepseek_en_fitting']['num_groups'] 
    

    for i in range(1, num_groups + 1):
        group_df = df_source[df_source['group'] == i]
        if group_df.empty: continue
        
        men_ids = group_df[group_df['gender'] == 1]['iid'].unique()
        women_ids = group_df[group_df['gender'] == 0]['iid'].unique()
        
        men_prefs = {m: group_df[group_df['iid'] == m].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs = {w: group_df[group_df['iid'] == w].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for w in women_ids}
        
        classic_match = classic_gale_shapley_matcher(men_prefs, women_prefs)
        classic_gs_matchings.append(classic_match)

    # 4. 计算所有策略下的E[numbp]
    results = {}
    print("\n--- Calculating E[numbp] for each strategy ---")
    
    results['deepseek (Eng)'] = calculate_expected_blocking_pairs(deepseek_en_matchings, source_scores, gender_map, params['eng']['beta0'], params['eng']['lambda'], "deepseek (Eng)")
    results['deepseek (Chinese)'] = calculate_expected_blocking_pairs(deepseek_zh_matchings, source_scores, gender_map, params['zh']['beta0'], params['zh']['lambda'], "deepseek (Chinese)")
    results['Classic GS (judged by Eng AI)'] = calculate_expected_blocking_pairs(classic_gs_matchings, source_scores, gender_map, params['eng']['beta0'], params['eng']['lambda'], "Classic GS (Eng Eval)")
    results['Classic GS (judged by Chi AI)'] = calculate_expected_blocking_pairs(classic_gs_matchings, source_scores, gender_map, params['zh']['beta0'], params['zh']['lambda'], "Classic GS (Chi Eval)")
    
    # 5. 打印具体的稳定性指标
    print("\n--- Stability Analysis Results: Expected Number of Blocking Pairs (E[numbp]) ---")
    for strategy, values in results.items():
        if values:
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"Strategy: {strategy:<30} | Mean E[numbp]: {mean_val:>7.4f} | Std Dev: {std_val:>7.4f} | N_groups: {len(values)}")
        else:
            print(f"Strategy: {strategy:<30} | No data to calculate or calculation resulted in zero.")

    # 6. 可视化对比结果
    plot_data = [{'Strategy': s, 'E[numbp]': v} for s, vals in results.items() for v in vals]
    if not plot_data:
        print("\nNo data to plot. Exiting.")
        exit()
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Strategy', y='E[numbp]', data=df_plot)
    plt.title('Stability Comparison: Expected Number of Blocking Pairs', fontsize=18)
    plt.ylabel('Expected # of Blocking Pairs (Lower is More Stable)')
    plt.xlabel('Matching Strategy & Evaluation Model')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    output_filename = "stability_comparison_sigmoid.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nFinal comparison plot saved to {output_filename}")
    plt.show()

