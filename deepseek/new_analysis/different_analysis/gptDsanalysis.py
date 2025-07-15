import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from load_data import load_all_group_data_for_model, load_source_scores, load_gender_map
from gale_shapley_classic import classic_gale_shapley_matcher

# --- 函数定义部分 (保持不变) ---
def sigmoid_model(x, beta0, lambda_param):
    return 1.0 / (1.0 + np.exp(-(beta0 + lambda_param * x)))

def predict_prob_switch_sigmoid(Sa, Sb, beta0, lambda_param):
    if any(v is None for v in [Sa, Sb, beta0, lambda_param]): return 0
    score_diff = Sa - Sb
    z = beta0 + lambda_param * score_diff
    return 1.0 / (1.0 + np.exp(-z))

def calculate_expected_blocking_pairs(matchings, source_scores, gender_map, beta0, lambda_param, model_label=""):
    if beta0 is None or lambda_param is None: return []
    expected_bp_counts = []
    for group_idx, matching_str_keys in enumerate(matchings):
        if not matching_str_keys: continue
        try:
            matching = {int(k): (int(v) if str(v).isdigit() else v) for k, v in matching_str_keys.items()}
        except (ValueError, TypeError): continue
        all_participants = list(matching.keys())
        men = {p for p in all_participants if gender_map.get(p) == 1}
        women = {p for p in all_participants if gender_map.get(p) == 0}
        if not men or not women:
            # print(f"[{model_label}] 警告: Group {group_idx + 1} 缺少男性或女性参与者，跳过。")
            continue
        total_prob_sum = 0
        for m in men:
            for w in women:
                m_partner = matching.get(m)
                if m_partner == w: continue
                w_partner = matching.get(w)
                is_m_single = m_partner is None or isinstance(m_partner, str)
                is_w_single = w_partner is None or isinstance(w_partner, str)
                if is_m_single and is_w_single:
                    prob1, prob2 = 1.0, 1.0
                else:
                    if is_m_single: prob1 = 1.0
                    else:
                        m_p = m_partner
                        score_m_w, score_m_mp = source_scores.get((m, w)), source_scores.get((m, m_p))
                        prob1 = predict_prob_switch_sigmoid(score_m_w, score_m_mp, beta0, lambda_param)
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
    
    # 1. 使用你之前拟合出的硬编码参数
    # 【注意】这里应该是你为GPT-4和DeepSeek分别拟合出的参数
    params = {
        'gpt4_en': {'beta0': -12.7006, 'lambda': 0.6782},
        'gpt4_zh': {'beta0': 0.0429,   'lambda': 0.0229},
        'deepseek_en': {'beta0': -0.1822, 'lambda': 0.1476},
        'deepseek_zh': {'beta0': -0.9801, 'lambda': 0.1117}
    }
    print("Using pre-fitted parameters for all models.")

    # 2. 加载所有需要的数据
    source_scores = load_source_scores(config["source_data_path"])
    gender_map = load_gender_map(config["source_data_path"])
    # 加载所有模型的匹配结果
    _, gpt4_en_matchings = load_all_group_data_for_model('gpt4_en_fitting')
    _, gpt4_zh_matchings = load_all_group_data_for_model('gpt4_zh_fitting')
    _, deepseek_en_matchings = load_all_group_data_for_model('deepseek_en_fitting')
    _, deepseek_zh_matchings = load_all_group_data_for_model('deepseek_zh_fitting')

    if not source_scores or not gender_map:
        print("无法加载源数据或性别映射，分析中止。")
        exit()

    # 3. 为经典Gale-Shapley算法生成匹配结果
    print("\nGenerating matchings for Classic Gale-Shapley...")
    df_source = pd.read_excel(config["source_data_path"])
    classic_gs_matchings = []
    # 假设所有实验的组结构和参与者都相同，以任一配置的组数为准
    num_groups = config['gpt4_en_fitting']['num_groups'] 
    
    for i in range(1, num_groups + 1):
        group_df = df_source[df_source['group'] == i]
        if group_df.empty: continue
        
        men_ids = group_df[group_df['gender'] == 1]['iid'].unique()
        women_ids = group_df[group_df['gender'] == 0]['iid'].unique()
        
        # 【关键修正】使用 'gpt_score' 作为统一的偏好排序标准
        if 'gpt_score' not in group_df.columns:
            print(f"错误: 源数据Excel文件中缺少 'gpt_score' 列，无法为经典GS算法生成偏好列表。")
            break
        
        men_prefs = {m: group_df[group_df['iid'] == m].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for m in men_ids}
        women_prefs = {w: group_df[group_df['iid'] == w].sort_values(by='gpt_score', ascending=False)['pid'].tolist() for w in women_ids}
        
        classic_match = classic_gale_shapley_matcher(men_prefs, women_prefs)
        classic_gs_matchings.append(classic_match)

    # 4. 计算所有策略下的E[numbp]
    results = {}
    print("\n--- Calculating E[numbp] for each strategy ---")
    
    # 策略组1: GPT-4 英文
    results['GPT-4 (Eng)'] = calculate_expected_blocking_pairs(gpt4_en_matchings, source_scores, gender_map, params['gpt4_en']['beta0'], params['gpt4_en']['lambda'], "GPT-4 (Eng)")
    # 策略组2: GPT-4 中文
    results['GPT-4 (Chinese)'] = calculate_expected_blocking_pairs(gpt4_zh_matchings, source_scores, gender_map, params['gpt4_zh']['beta0'], params['gpt4_zh']['lambda'], "GPT-4 (Chinese)")
    # 策略组3: DeepSeek 英文
    results['DeepSeek (Eng)'] = calculate_expected_blocking_pairs(deepseek_en_matchings, source_scores, gender_map, params['deepseek_en']['beta0'], params['deepseek_en']['lambda'], "DeepSeek (Eng)")
    # 策略组4: DeepSeek 中文
    results['DeepSeek (Chinese)'] = calculate_expected_blocking_pairs(deepseek_zh_matchings, source_scores, gender_map, params['deepseek_zh']['beta0'], params['deepseek_zh']['lambda'], "DeepSeek (Chinese)")

    # 策略组5: 经典GS算法 (可以被不同AI价值观评估)
    results['Classic GS (judged by GPT-Eng)'] = calculate_expected_blocking_pairs(classic_gs_matchings, source_scores, gender_map, params['gpt4_en']['beta0'], params['gpt4_en']['lambda'], "Classic GS (GPT-Eng Eval)")
    results['Classic GS (judged by DS-Eng)'] = calculate_expected_blocking_pairs(classic_gs_matchings, source_scores, gender_map, params['deepseek_en']['beta0'], params['deepseek_en']['lambda'], "Classic GS (DS-Eng Eval)")


    # 5. 打印和可视化
    # ... (这部分代码保持不变) ...
    print("\n--- Stability Analysis Results: Expected Number of Blocking Pairs (E[numbp]) ---")
    plot_data = []
    for strategy, values in results.items():
        if values:
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"Strategy: {strategy:<30} | Mean E[numbp]: {mean_val:>7.4f} | Std Dev: {std_val:>7.4f} | N_groups: {len(values)}")
            for v in values:
                plot_data.append({'Strategy': strategy, 'E[numbp]': v})
        else:
            print(f"Strategy: {strategy:<30} | No data to calculate or calculation resulted in zero.")
    if not plot_data:
        print("\nNo data to plot. Exiting.")
        exit()
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Strategy', y='E[numbp]', data=df_plot)
    plt.title('Stability Comparison: Expected Number of Blocking Pairs', fontsize=18)
    plt.ylabel('Expected # of Blocking Pairs (Lower is More Stable)')
    plt.xlabel('Matching Strategy & Evaluation Model')
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_filename = "stability_comparison_all_models.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nFinal comparison plot saved to {output_filename}")
    plt.show()