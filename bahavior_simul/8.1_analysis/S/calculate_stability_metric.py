import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# 从你的 load_data.py 和 config.py 导入
from load_data import load_all_group_data_for_model, load_source_scores
from config import config

# --- 1. 定义我们需要的概率预测函数 ---
def sigmoid_model(x, beta0, lambda_param):
    """标准的二维Sigmoid函数，用于拟合"""
    return 1.0 / (1.0 + np.exp(-(beta0 + lambda_param * x)))

def predict_prob_switch(Sa, Sb, beta0, lambda_param):
    """
    使用拟合出的参数，预测换人的概率
    Sa: 对新选择的分数
    Sb: 对旧选择的分数
    """
    score_diff = Sa - Sb
    return sigmoid_model(score_diff, beta0, lambda_param)

# --- 2. 拟合模型以获取参数 (这部分代码从之前的脚本中提取和简化) ---
def get_fitted_params(model_key):
    # a. 准备数据
    df_log, _ = load_all_group_data_for_model(model_key)
    if df_log.empty: return None, None
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None: return None, None
    
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

    # b. 分箱和拟合
    if df_taken.empty: return None, None
    df_taken['diff_bin'] = pd.cut(df_taken['score_diff'], bins=20)
    binned_taken = df_taken.groupby('diff_bin', observed=False)['result'].mean().reset_index()
    binned_taken['diff_mid'] = binned_taken['diff_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    binned_taken.dropna(inplace=True)
    
    x_data = binned_taken['diff_mid'].to_numpy()
    y_data = binned_taken['result'].to_numpy()
    
    try:
        popt, _ = curve_fit(sigmoid_model, x_data, y_data, p0=[0, 0.1], maxfev=5000)
        beta0_fit, lambda_fit = popt
        print(f"--- Model Parameters Fitted for {model_key} ---")
        print(f"Fitted beta0: {beta0_fit:.4f}, Fitted lambda: {lambda_fit:.4f}")
        return beta0_fit, lambda_fit
    except Exception as e:
        print(f"Could not fit model for {model_key}: {e}")
        return None, None

# --- 3. 计算期望阻塞对数量 ---
def calculate_expected_blocking_pairs(model_key, beta0, lambda_param):
    # 加载所有组的JSON最终匹配结果
    _, all_matchings = load_all_group_data_for_model(model_key)
    
    # 加载原始评分数据
    source_scores = load_source_scores(config["source_data_path"])
    if source_scores is None or not all_matchings:
        print("Missing source scores or matching data. Cannot calculate stability.")
        return []

    expected_bp_counts = []
    
    for group_idx, matching in enumerate(all_matchings):
        if not matching: continue
        
        # 区分男性和女性 (这里做一个简单的假设，你需要根据你的数据调整)
        # 假设ID < 200 为男性, 否则为女性
        all_participants = list(matching.keys())
        men = {p for p in all_participants if int(p) < 200}
        women = {p for p in all_participants if int(p) >= 200}
        
        total_prob_sum = 0
        
        # 遍历所有可能的 (男, 女) 组合
        for m in men:
            for w in women:
                m_partner = matching.get(m)
                
                # 如果 (m, w) 已经是配对，则他们不是潜在的阻塞对
                if str(m_partner) == str(w):
                    continue

                # 找到 w 的伴侣
                w_partner = matching.get(w)
                
                # --- 计算 prob1: m 更喜欢 w 的概率 ---
                # 如果 m 是单身/被拒绝，我们假设他会更喜欢任何一个潜在伴侣
                # P(m prefers w > m_p)
                if m_partner is None or str(m_partner).lower() == 'rejected':
                    prob1 = 1.0 # 单身男性总是会尝试去追求
                else:
                    m_partner = int(m_partner)
                    score_m_w = source_scores.get((m, w))
                    score_m_mp = source_scores.get((m, m_partner))
                    if score_m_w is None or score_m_mp is None: continue
                    prob1 = predict_prob_switch(score_m_w, score_m_mp, beta0, lambda_param)

                # --- 计算 prob2: w 更喜欢 m 的概率 ---
                # P(w prefers m > w_p)
                if w_partner is None or str(w_partner).lower() == 'rejected':
                    prob2 = 1.0 # 单身女性也总是会考虑
                else:
                    w_partner = int(w_partner)
                    score_w_m = source_scores.get((w, m))
                    score_w_wp = source_scores.get((w, w_partner))
                    if score_w_m is None or score_w_wp is None: continue
                    prob2 = predict_prob_switch(score_w_m, score_w_wp, beta0, lambda_param)

                # 计算该对成为阻塞对的概率并累加
                p_bp = prob1 * prob2
                total_prob_sum += p_bp
        
        expected_bp_counts.append(total_prob_sum)
        print(f"Group {group_idx + 1}: Expected number of blocking pairs = {total_prob_sum:.2f}")

    return expected_bp_counts


# --- 主程序 ---
if __name__ == '__main__':
    model_key_to_analyze = 'gpt4_en_fitting'
    
    print(f"--- Starting Stability Analysis for: {model_key_to_analyze} ---")
    
    # 1. 拟合模型，得到beta0和lambda
    beta0, lambda_val = get_fitted_params(model_key_to_analyze)
    
    if beta0 is not None and lambda_val is not None:
        # 2. 使用拟合出的参数计算期望阻塞对数量
        e_numbp_list = calculate_expected_blocking_pairs(model_key_to_analyze, beta0, lambda_val)
        
        if e_numbp_list:
            # 3. 报告结果
            mean_e_numbp = np.mean(e_numbp_list)
            std_e_numbp = np.std(e_numbp_list)
            
            print("\n--- Final Stability Results ---")
            print(f"Model: {config[model_key_to_analyze].get('label', model_key_to_analyze)}")
            print(f"Number of experiment groups analyzed: {len(e_numbp_list)}")
            print(f"Average Expected Number of Blocking Pairs (E[numbp]): {mean_e_numbp:.4f}")
            print(f"Standard Deviation of E[numbp]: {std_e_numbp:.4f}")

            # 4. 可视化结果分布
            plt.figure(figsize=(8, 6))
            sns.histplot(e_numbp_list, kde=True, bins=10)
            plt.title(f'Distribution of Expected Blocking Pairs\nfor {config[model_key_to_analyze].get("label", model_key_to_analyze)}')
            plt.xlabel('Expected Number of Blocking Pairs (E[numbp])')
            plt.ylabel('Frequency')
            plt.axvline(mean_e_numbp, color='red', linestyle='--', label=f'Mean = {mean_e_numbp:.2f}')
            plt.legend()
            output_filename = f"stability_results_{model_key_to_analyze}.png"
            plt.savefig(output_filename, dpi=300)
            print(f"Stability analysis plot saved to {output_filename}")
            plt.show()


# 终端输出:
# 首先会打印出拟合出的 beta0 和 lambda 值，让你确认模型参数。
# 然后会逐组打印计算出的期望阻塞对数量。
# 最后会给出一个总的平均值和标准差。Average Expected Number of Blocking Pairs 就是你最终想要的稳定性指标。
# 图片输出 (stability_results_...png):
# 这张直方图展示了在21组实验中，期望阻塞对数量的分布情况。
# 如果分布很集中，说明该模型下的匹配稳定性是比较一致的。如果分布很分散，说明稳定性波动很大。
# 红色的均值线就是最终的平均指标。