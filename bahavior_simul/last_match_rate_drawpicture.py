import matplotlib.pyplot as plt
def analyze_matching_rates(matchings_list):
    rates = []
    for matching_data in matchings_list:
        matched_count = 0
        total_participants = len(matching_data)
        if total_participants == 0: continue
        
        # 统计配对成功的人数 (排除单身或被拒绝的)
        matched_ids = {p for p, partner in matching_data.items() if partner is not None and partner != 'rejected'}
        rates.append(len(matched_ids) / total_participants * 100)
    return rates

rates_en_dist = analyze_matching_rates(matchings_en)
rates_zh_dist = analyze_matching_rates(matchings_zh)

# 创建DataFrame用于绘图
plot_df_rate = pd.DataFrame({
    '配对率 (%)': rates_en_dist + rates_zh_dist,
    'Prompt 语言': ['English'] * len(rates_en_dist) + ['Chinese'] * len(rates_zh_dist)
})

# 使用箱形图+散点图可视化
plt.figure(figsize=(12, 8))
sns.boxplot(x='Prompt 语言', y='配对率 (%)', data=plot_df_rate, palette="pastel")
sns.stripplot(x='Prompt 语言', y='配对率 (%)', data=plot_df_rate, color=".25", size=4)
plt.title(f'中英文Prompt下最终配对率分布 (基于{num_groups}组实验)', fontsize=18)
plt.ylabel('配对率 (%)', fontsize=14)
plt.xlabel('Prompt 语言', fontsize=14)
# 添加平均值标注
mean_en = np.mean(rates_en_dist)
mean_zh = np.mean(rates_zh_dist)
plt.text(0, plt.ylim()[1]*0.95, f'平均值: {mean_en:.2f}%', ha='center', va='bottom', color='blue', fontsize=12)
plt.text(1, plt.ylim()[1]*0.95, f'平均值: {mean_zh:.2f}%', ha='center', va='bottom', color='orange', fontsize=12)
plt.show()