# 文件名: compare_4_reason_keywords_simplified_en.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import re

# 导入你的项目配置和数据加载函数
from config import config  # 假设config.py已更新
from load_data import load_all_group_data_for_model # 确保此函数能正确加载数据

# --- 配置 ---
sns.set_style(config.get("plot_style", "whitegrid"))
sns.set_context(config.get("plot_context", "talk"))

# 英文关键词类别 (可以根据需要调整)
KEYWORD_CATEGORIES_EN = {
    'Attractiveness': ['attractive', 'attraction', 'appearance', 'look', 'physical', 'beauty', 'handsome', 'pretty'],
    'Sincerity': ['sincere', 'sincerity', 'honest', 'honesty', 'genuine', 'trust', 'truthful'],
    'Intelligence': ['intelligence', 'intelligent', 'smart', 'intellectual', 'knowledgeable', 'bright', 'clever'],
    'Funny': ['funny', 'humor', 'humour', 'laugh', 'entertaining', 'jovial', 'witty'],
    'Ambition': ['ambition', 'ambitious', 'career', 'driven', 'goal', 'aspiring', 'success-oriented'],
    'Shared Interests': ['shared interest', 'common interest', 'hobby', 'activity', 'compatible', 'similarity', 'connect']
}

def simple_preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # 移除标点
    text = re.sub(r'\s+', ' ', text).strip() # 规范化空格
    return text.split() # 按空格分词

def calculate_keyword_proportions_simple(df, keyword_categories):
    all_tokens_flat = []
    
    if 'reason' not in df.columns or df['reason'].dropna().empty:
        return pd.Series({category: 0.0 for category in keyword_categories.keys()}, dtype=float), 0

    for reason_text in df['reason'].dropna().astype(str):
        all_tokens_flat.extend(simple_preprocess_text(reason_text))

    total_words = len(all_tokens_flat)
    if total_words == 0:
        return pd.Series({category: 0.0 for category in keyword_categories.keys()}, dtype=float), 0

    word_counts = Counter(all_tokens_flat)
    
    category_proportions = {}
    for category, keywords in keyword_categories.items():
        category_token_count = 0
        # 对关键词也进行小写处理以匹配
        lower_keywords = {kw.lower() for kw in keywords}
        
        for token, count in word_counts.items():
            if token in lower_keywords: # 直接在分词结果中查找小写关键词
                category_token_count += count
        
        category_proportions[category] = (category_token_count / total_words) * 100 if total_words > 0 else 0
        
    return pd.Series(category_proportions), total_words

def analyze_and_plot():
    print("--- Comparative Analysis 4: Simplified Keyword Proportion in Decision Reasons (English Output) ---")
    
    model1_key = 'gpt4_en' 
    model2_key = 'deepseek_en'

    if model1_key not in config or model2_key not in config:
        print(f"Error: Configuration for '{model1_key}' or '{model2_key}' not found in config.py.")
        return

    df_model1, _ = load_all_group_data_for_model(model1_key)
    df_model2, _ = load_all_group_data_for_model(model2_key)

    label_model1 = config[model1_key].get('label', 'Model 1') + f" (N={config[model1_key].get('num_groups', 0)})"
    label_model2 = config[model2_key].get('label', 'Model 2') + f" (N={config[model2_key].get('num_groups', 0)})"

    proportions_model1, total_words_model1 = calculate_keyword_proportions_simple(df_model1, KEYWORD_CATEGORIES_EN)
    proportions_model2, total_words_model2 = calculate_keyword_proportions_simple(df_model2, KEYWORD_CATEGORIES_EN)

    print(f"{label_model1} - Total Words in Reasons: {total_words_model1}")
    print(f"{label_model2} - Total Words in Reasons: {total_words_model2}")

    # --- 图表1: 横向条形对比图 (关键词占比) ---
    df_plot_model1_bar = proportions_model1.reset_index()
    df_plot_model1_bar.columns = ['Dimension', 'Proportion (%)']
    df_plot_model1_bar['Model'] = label_model1

    df_plot_model2_bar = proportions_model2.reset_index()
    df_plot_model2_bar.columns = ['Dimension', 'Proportion (%)']
    df_plot_model2_bar['Model'] = label_model2
    
    combined_plot_df_bar = pd.concat([df_plot_model1_bar, df_plot_model2_bar])

    plt.figure(figsize=config.get("figure_size_large", (14, 9)))
    sns.barplot(x='Proportion (%)', y='Dimension', hue='Model', data=combined_plot_df_bar, 
                palette=config.get("plot_palette_models", "viridis"), orient='h')
    plt.title('GPT-4 vs DeepSeek: Keyword Proportion by Dimension', fontsize=config.get("title_fontsize", 18))
    plt.xlabel('Proportion of Keywords in Reason Tokens (%)', fontsize=config.get("label_fontsize", 14))
    plt.ylabel('Evaluation Dimension', fontsize=config.get("label_fontsize", 14))
    plt.legend(title='Model')
    plt.tight_layout()
    output_filename_bar = "compare_4_keyword_proportion_bar.png"
    plt.savefig(output_filename_bar, dpi=300, bbox_inches='tight')
    print(f"Chart 1 (Bar Comparison) saved as: {output_filename_bar}")
    plt.close()

    # --- 图表2: 饼图 (每个模型内部各维度占比) ---
    def plot_pie_chart(proportions_series, model_label_full, filename_suffix):
        # 过滤掉占比为0的维度，使饼图更清晰
        proportions_to_plot = proportions_series[proportions_series > 0]
        if proportions_to_plot.empty:
            print(f"No keyword data to plot for {model_label_full}, skipping pie chart.")
            return

        plt.figure(figsize=config.get("figure_size_small", (8, 8)))
        plt.pie(proportions_to_plot, labels=proportions_to_plot.index, autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette(config.get("plot_palette_pastel", "pastel"), len(proportions_to_plot)))
        plt.title(f'Keyword Category Proportions for {model_label_full}', fontsize=config.get("title_fontsize", 16))
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        output_filename_pie = f"compare_4_keyword_pie_{filename_suffix}.png"
        plt.savefig(output_filename_pie, dpi=300, bbox_inches='tight')
        print(f"Chart 2 (Pie - {model_label_full}) saved as: {output_filename_pie}")
        plt.close()

    plot_pie_chart(proportions_model1, label_model1, "model1")
    plot_pie_chart(proportions_model2, label_model2, "model2")

    # --- 图表3: 排序对比图 (例如，用点图比较两个模型对维度的排序) ---
    # 合并两个模型的占比数据，并增加排名
    df_rank_model1 = proportions_model1.sort_values(ascending=False).reset_index()
    df_rank_model1.columns = ['Dimension', 'Proportion_M1']
    df_rank_model1['Rank_M1'] = df_rank_model1.index + 1

    df_rank_model2 = proportions_model2.sort_values(ascending=False).reset_index()
    df_rank_model2.columns = ['Dimension', 'Proportion_M2']
    df_rank_model2['Rank_M2'] = df_rank_model2.index + 1
    
    # 合并用于绘图的数据
    df_rank_compare = pd.merge(df_rank_model1[['Dimension', 'Rank_M1']], 
                               df_rank_model2[['Dimension', 'Rank_M2']], 
                               on='Dimension', how='outer').fillna(len(KEYWORD_CATEGORIES_EN) + 1) # 未提及的维度排最后

    # 为了更好的可视化，我们可以画一个散点图，X轴是模型1的排名，Y轴是模型2的排名
    # 或者用平行坐标图的思想来画连接线
    plt.figure(figsize=config.get("figure_size_medium", (10, 8)))
    for i, row in df_rank_compare.iterrows():
        plt.plot([1, 2], [row['Rank_M1'], row['Rank_M2']], marker='o', label=row['Dimension'] if i < 7 else None) # 只标前几个
        plt.text(1, row['Rank_M1'], f" {row['Dimension']}", va='center', ha='right', fontsize=9)
        plt.text(2, row['Rank_M2'], f" {row['Dimension']}", va='center', ha='left', fontsize=9)


    plt.xticks([1, 2], [label_model1.split(' (N=')[0], label_model2.split(' (N=')[0]]) # 只取模型名
    plt.gca().invert_yaxis() # 排名第一的在最上面
    plt.title('Ranking of Keyword Dimensions by Proportion', fontsize=config.get("title_fontsize", 18))
    plt.ylabel('Rank (Lower is Better/More Frequent)', fontsize=config.get("label_fontsize", 14))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # 如果标签太多，图例会很乱
    output_filename_rank = "compare_4_keyword_rank_comparison.png"
    plt.savefig(output_filename_rank, dpi=300, bbox_inches='tight')
    print(f"Chart 3 (Rank Comparison) saved as: {output_filename_rank}")
    plt.close()


if __name__ == '__main__':
    analyze_and_plot()