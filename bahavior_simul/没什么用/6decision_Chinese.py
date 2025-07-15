import pandas as pd
import numpy as np
import json
import csv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import re
from tqdm import tqdm

class ChineseMatchingAnalyzer:
    def __init__(self, base_path="/home/lsy/match/bahavior_simul/0629_gpt_Chinese/"):
        self.base_path = base_path
        self.all_decisions = []
        self.single_decisions = []
        self.switch_decisions = []
        self.group_results = {}
        
    def parse_scores_from_prompt(self, prompt):
        """从中文提示词中提取分数信息"""
        scores = {}
        
        # 提取各项分数
        patterns = {
            'attractive': r'吸引力：(\d+\.?\d*)/10',
            'sincere': r'真诚：(\d+\.?\d*)/10',
            'intelligence': r'智力：(\d+\.?\d*)/10',
            'funny': r'幽默感：(\d+\.?\d*)/10',
            'ambition': r'抱负：(\d+\.?\d*)/10',
            'shared_interests': r'共同兴趣：(\d+\.?\d*)/10'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt)
            if match:
                scores[key] = float(match.group(1))
        
        # 提取重要性权重
        weight_patterns = {
            'attractive_weight': r'吸引力：(\d+\.?\d*)，',
            'sincere_weight': r'真诚：(\d+\.?\d*)，',
            'intelligence_weight': r'智力：(\d+\.?\d*)，',
            'funny_weight': r'幽默感：(\d+\.?\d*)，',
            'ambition_weight': r'抱负：(\d+\.?\d*)，',
            'shared_interests_weight': r'共同兴趣：(\d+\.?\d*)'
        }
        
        # 在"重要性权重"之后查找
        weight_section = prompt.split('重要性权重')[1] if '重要性权重' in prompt else prompt
        
        for key, pattern in weight_patterns.items():
            match = re.search(pattern, weight_section)
            if match:
                scores[key] = float(match.group(1))
        
        return scores
    
    def calculate_weighted_score(self, scores):
        """计算加权总分"""
        attributes = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
        total_score = 0
        total_weight = 0
        
        for attr in attributes:
            if attr in scores and f"{attr}_weight" in scores:
                score = scores[attr]
                weight = scores[f"{attr}_weight"]
                total_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0
    
    def parse_csv_group(self, group_num):
        """解析单个组的CSV文件"""
        csv_path = f"{self.base_path}0629_gpt4_Chinese_group{group_num}.csv"
        decisions = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:  # 确保有足够的列
                        prompt = row[0]
                        response = row[1]
                        decision = int(row[5]) if row[5].isdigit() else 0
                        
                        # 解析是否接受
                        accepted = decision == 1
                        
                        # 判断是单身还是已配对
                        is_single = '目前已匹配' not in prompt
                        
                        # 解析分数
                        scores = self.parse_scores_from_prompt(prompt)
                        
                        if is_single:
                            # 单身情况：计算对象的总分
                            proposer_score = self.calculate_weighted_score(scores)
                            
                            decisions.append({
                                'type': 'single',
                                'proposer_score': proposer_score,
                                'accepted': accepted,
                                'group': group_num
                            })
                            
                        else:
                            # 已配对情况：需要解析当前伴侣和新追求者的分数
                            # 提取当前伴侣分数
                            current_section = prompt.split('一位新')[0] if '一位新' in prompt else prompt
                            current_scores = self.parse_scores_from_prompt(current_section)
                            current_score = self.calculate_weighted_score(current_scores)
                            
                            # 提取新追求者分数
                            new_section = prompt.split('一位新')[1] if '一位新' in prompt else prompt
                            new_scores = self.parse_scores_from_prompt(new_section)
                            new_score = self.calculate_weighted_score(new_scores)
                            
                            score_diff = new_score - current_score
                            
                            decisions.append({
                                'type': 'switch',
                                'current_score': current_score,
                                'new_score': new_score,
                                'score_diff': score_diff,
                                'accepted': accepted,
                                'group': group_num
                            })
        
        except Exception as e:
            print(f"Error reading group {group_num}: {e}")
        
        return decisions
    
    def fit_single_threshold(self, single_decisions):
        """拟合单身接受阈值S"""
        if not single_decisions:
            return None
        
        # 提取接受和拒绝的分数
        accepted_scores = [d['proposer_score'] for d in single_decisions if d['accepted']]
        rejected_scores = [d['proposer_score'] for d in single_decisions if not d['accepted']]
        
        if not accepted_scores or not rejected_scores:
            # 如果全部接受或全部拒绝，使用中位数
            all_scores = [d['proposer_score'] for d in single_decisions]
            return np.median(all_scores) if all_scores else 5.0
        
        # 找最优阈值：最小化分类错误
        min_accepted = min(accepted_scores) if accepted_scores else 0
        max_rejected = max(rejected_scores) if rejected_scores else 10
        
        # 阈值应该在最大拒绝分数和最小接受分数之间
        if max_rejected < min_accepted:
            threshold = (max_rejected + min_accepted) / 2
        else:
            # 如果有重叠，找最佳分割点
            all_scores = sorted([d['proposer_score'] for d in single_decisions])
            best_threshold = 5.0
            best_accuracy = 0
            
            for i in range(len(all_scores) - 1):
                threshold = (all_scores[i] + all_scores[i+1]) / 2
                
                # 计算这个阈值的准确率
                correct = 0
                for d in single_decisions:
                    predicted = d['proposer_score'] >= threshold
                    if predicted == d['accepted']:
                        correct += 1
                
                accuracy = correct / len(single_decisions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            threshold = best_threshold
        
        return threshold
    
    def fit_lambda(self, switch_decisions):
        """拟合λ参数"""
        if not switch_decisions:
            return None
        
        # 提取分数差和决策
        score_diffs = np.array([abs(d['score_diff']) for d in switch_decisions])
        decisions = np.array([1 if d['accepted'] else 0 for d in switch_decisions])
        
        # 定义概率函数: P(accept) = exp(λ * |S_a - S_b|) / (1 + exp(λ * |S_a - S_b|))
        # 使用逻辑回归形式更稳定
        def neg_log_likelihood(lambda_param):
            # 计算概率
            z = lambda_param * score_diffs
            # 防止数值溢出
            z = np.clip(z, -100, 100)
            probs = 1 / (1 + np.exp(-z))
            
            # 避免log(0)
            probs = np.clip(probs, 1e-10, 1-1e-10)
            
            # 负对数似然
            nll = -np.sum(decisions * np.log(probs) + (1 - decisions) * np.log(1 - probs))
            return nll
        
        # 优化
        result = minimize(neg_log_likelihood, x0=[1.0], bounds=[(0.01, 10)], method='L-BFGS-B')
        lambda_opt = result.x[0]
        
        return lambda_opt
    
    def analyze_all_groups(self):
        """分析所有21个组"""
        print("分析中文组数据...")
        
        all_single = []
        all_switch = []
        
        for group_num in tqdm(range(1, 22)):
            decisions = self.parse_csv_group(group_num)
            
            if decisions:
                # 分离单身和换伴侣决策
                single = [d for d in decisions if d['type'] == 'single']
                switch = [d for d in decisions if d['type'] == 'switch']
                
                all_single.extend(single)
                all_switch.extend(switch)
                
                # 单组分析
                s_threshold = self.fit_single_threshold(single) if single else None
                lambda_param = self.fit_lambda(switch) if switch else None
                
                self.group_results[group_num] = {
                    'n_single': len(single),
                    'n_switch': len(switch),
                    's_threshold': s_threshold,
                    'lambda': lambda_param,
                    'single_accept_rate': np.mean([d['accepted'] for d in single]) if single else None,
                    'switch_accept_rate': np.mean([d['accepted'] for d in switch]) if switch else None
                }
        
        # 总体分析
        self.overall_s = self.fit_single_threshold(all_single)
        self.overall_lambda = self.fit_lambda(all_switch)
        
        self.all_decisions = all_single + all_switch
        self.single_decisions = all_single
        self.switch_decisions = all_switch
    
    def plot_results(self):
        """可视化结果"""
        # 准备数据
        groups = sorted(self.group_results.keys())
        lambdas = [self.group_results[g].get('lambda', None) for g in groups]
        thresholds = [self.group_results[g].get('s_threshold', None) for g in groups]
        
        # 过滤None值
        valid_lambdas = [(g, l) for g, l in zip(groups, lambdas) if l is not None]
        valid_thresholds = [(g, t) for g, t in zip(groups, thresholds) if t is not None]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Lambda by group
        if valid_lambdas:
            g, l = zip(*valid_lambdas)
            axes[0, 0].bar(g, l, color='blue', alpha=0.7)
            axes[0, 0].axhline(y=self.overall_lambda, color='red', linestyle='--',
                              label=f'总体 λ={self.overall_lambda:.3f}')
            axes[0, 0].set_xlabel('组别')
            axes[0, 0].set_ylabel('λ 参数')
            axes[0, 0].set_title('各组 Lambda 参数')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # S threshold by group
        if valid_thresholds:
            g, t = zip(*valid_thresholds)
            axes[0, 1].bar(g, t, color='green', alpha=0.7)
            axes[0, 1].axhline(y=self.overall_s, color='red', linestyle='--',
                              label=f'总体阈值 S={self.overall_s:.2f}')
            axes[0, 1].set_xlabel('组别')
            axes[0, 1].set_ylabel('接受阈值 S')
            axes[0, 1].set_title('各组单身接受阈值')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 接受概率 vs 分数差（换伴侣情况）
        if self.switch_decisions:
            score_diffs = [abs(d['score_diff']) for d in self.switch_decisions]
            accepted = [d['accepted'] for d in self.switch_decisions]
            
            # 分箱统计
            bins = np.linspace(0, max(score_diffs), 10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            accept_rates = []
            
            for i in range(len(bins)-1):
                mask = (np.array(score_diffs) >= bins[i]) & (np.array(score_diffs) < bins[i+1])
                if np.sum(mask) > 0:
                    accept_rates.append(np.mean([accepted[j] for j in range(len(accepted)) if mask[j]]))
                else:
                    accept_rates.append(None)
            
            # 画实际数据
            valid_points = [(c, r) for c, r in zip(bin_centers, accept_rates) if r is not None]
            if valid_points:
                c, r = zip(*valid_points)
                axes[1, 0].scatter(c, r, s=100, alpha=0.7, label='实际数据')
            
            # 画拟合曲线
            x_fit = np.linspace(0, max(score_diffs), 100)
            y_fit = 1 / (1 + np.exp(-self.overall_lambda * x_fit))
            axes[1, 0].plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合曲线 (λ={self.overall_lambda:.3f})')
            
            axes[1, 0].set_xlabel('分数差 |S_a - S_b|')
            axes[1, 0].set_ylabel('接受概率')
            axes[1, 0].set_title('换伴侣接受概率 vs 分数差')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 单身接受分布
        if self.single_decisions:
            accepted_single = [d['proposer_score'] for d in self.single_decisions if d['accepted']]
            rejected_single = [d['proposer_score'] for d in self.single_decisions if not d['accepted']]
            
            if accepted_single:
                axes[1, 1].hist(accepted_single, bins=20, alpha=0.5, label='接受', color='green')
            if rejected_single:
                axes[1, 1].hist(rejected_single, bins=20, alpha=0.5, label='拒绝', color='red')
            
            axes[1, 1].axvline(x=self.overall_s, color='black', linestyle='--', linewidth=2,
                               label=f'阈值 S={self.overall_s:.2f}')
            axes[1, 1].set_xlabel('追求者得分')
            axes[1, 1].set_ylabel('频数')
            axes[1, 1].set_title('单身接受/拒绝分布')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chinese_matching_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """评估模型性能"""
        # 单身模型评估
        if self.single_decisions:
            single_pred = [d['proposer_score'] >= self.overall_s for d in self.single_decisions]
            single_true = [d['accepted'] for d in self.single_decisions]
            single_accuracy = accuracy_score(single_true, single_pred)
        else:
            single_accuracy = None
        
        # 换伴侣模型评估
        if self.switch_decisions:
            switch_probs = [1/(1+np.exp(-self.overall_lambda*abs(d['score_diff']))) 
                           for d in self.switch_decisions]
            switch_pred = [p > 0.5 for p in switch_probs]
            switch_true = [d['accepted'] for d in self.switch_decisions]
            switch_accuracy = accuracy_score(switch_true, switch_pred)
        else:
            switch_accuracy = None
        
        return single_accuracy, switch_accuracy
    
    def print_summary(self):
        """打印汇总结果"""
        print("\n" + "="*60)
        print("中文组匹配分析汇总")
        print("="*60)
        
        print(f"\n总体样本数: {len(self.all_decisions)}")
        print(f"  - 单身决策: {len(self.single_decisions)}")
        print(f"  - 换伴侣决策: {len(self.switch_decisions)}")
        
        print(f"\n总体参数:")
        print(f"  - 单身接受阈值 S = {self.overall_s:.3f}")
        print(f"  - 换伴侣参数 λ = {self.overall_lambda:.3f}")
        
        single_acc, switch_acc = self.evaluate_model()
        print(f"\n模型准确率:")
        if single_acc is not None:
            print(f"  - 单身模型: {single_acc:.1%}")
        if switch_acc is not None:
            print(f"  - 换伴侣模型: {switch_acc:.1%}")
        
        print(f"\n接受概率函数:")
        print(f"  - 单身: P(accept) = 1 if Score >= {self.overall_s:.3f}, else 0")
        print(f"  - 换伴侣: P(accept) = 1 / (1 + exp(-{self.overall_lambda:.3f} * |S_a - S_b|))")
        
        # 保存详细结果
        results_df = pd.DataFrame(self.group_results).T
        results_df.to_csv('chinese_group_analysis_results.csv')
        print("\n详细结果已保存至 'chinese_group_analysis_results.csv'")

def main():
    """主函数"""
    analyzer = ChineseMatchingAnalyzer()
    
    # 分析所有组
    analyzer.analyze_all_groups()
    
    # 可视化
    analyzer.plot_results()
    
    # 打印汇总
    analyzer.print_summary()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()