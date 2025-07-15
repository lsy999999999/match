import pandas as pd
import numpy as np
import json
import re
from scipy.optimize import minimize_scalar, curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Remove sklearn dependency - implement mean_squared_error manually
def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

class AcceptanceProbabilityAnalyzer:
    def __init__(self, base_path, csv_prefix, json_prefix):
        self.base_path = base_path
        self.csv_prefix = csv_prefix
        self.json_prefix = json_prefix
        self.all_decisions = []
        self.single_thresholds = []
        
    def extract_scores_from_prompt(self, prompt):
        """Extract individual scores from the prompt string"""
        # Pattern to extract scores like "attractiveness: 7/10"
        pattern = r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
        
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            scores = [float(match.group(i)) for i in range(1, 7)]
            return scores
        return None
    
    def extract_importance_weights(self, prompt):
        """Extract importance weights from the prompt string"""
        # Pattern to extract importance weights
        pattern = r'importance weights.*?attractiveness:\s*(\d+(?:\.\d+)?).*?sincerity:\s*(\d+(?:\.\d+)?).*?intelligence:\s*(\d+(?:\.\d+)?).*?being funny:\s*(\d+(?:\.\d+)?).*?ambition:\s*(\d+(?:\.\d+)?).*?shared interests:\s*(\d+(?:\.\d+)?)'
        
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            weights = [float(match.group(i)) for i in range(1, 7)]
            return weights
        return None
    
    def calculate_weighted_score(self, scores, weights):
        """Calculate weighted average score"""
        if scores is None or weights is None:
            return None
        
        scores = np.array(scores)
        weights = np.array(weights)
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Calculate weighted average
        return np.sum(scores * weights)
    
    def parse_csv_file(self, group_id):
        """Parse a single CSV file and extract decision data"""
        csv_path = f"{self.base_path}/{self.csv_prefix}{group_id}.csv"
        
        try:
            # First, let's check if the file exists
            import os
            if not os.path.exists(csv_path):
                print(f"Warning: File not found: {csv_path}")
                return
            
            # Read CSV with proper handling
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"Warning: Empty file: {csv_path}")
                return
            
            # Process each line
            for line_num, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Try to parse as CSV
                import csv
                reader = csv.reader([line])
                try:
                    row = next(reader)
                except:
                    continue
                
                if len(row) < 6:  # Need at least prompt, response, target, proposer, current/empty, result
                    continue
                
                prompt = str(row[0]).strip()
                response = str(row[1]).strip()
                
                # Debug first few lines
                if line_num < 2:
                    print(f"\nGroup {group_id}, Line {line_num}:")
                    print(f"  Prompt length: {len(prompt)}")
                    print(f"  Response: {response[:100]}...")
                
                # Skip if no valid prompt
                if not prompt or len(prompt) < 50:
                    continue
                
                # Skip API errors
                if 'API_ERROR' in response or 'API_ERROR' in prompt:
                    # But try to extract the decision from API_ERROR responses
                    if 'YES' in response:
                        decision = 1
                    elif 'NO' in response:
                        decision = 0
                    else:
                        continue
                else:
                    # Extract decision from normal responses
                    decision = 1 if 'YES' in response.upper() else 0
                
                # Extract scores and weights
                scores = self.extract_scores_from_prompt(prompt)
                weights = self.extract_importance_weights(prompt)
                
                if scores and weights:
                    weighted_score = self.calculate_weighted_score(scores, weights)
                    
                    # Check if it's a single person decision or switching decision
                    if 'currently matched with a partner' in prompt:
                        # This is a switching decision - extract both scores
                        # Extract current partner scores
                        current_pattern = r"current partner's scores.*?" + r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
                        current_match = re.search(current_pattern, prompt, re.IGNORECASE | re.DOTALL)
                        
                        # Extract new person scores
                        new_pattern = r"(?:new woman is courting you|new man is courting you) with scores:.*?" + r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
                        new_match = re.search(new_pattern, prompt, re.IGNORECASE | re.DOTALL)
                        
                        if current_match and new_match:
                            current_scores = [float(current_match.group(i)) for i in range(1, 7)]
                            new_scores = [float(new_match.group(i)) for i in range(1, 7)]
                            current_weighted = self.calculate_weighted_score(current_scores, weights)
                            new_weighted = self.calculate_weighted_score(new_scores, weights)
                            
                            self.all_decisions.append({
                                'type': 'switch',
                                'current_score': current_weighted,
                                'new_score': new_weighted,
                                'score_diff': new_weighted - current_weighted,
                                'decision': decision,
                                'group': group_id
                            })
                    else:
                        # Single person decision
                        self.all_decisions.append({
                            'type': 'single',
                            'score': weighted_score,
                            'decision': decision,
                            'group': group_id
                        })
                        
                        # Track thresholds for single decisions
                        if decision == 0:  # Rejected
                            self.single_thresholds.append(weighted_score)
                            
        except Exception as e:
            print(f"Error parsing CSV for group {group_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def load_all_data(self, groups):
        """Load data from all groups"""
        for group_id in groups:
            print(f"Loading group {group_id}...")
            self.parse_csv_file(group_id)
        
        print(f"\nTotal decisions loaded: {len(self.all_decisions)}")
        single_count = sum(1 for d in self.all_decisions if d['type'] == 'single')
        switch_count = sum(1 for d in self.all_decisions if d['type'] == 'switch')
        print(f"Single decisions: {single_count}")
        print(f"Switch decisions: {switch_count}")
    
    def fit_lambda_parameter(self):
        """Fit the lambda parameter for the acceptance probability function"""
        if not self.all_decisions:
            print("ERROR: No decisions loaded! Cannot fit parameters.")
            return None
            
        # Separate single and switch decisions
        single_decisions = [d for d in self.all_decisions if d['type'] == 'single']
        switch_decisions = [d for d in self.all_decisions if d['type'] == 'switch']
        
        # Estimate threshold S for single decisions
        # Method 1: Original method - maximum rejected score
        if self.single_thresholds:
            # Use the maximum rejected score as an estimate for the threshold
            estimated_threshold = max(self.single_thresholds)
        else:
            # If no rejections, use the minimum accepted score
            accepted_scores = [d['score'] for d in single_decisions if d['decision'] == 1]
            estimated_threshold = min(accepted_scores) if accepted_scores else 5.0
        
        # Method 2: Average of max rejected and min accepted scores
        rejected_scores = [d['score'] for d in single_decisions if d['decision'] == 0]
        accepted_scores = [d['score'] for d in single_decisions if d['decision'] == 1]
        
        if rejected_scores and accepted_scores:
            max_rejected = max(rejected_scores)
            min_accepted = min(accepted_scores)
            estimated_threshold_s2 = (max_rejected + min_accepted) / 2
        elif rejected_scores:
            # Only rejections, use max rejected
            estimated_threshold_s2 = max(rejected_scores)
        elif accepted_scores:
            # Only acceptances, use min accepted
            estimated_threshold_s2 = min(accepted_scores)
        else:
            # No data, use default
            estimated_threshold_s2 = 5.0
        
        print(f"\nEstimated threshold S for single decisions:")
        print(f"  Method 1 (max rejected): {estimated_threshold:.2f}")
        print(f"  Method 2 (average of boundaries): {estimated_threshold_s2:.2f}")
        
        # You can choose which method to use for fitting
        # For now, let's report both but use Method 2 for fitting
        threshold_for_fitting = estimated_threshold_s2
        
        # Prepare data for fitting
        X_data = []
        y_data = []
        
        # New definition: Sa - Sb (without absolute value)
        # Sa = score of accepted partner after decision
        # Sb = score of rejected partner after decision
        
        # Statistics counter
        positive_diff_count = 0  # Count when accepted > rejected
        
        # Add single decision data
        for d in single_decisions:
            if d['decision'] == 1:  # Accepted
                # Sa = proposer's score, Sb = threshold (rejected below this)
                score_diff = d['score'] - threshold_for_fitting
            else:  # Rejected
                # Sa = threshold (accepted above this), Sb = proposer's score
                score_diff = threshold_for_fitting - d['score']
            
            X_data.append(score_diff)
            y_data.append(d['decision'])
            
            if score_diff > 0:
                positive_diff_count += 1
        
        # Add switch decision data
        for d in switch_decisions:
            if d['decision'] == 1:  # Switched
                # Sa = new partner score, Sb = current partner score
                score_diff = d['new_score'] - d['current_score']
            else:  # Stayed
                # Sa = current partner score, Sb = new proposer score
                score_diff = d['current_score'] - d['new_score']
            
            X_data.append(score_diff)
            y_data.append(d['decision'])
            
            if score_diff > 0:
                positive_diff_count += 1
        
        if not X_data:
            print("ERROR: No valid data for fitting!")
            return None
            
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Fitting with {len(X_data)} data points")
        print(f"Cases where accepted score > rejected score: {positive_diff_count} ({positive_diff_count/len(X_data)*100:.1f}%)")
        
        # Show distribution of score differences
        print(f"\nScore difference statistics:")
        print(f"  Min difference: {min(X_data):.2f}")
        print(f"  Max difference: {max(X_data):.2f}")
        print(f"  Mean difference: {np.mean(X_data):.2f}")
        print(f"  Median difference: {np.median(X_data):.2f}")
        
        # Define the probability function - sigmoid function
        def prob_function(x, lambda_param):
            return 1 / (1 + np.exp(-lambda_param * x))  # Sigmoid: P = 1/(1+e^(-λ(Sa-Sb)))
        
        # Fit using different methods
        print("\nFitting lambda parameter...")
        
        # Method 1: Direct curve fitting
        try:
            # Sigmoid function is already bounded in [0,1], no need for additional constraints
            popt, pcov = curve_fit(prob_function, X_data, y_data, p0=[1.0], bounds=(0, 10))
            lambda_fit = popt[0]
            print(f"Curve fit lambda: {lambda_fit:.4f}")
        except Exception as e:
            lambda_fit = None
            print(f"Curve fitting failed: {e}")
        
        # Method 2: Maximum likelihood estimation
        def negative_log_likelihood(lambda_param):
            if lambda_param <= 0:
                return np.inf
            
            # Sigmoid function automatically bounded in [0,1]
            probs = prob_function(X_data, lambda_param)
            # Small epsilon to avoid log(0)
            probs = np.clip(probs, 1e-10, 1-1e-10)
            
            # Log likelihood
            ll = np.sum(y_data * np.log(probs) + (1 - y_data) * np.log(1 - probs))
            return -ll
        
        result = minimize_scalar(negative_log_likelihood, bounds=(0.01, 10), method='bounded')
        lambda_mle = result.x
        print(f"MLE lambda: {lambda_mle:.4f}")
        
        # Method 3: Minimize MSE between predicted probabilities and actual decisions
        def mse_objective(lambda_param):
            probs = prob_function(X_data, lambda_param)
            return mean_squared_error(y_data, probs)
        
        result_mse = minimize_scalar(mse_objective, bounds=(0.01, 10), method='bounded')
        lambda_mse = result_mse.x
        print(f"MSE minimization lambda: {lambda_mse:.4f}")
        
        # Return results
        return {
            'threshold_S': estimated_threshold,
            'threshold_S2': estimated_threshold_s2,
            'threshold_used_for_fitting': threshold_for_fitting,
            'lambda_curvefit': lambda_fit,
            'lambda_mle': lambda_mle,
            'lambda_mse': lambda_mse,
            'positive_diff_count': positive_diff_count,
            'positive_diff_percentage': positive_diff_count/len(X_data)*100 if len(X_data) > 0 else 0,
            'data': {
                'X': X_data,
                'y': y_data,
                'single_decisions': single_decisions,
                'switch_decisions': switch_decisions
            }
        }
    
    def visualize_results(self, results):
        """Visualize the fitted acceptance probability function"""
        if results is None:
            print("No results to visualize!")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Acceptance rate vs score difference (now with signed differences)
        ax = axes[0, 0]
        X_data = results['data']['X']
        y_data = results['data']['y']
        
        # Bin the data - now including negative values
        x_min, x_max = min(X_data), max(X_data)
        bins = np.linspace(x_min, x_max, 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        acceptance_rates = []
        
        for i in range(len(bins)-1):
            mask = (X_data >= bins[i]) & (X_data < bins[i+1])
            if mask.sum() > 0:
                acceptance_rates.append(y_data[mask].mean())
            else:
                acceptance_rates.append(np.nan)
        
        ax.scatter(bin_centers, acceptance_rates, s=100, alpha=0.6, label='Empirical acceptance rate')
        
        # Plot fitted functions
        x_range = np.linspace(x_min, x_max, 100)
        for lambda_name, lambda_val in [('MLE', results['lambda_mle']), 
                                       ('MSE', results['lambda_mse'])]:
            if lambda_val:
                y_pred = 1 / (1 + np.exp(-lambda_val * x_range))  # Sigmoid function
                ax.plot(x_range, y_pred, label=f'{lambda_name}: λ={lambda_val:.3f}')
        
        ax.axvline(0, color='black', linestyle=':', alpha=0.5, label='Equal scores')
        ax.set_xlabel('Score Difference (Sa - Sb)')
        ax.set_ylabel('Acceptance Probability')
        ax.set_title('Acceptance Probability vs Score Difference\n(Sa=accepted, Sb=rejected)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of score differences
        ax = axes[0, 1]
        ax.hist(X_data[y_data == 1], bins=30, alpha=0.5, label='Accepted', density=True)
        ax.hist(X_data[y_data == 0], bins=30, alpha=0.5, label='Rejected', density=True)
        ax.set_xlabel('Score Difference |Sa - Sb|')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Score Differences by Decision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Single decisions scatter
        ax = axes[1, 0]
        single_data = results['data']['single_decisions']
        if single_data:
            scores = [d['score'] for d in single_data]
            decisions = [d['decision'] for d in single_data]
            colors = ['green' if d == 1 else 'red' for d in decisions]
            ax.scatter(scores, decisions, c=colors, alpha=0.6)
            ax.axvline(results['threshold_S'], color='blue', linestyle='--', 
                      label=f'Threshold S1={results["threshold_S"]:.2f}')
            ax.axvline(results['threshold_S2'], color='red', linestyle='--', 
                      label=f'Threshold S2={results["threshold_S2"]:.2f}')
            ax.set_xlabel('Weighted Score')
            ax.set_ylabel('Decision (0=Reject, 1=Accept)')
            ax.set_title('Single Person Decisions')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Switch decisions scatter
        ax = axes[1, 1]
        switch_data = results['data']['switch_decisions']
        if switch_data:
            current_scores = [d['current_score'] for d in switch_data]
            new_scores = [d['new_score'] for d in switch_data]
            decisions = [d['decision'] for d in switch_data]
            colors = ['green' if d == 1 else 'red' for d in decisions]
            ax.scatter(current_scores, new_scores, c=colors, alpha=0.6)
            
            # Add diagonal line
            min_score = min(min(current_scores), min(new_scores))
            max_score = max(max(current_scores), max(new_scores))
            ax.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5)
            
            ax.set_xlabel('Current Partner Score')
            ax.set_ylabel('New Partner Score')
            ax.set_title('Switch Decisions (Green=Switch, Red=Stay)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.base_path}/acceptance_probability_analysis_plus.png', dpi=300)
        plt.show()
        
        return fig


def main():
    # Initialize analyzer
    base_path = "/home/lsy/match/bahavior_simul/0627_gpt4_eng"
    csv_prefix = "0618_gpt4_turbo_random_group"  # Updated to match actual filenames
    json_prefix = "0618_gpt4_turbo_random_group"
    
    analyzer = AcceptanceProbabilityAnalyzer(base_path, csv_prefix, json_prefix)
    
    # Load data from all groups
    groups = list(range(1, 22))  # Groups 1-21
    analyzer.load_all_data(groups)
    
    # Fit lambda parameter
    results = analyzer.fit_lambda_parameter()
    
    if results:
        # Print summary
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Estimated threshold S1 (max rejected): {results['threshold_S']:.3f}")
        print(f"Estimated threshold S2 (average of boundaries): {results['threshold_S2']:.3f}")
        print(f"Threshold used for fitting: {results['threshold_used_for_fitting']:.3f}")
        print(f"\nScore difference analysis:")
        print(f"  Cases where accepted > rejected: {results['positive_diff_count']} ({results['positive_diff_percentage']:.1f}%)")
        print(f"\nLambda parameters (for sigmoid function P = 1/(1+e^(-λ(Sa-Sb)))):")
        print(f"  Lambda (MLE method): {results['lambda_mle']:.4f}")
        print(f"  Lambda (MSE method): {results['lambda_mse']:.4f}")
        if results['lambda_curvefit']:
            print(f"  Lambda (Curve fit): {results['lambda_curvefit']:.4f}")
        
        # Visualize results
        analyzer.visualize_results(results)
        
        # Save results
        with open(f'{base_path}/lambda_analysis_results_plus.json', 'w') as f:
            save_results = {
                'threshold_S': results['threshold_S'],
                'threshold_S2': results['threshold_S2'],
                'threshold_used_for_fitting': results['threshold_used_for_fitting'],
                'positive_diff_count': results['positive_diff_count'],
                'positive_diff_percentage': results['positive_diff_percentage'],
                'lambda_mle': results['lambda_mle'],
                'lambda_mse': results['lambda_mse'],
                'lambda_curvefit': results['lambda_curvefit'] if results['lambda_curvefit'] else None,
                'total_decisions': len(analyzer.all_decisions),
                'single_decisions': len([d for d in analyzer.all_decisions if d['type'] == 'single']),
                'switch_decisions': len([d for d in analyzer.all_decisions if d['type'] == 'switch'])
            }
            json.dump(save_results, f, indent=2)
        
        print(f"\nResults saved to {base_path}/lambda_analysis_results_plus.json")
    else:
        print("\nERROR: Could not fit parameters due to insufficient data.")
    
    return results


if __name__ == "__main__":
    results = main()


# 1. 概率函数改为Sigmoid：
# pythonP = 1 / (1 + e^(-λ(Sa-Sb)))
# 这个函数的优点：

# 始终在[0,1]范围内
# 当Sa > Sb时，P > 0.5
# 当Sa = Sb时，P = 0.5
# 当Sa < Sb时，P < 0.5
# λ控制曲线的陡峭程度

# 2. λ参数限制为正值：

# 搜索范围：[0.01, 10]
# λ越大，决策越"理性"（对分数差异越敏感）
# λ越小，决策越"随机"（对分数差异不敏感）



# 1. 左上：接受概率 vs 分数差异
# 含义：

# X轴：分数差异 (Sa - Sb)，范围从-5到+7
# Y轴：接受概率（0-1）
# 蓝点：实际数据的接受率
# 橙线：拟合的sigmoid曲线，λ=0.010（几乎是平的）

# 关键发现：

# λ值极小（0.010）：说明GPT-4对分数差异不敏感
# 概率曲线几乎是水平的（约0.5），意味着决策基本随机
# 即使分数差异很大（如+5分），接受概率也只是略高于50%
# 这表明GPT-4的决策并非主要基于总分差异

# 2. 右上：分数差异分布
# 含义：

# 蓝色：被接受决策的分数差异分布
# 橙色：被拒绝决策的分数差异分布

# 关键发现：

# 两个分布高度重叠，都集中在0-2之间
# 这证实了决策与分数差异的弱相关性
# 即使Sa>Sb（正差异），也有很多被拒绝的情况

# 3. 左下：单身决策散点图
# 含义：

# 所有点都在y=0（红色）或y=1（绿色）线上
# S₁=8.40（蓝线）：最高拒绝分数
# S₂=7.20（红线）：决策边界中点

# 关键发现：

# 没有灰色地带：7.2-8.4之间没有数据点
# 低于7.2全部拒绝，高于8.4全部接受
# 这种"断崖式"分布很不自然，可能是数据采样问题

# 4. 右下：换伴侣决策散点图
# 含义：

# 对角线上方：新人分数更高
# 对角线下方：现任分数更高

# 关键发现：

# 大量红点（不换）在对角线上方，说明即使新人分数更高也不换
# 一些绿点（换）在对角线下方，说明有时选择分数更低的人
# 决策模式复杂，不是简单的"选高分"

# 总体结论：

# λ=0.010表明模型失效：

# sigmoid函数退化成了接近0.5的常数函数
# 说明分数差异不是决策的主要因素


# 可能的原因：

# GPT-4可能更关注某个特定维度而非总分
# 存在非线性决策规则
# 可能有阈值效应（如某维度必须>7）


# 模型改进建议：

# 分析各个维度的影响，而不只是总分
# 考虑使用决策树或更复杂的模型
# 研究是否存在"一票否决"的维度


# 数据质量问题：

# 单身决策的断崖式分布不自然
# 可能需要检查数据生成过程



# 这个结果表明，简单的基于总分差异的模型无法很好地解释GPT-4的约会决策行为。需要更深入的分析来理解真正的决策机制。