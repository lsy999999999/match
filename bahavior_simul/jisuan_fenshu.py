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
        
        # Add single decision data
        for d in single_decisions:
            score_diff = abs(d['score'] - threshold_for_fitting)
            X_data.append(score_diff)
            y_data.append(d['decision'])
        
        # Add switch decision data
        for d in switch_decisions:
            score_diff = abs(d['new_score'] - d['current_score'])
            X_data.append(score_diff)
            y_data.append(d['decision'])
        
        if not X_data:
            print("ERROR: No valid data for fitting!")
            return None
            
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Fitting with {len(X_data)} data points")
        
        # Define the probability function
        def prob_function(x, lambda_param):
            return np.exp(-lambda_param * x)
        
        # Fit using different methods
        print("\nFitting lambda parameter...")
        
        # Method 1: Direct curve fitting
        try:
            popt, pcov = curve_fit(prob_function, X_data, y_data, p0=[1.0], bounds=(0, 10))
            lambda_fit = popt[0]
            print(f"Curve fit lambda: {lambda_fit:.4f}")
        except:
            lambda_fit = None
            print("Curve fitting failed")
        
        # Method 2: Maximum likelihood estimation
        def negative_log_likelihood(lambda_param):
            if lambda_param <= 0:
                return np.inf
            
            probs = prob_function(X_data, lambda_param)
            # Clip probabilities to avoid log(0)
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
        
        # Plot 1: Acceptance rate vs score difference
        ax = axes[0, 0]
        X_data = results['data']['X']
        y_data = results['data']['y']
        
        # Bin the data
        bins = np.linspace(0, X_data.max(), 20)
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
        x_range = np.linspace(0, X_data.max(), 100)
        for lambda_name, lambda_val in [('MLE', results['lambda_mle']), 
                                       ('MSE', results['lambda_mse'])]:
            if lambda_val:
                y_pred = np.exp(-lambda_val * x_range)
                ax.plot(x_range, y_pred, label=f'{lambda_name}: λ={lambda_val:.3f}')
        
        ax.set_xlabel('Score Difference |Sa - Sb|')
        ax.set_ylabel('Acceptance Probability')
        ax.set_title('Acceptance Probability vs Score Difference')
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
        plt.savefig(f'{self.base_path}/acceptance_probability_analysis.png', dpi=300)
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
        print(f"Lambda (MLE method): {results['lambda_mle']:.4f}")
        print(f"Lambda (MSE method): {results['lambda_mse']:.4f}")
        if results['lambda_curvefit']:
            print(f"Lambda (Curve fit): {results['lambda_curvefit']:.4f}")
        
        # Visualize results
        analyzer.visualize_results(results)
        
        # Save results
        with open(f'{base_path}/lambda_analysis_results.json', 'w') as f:
            save_results = {
                'threshold_S': results['threshold_S'],
                'threshold_S2': results['threshold_S2'],
                'threshold_used_for_fitting': results['threshold_used_for_fitting'],
                'lambda_mle': results['lambda_mle'],
                'lambda_mse': results['lambda_mse'],
                'lambda_curvefit': results['lambda_curvefit'] if results['lambda_curvefit'] else None,
                'total_decisions': len(analyzer.all_decisions),
                'single_decisions': len([d for d in analyzer.all_decisions if d['type'] == 'single']),
                'switch_decisions': len([d for d in analyzer.all_decisions if d['type'] == 'switch'])
            }
            json.dump(save_results, f, indent=2)
        
        print(f"\nResults saved to {base_path}/lambda_analysis_results.json")
    else:
        print("\nERROR: Could not fit parameters due to insufficient data.")
    
    return results


if __name__ == "__main__":
    results = main()


#  threshold_S （阈值 S）

# 含义：单身时能接受的最低加权分数
# 计算方法：
# python# 如果有人拒绝了某些追求者
# threshold_S = max(所有被拒绝者的加权分数)

# # 如果没有人被拒绝
# threshold_S = min(所有被接受者的加权分数)
# S_{2}是拒绝的最高分,和接受的最低分的平均数。
# 例子：如果某人拒绝了分数为4.8和5.1的追求者，但接受了5.3的，那么 S ≈ 5.1

# 2. lambda_mle （最大似然估计的λ）

# 含义：使观测数据出现概率最大的λ值
# 计算方法：
# python# 对于每个决策，计算其在给定λ下的概率
# P(接受) = e^(-λ × |分数差|)

# # 最大化对数似然函数
# L(λ) = Σ[yi × log(Pi) + (1-yi) × log(1-Pi)]
# # 其中 yi 是实际决策(0或1)，Pi 是预测概率


# 3. lambda_mse （均方误差最小的λ）

# 使用的是用S_2估计

# 含义：使预测概率与实际决策差异最小的λ值
# 计算方法：
# python# 最小化均方误差
# MSE(λ) = 1/n × Σ(实际决策 - 预测概率)²

# # 预测概率 = e^(-λ × |分数差|)


# 4. lambda_curvefit （曲线拟合的λ）

# 含义：直接拟合指数函数得到的λ值
# 计算方法：使用 scipy 的 curve_fit 直接拟合 函数y = e^(-λx)

# 5. total_decisions （总决策数）

# 含义：所有组中提取的有效决策总数
# 计算：单身决策数 + 换伴侣决策数

# 6. single_decisions （单身决策数）

# 含义：单身状态下是否接受追求者的决策数
# 识别方法：提示词中不包含 “currently matched with a partner”

# 7. switch_decisions （换伴侣决策数）

# 含义：已有伴侣时是否换人的决策数
# 识别方法：提示词中包含 “currently matched with a partner”