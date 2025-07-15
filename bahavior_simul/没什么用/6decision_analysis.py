import pandas as pd
import numpy as np
import json
import re
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MultiDimensionalAcceptanceAnalyzer:
    def __init__(self, base_path, csv_prefix):
        self.base_path = base_path
        self.csv_prefix = csv_prefix
        self.all_decisions = []
        self.dimension_names = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared_interests']
        
    def extract_all_scores_from_prompt(self, prompt):
        """Extract individual scores for all 6 dimensions from the prompt string"""
        pattern = r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
        
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            scores = np.array([float(match.group(i)) for i in range(1, 7)])
            return scores
        return None
    
    def extract_importance_weights(self, prompt):
        """Extract importance weights from the prompt string"""
        pattern = r'importance weights.*?attractiveness:\s*(\d+(?:\.\d+)?).*?sincerity:\s*(\d+(?:\.\d+)?).*?intelligence:\s*(\d+(?:\.\d+)?).*?being funny:\s*(\d+(?:\.\d+)?).*?ambition:\s*(\d+(?:\.\d+)?).*?shared interests:\s*(\d+(?:\.\d+)?)'
        
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            weights = np.array([float(match.group(i)) for i in range(1, 7)])
            return weights / weights.sum()  # Normalize
        return None
    
    def parse_csv_file(self, group_id):
        """Parse a single CSV file and extract decision data with all dimensions"""
        csv_path = f"{self.base_path}/{self.csv_prefix}{group_id}.csv"
        
        try:
            import os
            if not os.path.exists(csv_path):
                print(f"Warning: File not found: {csv_path}")
                return
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"Warning: Empty file: {csv_path}")
                return
            
            for line_num, line in enumerate(lines):
                if not line.strip():
                    continue
                
                import csv
                reader = csv.reader([line])
                try:
                    row = next(reader)
                except:
                    continue
                
                if len(row) < 6:
                    continue
                
                prompt = str(row[0]).strip()
                response = str(row[1]).strip()
                
                if not prompt or len(prompt) < 50:
                    continue
                
                # Extract decision
                if 'YES' in response.upper():
                    decision = 1
                elif 'NO' in response.upper():
                    decision = 0
                else:
                    continue
                
                # Extract scores and weights
                weights = self.extract_importance_weights(prompt)
                
                if weights is None:
                    continue
                
                # Check if it's a single person decision or switching decision
                if 'currently matched with a partner' in prompt:
                    # Switching decision - extract both score vectors
                    current_pattern = r"current partner's scores.*?" + r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
                    current_match = re.search(current_pattern, prompt, re.IGNORECASE | re.DOTALL)
                    
                    new_pattern = r"(?:new woman is courting you|new man is courting you) with scores:.*?" + r'attractiveness:\s*(\d+(?:\.\d+)?)/10.*?sincerity:\s*(\d+(?:\.\d+)?)/10.*?intelligence:\s*(\d+(?:\.\d+)?)/10.*?being funny:\s*(\d+(?:\.\d+)?)/10.*?ambition:\s*(\d+(?:\.\d+)?)/10.*?shared interests:\s*(\d+(?:\.\d+)?)/10'
                    new_match = re.search(new_pattern, prompt, re.IGNORECASE | re.DOTALL)
                    
                    if current_match and new_match:
                        current_scores = np.array([float(current_match.group(i)) for i in range(1, 7)])
                        new_scores = np.array([float(new_match.group(i)) for i in range(1, 7)])
                        
                        self.all_decisions.append({
                            'type': 'switch',
                            'current_scores': current_scores,
                            'new_scores': new_scores,
                            'weights': weights,
                            'decision': decision,
                            'group': group_id
                        })
                else:
                    # Single person decision
                    scores = self.extract_all_scores_from_prompt(prompt)
                    if scores is not None:
                        self.all_decisions.append({
                            'type': 'single',
                            'scores': scores,
                            'weights': weights,
                            'decision': decision,
                            'group': group_id
                        })
                        
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
    
    def fit_multidimensional_model(self):
        """Fit lambda vector for multi-dimensional model without absolute value"""
        if not self.all_decisions:
            print("ERROR: No decisions loaded!")
            return None
        
        # Separate single and switch decisions
        single_decisions = [d for d in self.all_decisions if d['type'] == 'single']
        switch_decisions = [d for d in self.all_decisions if d['type'] == 'switch']
        
        # First, estimate threshold S for each dimension (single decisions)
        S_estimates = self.estimate_thresholds(single_decisions)
        
        # Prepare data for fitting
        X_data = []  # Score differences for each dimension
        y_data = []  # Decisions
        weights_data = []  # Importance weights
        
        # Statistics
        positive_diff_counts = np.zeros(6)
        total_counts = 0
        
        # Add single decision data
        for d in single_decisions:
            if d['decision'] == 1:  # Accepted
                score_diff = d['scores'] - S_estimates
            else:  # Rejected
                score_diff = S_estimates - d['scores']
            
            X_data.append(score_diff)  # No absolute value
            y_data.append(d['decision'])
            weights_data.append(d['weights'])
            
            positive_diff_counts += (score_diff > 0)
            total_counts += 1
        
        # Add switch decision data
        for d in switch_decisions:
            if d['decision'] == 1:  # Switched
                score_diff = d['new_scores'] - d['current_scores']
            else:  # Stayed
                score_diff = d['current_scores'] - d['new_scores']
            
            X_data.append(score_diff)  # No absolute value
            y_data.append(d['decision'])
            weights_data.append(d['weights'])
            
            positive_diff_counts += (score_diff > 0)
            total_counts += 1
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        weights_data = np.array(weights_data)
        
        print(f"\nFitting with {len(X_data)} data points")
        print("\nPositive difference percentages by dimension:")
        for i, name in enumerate(self.dimension_names):
            print(f"  {name}: {positive_diff_counts[i]/total_counts*100:.1f}%")
        
        # Fit lambda vector using exponential model
        lambda_vector = self.fit_lambda_exp(X_data, y_data, weights_data)
        
        # Calculate fit statistics
        predictions = self.predict_probability_exp(X_data, lambda_vector, weights_data)
        predictions = np.minimum(predictions, 1.0)  # Cap at 1
        accuracy = np.mean((predictions > 0.5) == y_data)
        
        results = {
            'S_estimates': S_estimates,
            'lambda_vector': lambda_vector,
            'accuracy': accuracy,
            'positive_diff_percentages': (positive_diff_counts/total_counts*100).tolist(),
            'data': {
                'X': X_data,
                'y': y_data,
                'weights': weights_data,
                'single_decisions': single_decisions,
                'switch_decisions': switch_decisions
            }
        }
        
        return results
    
    def estimate_thresholds(self, single_decisions):
        """Estimate threshold S for each dimension"""
        S_estimates = np.zeros(6)
        
        for dim in range(6):
            rejected_scores = []
            accepted_scores = []
            
            for d in single_decisions:
                if d['decision'] == 0:
                    rejected_scores.append(d['scores'][dim])
                else:
                    accepted_scores.append(d['scores'][dim])
            
            if rejected_scores and accepted_scores:
                # Method 2: Average of boundaries
                S_estimates[dim] = (max(rejected_scores) + min(accepted_scores)) / 2
            elif rejected_scores:
                S_estimates[dim] = max(rejected_scores)
            elif accepted_scores:
                S_estimates[dim] = min(accepted_scores)
            else:
                S_estimates[dim] = 5.0  # Default
        
        print("\nEstimated thresholds S for each dimension:")
        for i, name in enumerate(self.dimension_names):
            print(f"  {name}: {S_estimates[i]:.2f}")
        
        return S_estimates
    
    def predict_probability_exp(self, X_diff, lambda_vector, weights):
        """Calculate probability using exponential function without absolute value"""
        # P = exp(sum(λi * Xi_diff * wi))
        weighted_sum = np.sum(X_diff * lambda_vector * weights, axis=1)
        return np.exp(weighted_sum)
    
    def fit_lambda_exp(self, X_data, y_data, weights_data):
        """Fit lambda vector using exponential model with MLE"""
        
        def negative_log_likelihood(lambda_vector):
            probs = self.predict_probability_exp(X_data, lambda_vector, weights_data)
            # Cap probabilities at 1 since exp can exceed 1
            probs = np.minimum(probs, 1.0)
            probs = np.clip(probs, 1e-10, 1-1e-10)
            ll = np.sum(y_data * np.log(probs) + (1 - y_data) * np.log(1 - probs))
            return -ll
        
        # Initial guess
        lambda0 = np.ones(6) * 0.1
        
        # Optimize with bounds allowing negative values
        result = minimize(negative_log_likelihood, lambda0, 
                         bounds=[(-5, 5) for _ in range(6)],
                         method='L-BFGS-B')
        
        lambda_vector = result.x
        
        print("\nFitted lambda vector (exponential model, no absolute value):")
        for i, name in enumerate(self.dimension_names):
            print(f"  λ_{name}: {lambda_vector[i]:.4f}")
        
        return lambda_vector
    
    def visualize_results(self, results):
        """Visualize the results with importance analysis"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Lambda values plot
        ax1 = fig.add_subplot(gs[0, :])
        lambda_vector = results['lambda_vector']
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        bars = ax1.bar(self.dimension_names, lambda_vector, color=colors)
        ax1.set_ylabel('Lambda value', fontsize=12)
        ax1.set_title('Lambda Parameters for Each Dimension (Higher = More Important)', fontsize=14)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, lambda_vector):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Threshold S values plot
        ax2 = fig.add_subplot(gs[1, :])
        S_estimates = results['S_estimates']
        bars2 = ax2.bar(self.dimension_names, S_estimates, color=colors, alpha=0.7)
        ax2.set_ylabel('Threshold S', fontsize=12)
        ax2.set_title('Estimated Thresholds for Each Dimension', fontsize=14)
        ax2.set_ylim(0, 10)
        
        # Add value labels
        for bar, val in zip(bars2, S_estimates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # Positive difference percentages
        ax3 = fig.add_subplot(gs[2, :])
        pos_diff_pct = results['positive_diff_percentages']
        bars3 = ax3.bar(self.dimension_names, pos_diff_pct, color=colors, alpha=0.5)
        ax3.set_ylabel('Percentage (%)', fontsize=12)
        ax3.set_title('Percentage of Cases Where Accepted > Rejected (by Dimension)', fontsize=14)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (random)')
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars3, pos_diff_pct):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        plt.suptitle(f'Multi-dimensional Analysis Results (Exponential, No Absolute Value)\nModel Accuracy: {results["accuracy"]:.3f}', 
                     fontsize=16)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.base_path}/multidim_analysis_exp_no_abs_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    # Initialize analyzer
    base_path = "/home/lsy/match/bahavior_simul/0627_gpt4_eng"
    csv_prefix = "0618_gpt4_turbo_random_group"
    
    analyzer = MultiDimensionalAcceptanceAnalyzer(base_path, csv_prefix)
    
    # Load data
    groups = list(range(1, 22))
    analyzer.load_all_data(groups)
    
    # Fit model
    results = analyzer.fit_multidimensional_model()
    
    if results:
        print(f"\nModel accuracy: {results['accuracy']:.3f}")
        
        # Visualize
        analyzer.visualize_results(results)
        
        # Save results
        save_results = {
            'S_estimates': results['S_estimates'].tolist(),
            'lambda_vector': results['lambda_vector'].tolist(),
            'dimension_names': analyzer.dimension_names,
            'accuracy': results['accuracy'],
            'positive_diff_percentages': results['positive_diff_percentages'],
            'model_type': 'exponential_no_absolute_value',
            'total_decisions': len(analyzer.all_decisions)
        }
        
        with open(f'{base_path}/multidim_lambda_exp_no_abs.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\nResults saved to {base_path}/multidim_lambda_exp_no_abs.json")
    
    return results


if __name__ == "__main__":
    results = main()