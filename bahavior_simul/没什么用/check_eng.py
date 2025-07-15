import pandas as pd
import numpy as np
import json
import csv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import re
from tqdm import tqdm
import os

class EnglishMatchingAnalyzer:
    def __init__(self, base_path="/home/lsy/match/bahavior_simul/0627_gpt4_eng/"):
        self.base_path = base_path
        self.output_path = base_path  # Save outputs to the same directory
        self.all_decisions = []
        self.single_decisions = []
        self.switch_decisions = []
        self.group_results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def parse_scores_from_prompt(self, prompt):
        """Extract scores from English prompt"""
        scores = {}
        
        # Extract individual scores
        patterns = {
            'attractive': r'attractiveness:\s*(\d+\.?\d*)/10',
            'sincere': r'sincerity:\s*(\d+\.?\d*)/10',
            'intelligence': r'intelligence:\s*(\d+\.?\d*)/10',
            'funny': r'being funny:\s*(\d+\.?\d*)/10',
            'ambition': r'ambition:\s*(\d+\.?\d*)/10',
            'shared_interests': r'shared interests:\s*(\d+\.?\d*)/10'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        
        # Extract importance weights
        weight_patterns = {
            'attractive_weight': r'attractiveness:\s*(\d+\.?\d*)[;.]',
            'sincere_weight': r'sincerity:\s*(\d+\.?\d*)[;.]',
            'intelligence_weight': r'intelligence:\s*(\d+\.?\d*)[;.]',
            'funny_weight': r'being funny:\s*(\d+\.?\d*)[;.]',
            'ambition_weight': r'ambition:\s*(\d+\.?\d*)[;.]',
            'shared_interests_weight': r'shared interests:\s*(\d+\.?\d*)'
        }
        
        # Look for weights after "importance weights" or similar phrases
        weight_section = prompt.split('importance weights')[1] if 'importance weights' in prompt.lower() else prompt
        
        for key, pattern in weight_patterns.items():
            match = re.search(pattern, weight_section, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        
        return scores
    
    def calculate_weighted_score(self, scores):
        """Calculate weighted total score"""
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
        """Parse single group CSV file"""
        csv_path = f"{self.base_path}0618_gpt4_turbo_random_group{group_num}.csv"
        decisions = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:
                        prompt = row[0]
                        response = row[1]
                        decision = int(row[5]) if row[5].isdigit() else 0
                        
                        accepted = decision == 1
                        
                        # Check if single or partnered
                        is_single = 'currently matched' not in prompt.lower() and 'current partner' not in prompt.lower()
                        
                        scores = self.parse_scores_from_prompt(prompt)
                        
                        if is_single:
                            proposer_score = self.calculate_weighted_score(scores)
                            
                            decisions.append({
                                'type': 'single',
                                'proposer_score': proposer_score,
                                'accepted': accepted,
                                'group': group_num
                            })
                            
                        else:
                            # Extract current partner scores
                            if 'new woman' in prompt.lower() or 'new man' in prompt.lower():
                                # Split to get current and new partner sections
                                parts = re.split(r'new woman|new man', prompt, flags=re.IGNORECASE)
                                if len(parts) >= 2:
                                    current_section = parts[0]
                                    new_section = parts[1]
                                else:
                                    current_section = prompt
                                    new_section = prompt
                            else:
                                current_section = prompt
                                new_section = prompt
                            
                            current_scores = self.parse_scores_from_prompt(current_section)
                            current_score = self.calculate_weighted_score(current_scores)
                            
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
        """Fit single acceptance threshold S"""
        if not single_decisions:
            return None
        
        accepted_scores = [d['proposer_score'] for d in single_decisions if d['accepted']]
        rejected_scores = [d['proposer_score'] for d in single_decisions if not d['accepted']]
        
        if not accepted_scores or not rejected_scores:
            all_scores = [d['proposer_score'] for d in single_decisions]
            return np.median(all_scores) if all_scores else 5.0
        
        min_accepted = min(accepted_scores) if accepted_scores else 0
        max_rejected = max(rejected_scores) if rejected_scores else 10
        
        if max_rejected < min_accepted:
            threshold = (max_rejected + min_accepted) / 2
        else:
            all_scores = sorted([d['proposer_score'] for d in single_decisions])
            best_threshold = 5.0
            best_accuracy = 0
            
            for i in range(len(all_scores) - 1):
                threshold = (all_scores[i] + all_scores[i+1]) / 2
                
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
    
    def fit_lambda_exponential(self, switch_decisions):
        """Fit λ parameter using exponential function"""
        if not switch_decisions:
            return None
        
        score_diffs = np.array([abs(d['score_diff']) for d in switch_decisions])
        decisions = np.array([1 if d['accepted'] else 0 for d in switch_decisions])
        
        def neg_log_likelihood(lambda_param):
            # P(accept) = min(1, exp(λ * |S_a - S_b|))
            probs = np.minimum(1.0, np.exp(lambda_param * score_diffs))
            probs = np.clip(probs, 1e-10, 1-1e-10)
            
            nll = -np.sum(decisions * np.log(probs) + (1 - decisions) * np.log(1 - probs))
            return nll
        
        result = minimize(neg_log_likelihood, x0=[0.5], bounds=[(-5, 5)], method='L-BFGS-B')
        lambda_opt = result.x[0]
        
        return lambda_opt
    
    def fit_lambda_sigmoid(self, switch_decisions):
        """Fit λ parameter using sigmoid function"""
        if not switch_decisions:
            return None
        
        score_diffs = np.array([abs(d['score_diff']) for d in switch_decisions])
        decisions = np.array([1 if d['accepted'] else 0 for d in switch_decisions])
        
        def neg_log_likelihood(lambda_param):
            # P(accept) = 1 / (1 + exp(-λ * |S_a - S_b|))
            z = lambda_param * score_diffs
            z = np.clip(z, -100, 100)
            probs = 1 / (1 + np.exp(-z))
            probs = np.clip(probs, 1e-10, 1-1e-10)
            
            nll = -np.sum(decisions * np.log(probs) + (1 - decisions) * np.log(1 - probs))
            return nll
        
        result = minimize(neg_log_likelihood, x0=[1.0], bounds=[(0.01, 10)], method='L-BFGS-B')
        lambda_opt = result.x[0]
        
        return lambda_opt
    
    def analyze_all_groups(self):
        """Analyze all 21 groups"""
        print("Analyzing English group data...")
        
        all_single = []
        all_switch = []
        
        for group_num in tqdm(range(1, 22)):
            decisions = self.parse_csv_group(group_num)
            
            if decisions:
                single = [d for d in decisions if d['type'] == 'single']
                switch = [d for d in decisions if d['type'] == 'switch']
                
                all_single.extend(single)
                all_switch.extend(switch)
                
                s_threshold = self.fit_single_threshold(single) if single else None
                lambda_exp = self.fit_lambda_exponential(switch) if switch else None
                lambda_sig = self.fit_lambda_sigmoid(switch) if switch else None
                
                self.group_results[group_num] = {
                    'n_single': len(single),
                    'n_switch': len(switch),
                    's_threshold': s_threshold,
                    'lambda_exponential': lambda_exp,
                    'lambda_sigmoid': lambda_sig,
                    'single_accept_rate': np.mean([d['accepted'] for d in single]) if single else None,
                    'switch_accept_rate': np.mean([d['accepted'] for d in switch]) if switch else None
                }
        
        # Overall analysis
        self.overall_s = self.fit_single_threshold(all_single)
        self.overall_lambda_exp = self.fit_lambda_exponential(all_switch)
        self.overall_lambda_sig = self.fit_lambda_sigmoid(all_switch)
        
        self.all_decisions = all_single + all_switch
        self.single_decisions = all_single
        self.switch_decisions = all_switch
    
    def plot_results(self):
        """Visualize results with English labels"""
        groups = sorted(self.group_results.keys())
        lambdas_exp = [self.group_results[g].get('lambda_exponential', None) for g in groups]
        lambdas_sig = [self.group_results[g].get('lambda_sigmoid', None) for g in groups]
        thresholds = [self.group_results[g].get('s_threshold', None) for g in groups]
        
        valid_lambdas_exp = [(g, l) for g, l in zip(groups, lambdas_exp) if l is not None]
        valid_lambdas_sig = [(g, l) for g, l in zip(groups, lambdas_sig) if l is not None]
        valid_thresholds = [(g, t) for g, t in zip(groups, thresholds) if t is not None]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Lambda exponential by group
        if valid_lambdas_exp:
            g, l = zip(*valid_lambdas_exp)
            axes[0, 0].bar(g, l, color='blue', alpha=0.7)
            axes[0, 0].axhline(y=self.overall_lambda_exp, color='red', linestyle='--',
                              label=f'Overall λ_exp={self.overall_lambda_exp:.3f}')
            axes[0, 0].set_xlabel('Group')
            axes[0, 0].set_ylabel('λ (Exponential)')
            axes[0, 0].set_title('Lambda Parameter by Group (Exponential)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Lambda sigmoid by group
        if valid_lambdas_sig:
            g, l = zip(*valid_lambdas_sig)
            axes[0, 1].bar(g, l, color='green', alpha=0.7)
            axes[0, 1].axhline(y=self.overall_lambda_sig, color='red', linestyle='--',
                              label=f'Overall λ_sig={self.overall_lambda_sig:.3f}')
            axes[0, 1].set_xlabel('Group')
            axes[0, 1].set_ylabel('λ (Sigmoid)')
            axes[0, 1].set_title('Lambda Parameter by Group (Sigmoid)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # S threshold by group
        if valid_thresholds:
            g, t = zip(*valid_thresholds)
            axes[0, 2].bar(g, t, color='orange', alpha=0.7)
            axes[0, 2].axhline(y=self.overall_s, color='red', linestyle='--',
                              label=f'Overall S={self.overall_s:.2f}')
            axes[0, 2].set_xlabel('Group')
            axes[0, 2].set_ylabel('Acceptance Threshold S')
            axes[0, 2].set_title('Single Acceptance Threshold by Group')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Acceptance probability vs score difference (both models)
        if self.switch_decisions:
            score_diffs = [abs(d['score_diff']) for d in self.switch_decisions]
            accepted = [d['accepted'] for d in self.switch_decisions]
            
            # Binned statistics
            max_diff = max(score_diffs) if score_diffs else 1
            bins = np.linspace(0, max_diff, 10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            accept_rates = []
            
            for i in range(len(bins)-1):
                mask = (np.array(score_diffs) >= bins[i]) & (np.array(score_diffs) < bins[i+1])
                if np.sum(mask) > 0:
                    accept_rates.append(np.mean([accepted[j] for j in range(len(accepted)) if mask[j]]))
                else:
                    accept_rates.append(None)
            
            # Plot actual data
            valid_points = [(c, r) for c, r in zip(bin_centers, accept_rates) if r is not None]
            if valid_points:
                c, r = zip(*valid_points)
                axes[1, 0].scatter(c, r, s=100, alpha=0.7, label='Actual Data', color='black')
            
            # Plot fitted curves
            x_fit = np.linspace(0, max_diff, 100)
            
            # Exponential model
            y_fit_exp = np.minimum(1.0, np.exp(self.overall_lambda_exp * x_fit))
            axes[1, 0].plot(x_fit, y_fit_exp, 'b-', linewidth=2, 
                           label=f'Exponential: exp({self.overall_lambda_exp:.3f} * |ΔS|)')
            
            # Sigmoid model
            y_fit_sig = 1 / (1 + np.exp(-self.overall_lambda_sig * x_fit))
            axes[1, 0].plot(x_fit, y_fit_sig, 'g--', linewidth=2,
                           label=f'Sigmoid: 1/(1+exp(-{self.overall_lambda_sig:.3f} * |ΔS|))')
            
            axes[1, 0].set_xlabel('Score Difference |S_a - S_b|')
            axes[1, 0].set_ylabel('Acceptance Probability')
            axes[1, 0].set_title('Switch Partner Acceptance Probability vs Score Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(-0.05, 1.05)
        
        # Single acceptance distribution
        if self.single_decisions:
            accepted_single = [d['proposer_score'] for d in self.single_decisions if d['accepted']]
            rejected_single = [d['proposer_score'] for d in self.single_decisions if not d['accepted']]
            
            if accepted_single:
                axes[1, 1].hist(accepted_single, bins=20, alpha=0.5, label='Accepted', color='green')
            if rejected_single:
                axes[1, 1].hist(rejected_single, bins=20, alpha=0.5, label='Rejected', color='red')
            
            axes[1, 1].axvline(x=self.overall_s, color='black', linestyle='--', linewidth=2,
                               label=f'Threshold S={self.overall_s:.2f}')
            axes[1, 1].set_xlabel('Proposer Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Single Accept/Reject Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Model comparison
        if self.switch_decisions:
            score_diffs = np.array([abs(d['score_diff']) for d in self.switch_decisions])
            decisions = np.array([d['accepted'] for d in self.switch_decisions])
            
            # Calculate AIC for both models
            # Exponential model
            probs_exp = np.minimum(1.0, np.exp(self.overall_lambda_exp * score_diffs))
            probs_exp = np.clip(probs_exp, 1e-10, 1-1e-10)
            ll_exp = np.sum(decisions * np.log(probs_exp) + (1 - decisions) * np.log(1 - probs_exp))
            aic_exp = -2 * ll_exp + 2 * 1  # 1 parameter
            
            # Sigmoid model
            z = self.overall_lambda_sig * score_diffs
            z = np.clip(z, -100, 100)
            probs_sig = 1 / (1 + np.exp(-z))
            probs_sig = np.clip(probs_sig, 1e-10, 1-1e-10)
            ll_sig = np.sum(decisions * np.log(probs_sig) + (1 - decisions) * np.log(1 - probs_sig))
            aic_sig = -2 * ll_sig + 2 * 1  # 1 parameter
            
            # Accuracy
            acc_exp = accuracy_score(decisions, probs_exp > 0.5)
            acc_sig = accuracy_score(decisions, probs_sig > 0.5)
            
            # Plot comparison
            comparison_data = {
                'Model': ['Exponential', 'Sigmoid'],
                'AIC': [aic_exp, aic_sig],
                'Accuracy': [acc_exp, acc_sig]
            }
            
            x_pos = np.arange(len(comparison_data['Model']))
            width = 0.35
            
            ax2 = axes[1, 2]
            ax3 = ax2.twinx()
            
            bars1 = ax2.bar(x_pos - width/2, comparison_data['AIC'], width, 
                            label='AIC (lower is better)', color='purple', alpha=0.7)
            bars2 = ax3.bar(x_pos + width/2, comparison_data['Accuracy'], width,
                            label='Accuracy', color='orange', alpha=0.7)
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('AIC', color='purple')
            ax3.set_ylabel('Accuracy', color='orange')
            ax2.set_title('Model Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(comparison_data['Model'])
            ax2.tick_params(axis='y', labelcolor='purple')
            ax3.tick_params(axis='y', labelcolor='orange')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars1, comparison_data['AIC']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom')
            
            for bar, val in zip(bars2, comparison_data['Accuracy']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'english_matching_analysis_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to: {output_file}")
    
    def evaluate_models(self):
        """Evaluate both exponential and sigmoid models"""
        results = {}
        
        # Single model evaluation
        if self.single_decisions:
            single_pred = [d['proposer_score'] >= self.overall_s for d in self.single_decisions]
            single_true = [d['accepted'] for d in self.single_decisions]
            results['single_accuracy'] = accuracy_score(single_true, single_pred)
        else:
            results['single_accuracy'] = None
        
        # Switch models evaluation
        if self.switch_decisions:
            # Exponential model
            switch_probs_exp = [min(1.0, np.exp(self.overall_lambda_exp * abs(d['score_diff']))) 
                               for d in self.switch_decisions]
            switch_pred_exp = [p > 0.5 for p in switch_probs_exp]
            switch_true = [d['accepted'] for d in self.switch_decisions]
            results['switch_accuracy_exp'] = accuracy_score(switch_true, switch_pred_exp)
            
            # Sigmoid model
            switch_probs_sig = [1/(1+np.exp(-self.overall_lambda_sig*abs(d['score_diff']))) 
                               for d in self.switch_decisions]
            switch_pred_sig = [p > 0.5 for p in switch_probs_sig]
            results['switch_accuracy_sig'] = accuracy_score(switch_true, switch_pred_sig)
        else:
            results['switch_accuracy_exp'] = None
            results['switch_accuracy_sig'] = None
        
        return results
    
    def save_results(self):
        """Save all results to JSON and CSV files"""
        # Prepare summary data
        summary = {
            'overall_parameters': {
                'single_threshold_S': self.overall_s,
                'lambda_exponential': self.overall_lambda_exp,
                'lambda_sigmoid': self.overall_lambda_sig,
                'total_samples': len(self.all_decisions),
                'single_samples': len(self.single_decisions),
                'switch_samples': len(self.switch_decisions)
            },
            'model_evaluation': self.evaluate_models(),
            'group_results': self.group_results
        }
        
        # Save to JSON
        json_file = os.path.join(self.output_path, 'english_matching_analysis_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"Results saved to: {json_file}")
        
        # Save group results to CSV
        results_df = pd.DataFrame(self.group_results).T
        csv_file = os.path.join(self.output_path, 'english_group_analysis_results.csv')
        results_df.to_csv(csv_file)
        print(f"Group results saved to: {csv_file}")
    
    def print_summary(self):
        """Print summary in English"""
        print("\n" + "="*60)
        print("English Group Matching Analysis Summary")
        print("="*60)
        
        print(f"\nTotal samples: {len(self.all_decisions)}")
        print(f"  - Single decisions: {len(self.single_decisions)}")
        print(f"  - Switch partner decisions: {len(self.switch_decisions)}")
        
        print(f"\nOverall Parameters:")
        print(f"  - Single acceptance threshold S = {self.overall_s:.3f}")
        print(f"  - Switch partner λ (exponential) = {self.overall_lambda_exp:.3f}")
        print(f"  - Switch partner λ (sigmoid) = {self.overall_lambda_sig:.3f}")
        
        eval_results = self.evaluate_models()
        print(f"\nModel Accuracy:")
        if eval_results['single_accuracy'] is not None:
            print(f"  - Single model: {eval_results['single_accuracy']:.1%}")
        if eval_results['switch_accuracy_exp'] is not None:
            print(f"  - Switch model (exponential): {eval_results['switch_accuracy_exp']:.1%}")
        if eval_results['switch_accuracy_sig'] is not None:
            print(f"  - Switch model (sigmoid): {eval_results['switch_accuracy_sig']:.1%}")
        
        print(f"\nAcceptance Probability Functions:")
        print(f"  - Single: P(accept) = 1 if Score >= {self.overall_s:.3f}, else 0")
        print(f"  - Switch (Exponential): P(accept) = min(1, exp({self.overall_lambda_exp:.3f} * |S_a - S_b|))")
        print(f"  - Switch (Sigmoid): P(accept) = 1 / (1 + exp(-{self.overall_lambda_sig:.3f} * |S_a - S_b|))")
        print(f"    where S_a is new partner score, S_b is current partner score")

def compare_languages(chinese_results_file, english_analyzer):
    """Compare Chinese and English results"""
    # Load Chinese results
    with open(chinese_results_file, 'r', encoding='utf-8') as f:
        chinese_results = json.load(f)
    
    print("\n" + "="*60)
    print("Language Comparison: Chinese vs English")
    print("="*60)
    
    # Compare single thresholds
    cn_s = chinese_results['overall_parameters']['single_threshold_S']
    en_s = english_analyzer.overall_s
    print(f"\nSingle Acceptance Threshold S:")
    print(f"  Chinese: {cn_s:.3f}")
    print(f"  English: {en_s:.3f}")
    print(f"  Difference: {en_s - cn_s:.3f} (English {'higher' if en_s > cn_s else 'lower'})")
    
    # Compare lambda parameters
    cn_lambda_exp = chinese_results['overall_parameters']['lambda_exponential']
    en_lambda_exp = english_analyzer.overall_lambda_exp
    cn_lambda_sig = chinese_results['overall_parameters']['lambda_sigmoid']
    en_lambda_sig = english_analyzer.overall_lambda_sig
    
    print(f"\nSwitch Partner λ (Exponential):")
    print(f"  Chinese: {cn_lambda_exp:.3f}")
    print(f"  English: {en_lambda_exp:.3f}")
    print(f"  Difference: {en_lambda_exp - cn_lambda_exp:.3f}")
    
    print(f"\nSwitch Partner λ (Sigmoid):")
    print(f"  Chinese: {cn_lambda_sig:.3f}")
    print(f"  English: {en_lambda_sig:.3f}")
    print(f"  Difference: {en_lambda_sig - cn_lambda_sig:.3f}")
    
    # Compare sample sizes
    print(f"\nSample Sizes:")
    print(f"  Chinese: {chinese_results['overall_parameters']['total_samples']} total")
    print(f"  English: {len(english_analyzer.all_decisions)} total")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Threshold comparison
    langs = ['Chinese', 'English']
    thresholds = [cn_s, en_s]
    axes[0].bar(langs, thresholds, color=['red', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Threshold S')
    axes[0].set_title('Single Acceptance Threshold Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # Lambda exponential comparison
    lambdas_exp = [cn_lambda_exp, en_lambda_exp]
    axes[1].bar(langs, lambdas_exp, color=['red', 'blue'], alpha=0.7)
    axes[1].set_ylabel('λ (Exponential)')
    axes[1].set_title('Exponential Lambda Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Lambda sigmoid comparison
    lambdas_sig = [cn_lambda_sig, en_lambda_sig]
    axes[2].bar(langs, lambdas_sig, color=['red', 'blue'], alpha=0.7)
    axes[2].set_ylabel('λ (Sigmoid)')
    axes[2].set_title('Sigmoid Lambda Comparison')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(english_analyzer.output_path, 'chinese_english_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nComparison plot saved to: {output_file}")

def main():
    """Main function"""
    analyzer = EnglishMatchingAnalyzer()
    
    # Analyze all groups
    analyzer.analyze_all_groups()
    
    # Visualize results
    analyzer.plot_results()
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    analyzer.print_summary()
    
    # Compare with Chinese results if available
    chinese_results_file = "/home/lsy/match/bahavior_simul/0629_gpt_Chinese/chinese_matching_analysis_results.json"
    if os.path.exists(chinese_results_file):
        compare_languages(chinese_results_file, analyzer)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()


# ============================================================
# English Group Matching Analysis Summary
# ============================================================

# Total samples: 1315
#   - Single decisions: 371
#   - Switch partner decisions: 944

# Overall Parameters:
#   - Single acceptance threshold S = 6.561
#   - Switch partner λ (exponential) = 0.500
#   - Switch partner λ (sigmoid) = 0.010

# Model Accuracy:
#   - Single model: 86.3%
#   - Switch model (exponential): 3.3%
#   - Switch model (sigmoid): 3.6%

# Acceptance Probability Functions:
#   - Single: P(accept) = 1 if Score >= 6.561, else 0
#   - Switch (Exponential): P(accept) = min(1, exp(0.500 * |S_a - S_b|))
#   - Switch (Sigmoid): P(accept) = 1 / (1 + exp(-0.010 * |S_a - S_b|))
#     where S_a is new partner score, S_b is current partner score

# ============================================================
# Language Comparison: Chinese vs English
# ============================================================

# Single Acceptance Threshold S:
#   Chinese: 4.314
#   English: 6.561
#   Difference: 2.247 (English higher)

# Switch Partner λ (Exponential):
#   Chinese: 0.500
#   English: 0.500
#   Difference: 0.000

# Switch Partner λ (Sigmoid):
#   Chinese: 0.050
#   English: 0.010
#   Difference: -0.040

# Sample Sizes:
#   Chinese: 1693 total
#   English: 1315 total