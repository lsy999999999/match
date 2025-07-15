import pandas as pd
import numpy as np
import json
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class StabilityAnalyzer:
    def __init__(self, base_path="/home/lsy/match/bahavior_simul/0627_gpt4_eng/"):
        self.base_path = base_path
        self.lambda_exp = 0.500  # From previous analysis
        self.lambda_sig = 0.010  # From previous analysis
        self.all_blocking_pairs = []
        self.group_results = {}
        
    def load_matching_results(self, group_num):
        """Load matching results from JSON file"""
        json_path = f"{self.base_path}0618_gpt4_turbo_random_group{group_num}.json"
        try:
            with open(json_path, 'r') as f:
                matching = json.load(f)
            return matching
        except Exception as e:
            print(f"Error loading group {group_num}: {e}")
            return None
    
    def parse_scores_from_csv(self, group_num):
        """Parse scores from CSV to build preference matrix"""
        csv_path = f"{self.base_path}0618_gpt4_turbo_random_group{group_num}.csv"
        scores_dict = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 4:
                        # Extract proposer and target IDs
                        try:
                            target_id = str(row[2])
                            proposer_id = str(row[3])
                            
                            # Skip if IDs are empty
                            if not target_id or not proposer_id:
                                continue
                            
                            # Parse prompt to get score
                            prompt = row[0]
                            score = self.extract_overall_score(prompt)
                            
                            # Store score: scores_dict[evaluator][evaluated] = score
                            if proposer_id not in scores_dict:
                                scores_dict[proposer_id] = {}
                            scores_dict[proposer_id][target_id] = score
                            
                        except Exception as e:
                            continue
                            
        except Exception as e:
            print(f"Error parsing CSV for group {group_num}: {e}")
        
        return scores_dict
    
    def extract_overall_score(self, prompt):
        """Extract weighted score from prompt"""
        import re
        
        # Extract scores
        scores = {}
        patterns = {
            'attractiveness': r'attractiveness:\s*(\d+\.?\d*)/10',
            'sincerity': r'sincerity:\s*(\d+\.?\d*)/10',
            'intelligence': r'intelligence:\s*(\d+\.?\d*)/10',
            'funny': r'being funny:\s*(\d+\.?\d*)/10',
            'ambition': r'ambition:\s*(\d+\.?\d*)/10',
            'shared_interests': r'shared interests:\s*(\d+\.?\d*)/10'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        
        # Extract weights
        weights = {}
        weight_patterns = {
            'attractiveness': r'attractiveness:\s*(\d+\.?\d*)[;.]',
            'sincerity': r'sincerity:\s*(\d+\.?\d*)[;.]',
            'intelligence': r'intelligence:\s*(\d+\.?\d*)[;.]',
            'funny': r'being funny:\s*(\d+\.?\d*)[;.]',
            'ambition': r'ambition:\s*(\d+\.?\d*)[;.]',
            'shared_interests': r'shared interests:\s*(\d+\.?\d*)'
        }
        
        weight_section = prompt.split('importance weights')[1] if 'importance weights' in prompt.lower() else prompt
        
        for key, pattern in weight_patterns.items():
            match = re.search(pattern, weight_section, re.IGNORECASE)
            if match:
                weights[key] = float(match.group(1))
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for attr in scores:
            if attr in weights:
                total_score += scores[attr] * weights[attr]
                total_weight += weights[attr]
        
        if total_weight > 0:
            return total_score / total_weight
        return 5.0  # Default score
    
    def calculate_switch_probability(self, score_diff, model='exponential'):
        """Calculate probability of switching based on score difference"""
        abs_diff = abs(score_diff)
        
        if model == 'exponential':
            # P(accept) = min(1, exp(λ * |S_a - S_b|))
            prob = min(1.0, np.exp(self.lambda_exp * abs_diff))
        else:  # sigmoid
            # P(accept) = 1 / (1 + exp(-λ * |S_a - S_b|))
            prob = 1 / (1 + np.exp(-self.lambda_sig * abs_diff))
        
        return prob
    
    def analyze_group_stability(self, group_num, model='exponential'):
        """Analyze stability for a single group"""
        # Load matching results
        matching = self.load_matching_results(group_num)
        if not matching:
            return None
        
        # Parse scores
        scores_dict = self.parse_scores_from_csv(group_num)
        if not scores_dict:
            return None
        
        # Separate men and women based on matching
        men = set()
        women = set()
        matched_pairs = []
        
        for person, partner in matching.items():
            if partner != 'rejected' and partner is not None:
                # Determine gender based on ID patterns or matching structure
                # Assuming odd IDs are men, even IDs are women (adjust if needed)
                try:
                    if int(person) % 2 == 1:
                        men.add(person)
                        if partner not in ['rejected', None]:
                            women.add(partner)
                            matched_pairs.append((person, partner))
                    else:
                        women.add(person)
                        if partner not in ['rejected', None]:
                            men.add(partner)
                            matched_pairs.append((partner, person))
                except:
                    # If ID is not numeric, use another method
                    men.add(person)
                    if partner not in ['rejected', None]:
                        women.add(partner)
                        matched_pairs.append((person, partner))
        
        # Create reverse matching for quick lookup
        partner_of = {}
        for m, w in matched_pairs:
            partner_of[m] = w
            partner_of[w] = m
        
        # Calculate expected number of blocking pairs
        blocking_pair_probs = []
        
        for m in men:
            for w in women:
                # Skip if they are already matched
                if partner_of.get(m) == w:
                    continue
                
                # Get current partners
                m_partner = partner_of.get(m)
                w_partner = partner_of.get(w)
                
                # Skip if either is unmatched
                if not m_partner or not w_partner:
                    continue
                
                # Calculate score differences
                # Diff(m_p, w): m's preference between current partner and w
                try:
                    score_m_current = scores_dict.get(m, {}).get(m_partner, 5.0)
                    score_m_w = scores_dict.get(m, {}).get(w, 5.0)
                    diff_m = score_m_w - score_m_current  # Positive if w is better
                    
                    # Diff(w_p, m): w's preference between current partner and m
                    score_w_current = scores_dict.get(w, {}).get(w_partner, 5.0)
                    score_w_m = scores_dict.get(w, {}).get(m, 5.0)
                    diff_w = score_w_m - score_w_current  # Positive if m is better
                    
                    # Calculate probabilities
                    prob_m_switch = self.calculate_switch_probability(diff_m, model)
                    prob_w_switch = self.calculate_switch_probability(diff_w, model)
                    
                    # Probability of being a blocking pair
                    p_blocking = prob_m_switch * prob_w_switch
                    
                    blocking_pair_probs.append({
                        'man': m,
                        'woman': w,
                        'diff_m': diff_m,
                        'diff_w': diff_w,
                        'prob_m': prob_m_switch,
                        'prob_w': prob_w_switch,
                        'p_blocking': p_blocking
                    })
                    
                except Exception as e:
                    continue
        
        # Calculate expected number of blocking pairs
        expected_blocking_pairs = sum([bp['p_blocking'] for bp in blocking_pair_probs])
        
        return {
            'group': group_num,
            'num_men': len(men),
            'num_women': len(women),
            'num_matched_pairs': len(matched_pairs),
            'num_potential_blocking_pairs': len(blocking_pair_probs),
            'expected_blocking_pairs': expected_blocking_pairs,
            'blocking_pair_details': blocking_pair_probs
        }
    
    def analyze_all_groups(self, model='exponential'):
        """Analyze stability for all 21 groups"""
        print(f"Analyzing stability using {model} model...")
        
        all_blocking_pairs = []
        
        for group_num in tqdm(range(1, 22)):
            result = self.analyze_group_stability(group_num, model)
            if result:
                self.group_results[group_num] = result
                all_blocking_pairs.extend(result['blocking_pair_details'])
        
        # Calculate overall statistics
        total_expected = sum([r['expected_blocking_pairs'] for r in self.group_results.values()])
        total_pairs = sum([r['num_matched_pairs'] for r in self.group_results.values()])
        
        self.overall_stats = {
            'total_expected_blocking_pairs': total_expected,
            'total_matched_pairs': total_pairs,
            'average_blocking_pairs_per_group': total_expected / len(self.group_results) if self.group_results else 0,
            'stability_ratio': 1 - (total_expected / total_pairs) if total_pairs > 0 else 0
        }
        
        return self.overall_stats
    
    def plot_stability_analysis(self):
        """Visualize stability analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Expected blocking pairs by group
        groups = sorted(self.group_results.keys())
        expected_bp = [self.group_results[g]['expected_blocking_pairs'] for g in groups]
        
        axes[0, 0].bar(groups, expected_bp, color='red', alpha=0.7)
        axes[0, 0].axhline(y=np.mean(expected_bp), color='black', linestyle='--', 
                          label=f'Average: {np.mean(expected_bp):.2f}')
        axes[0, 0].set_xlabel('Group')
        axes[0, 0].set_ylabel('Expected Blocking Pairs')
        axes[0, 0].set_title('Expected Blocking Pairs by Group')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of blocking pair probabilities
        all_probs = []
        for result in self.group_results.values():
            all_probs.extend([bp['p_blocking'] for bp in result['blocking_pair_details']])
        
        axes[0, 1].hist(all_probs, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Blocking Pair Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Blocking Pair Probabilities')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score differences distribution
        all_diffs_m = []
        all_diffs_w = []
        for result in self.group_results.values():
            for bp in result['blocking_pair_details']:
                if bp['p_blocking'] > 0.01:  # Only significant blocking pairs
                    all_diffs_m.append(bp['diff_m'])
                    all_diffs_w.append(bp['diff_w'])
        
        axes[1, 0].hist2d(all_diffs_m, all_diffs_w, bins=30, cmap='YlOrRd')
        axes[1, 0].set_xlabel('Man Score Difference (new - current)')
        axes[1, 0].set_ylabel('Woman Score Difference (new - current)')
        axes[1, 0].set_title('Score Differences for Potential Blocking Pairs')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Stability metrics summary
        matched_pairs = [self.group_results[g]['num_matched_pairs'] for g in groups]
        stability_scores = [1 - (expected_bp[i] / matched_pairs[i]) if matched_pairs[i] > 0 else 1 
                           for i in range(len(groups))]
        
        axes[1, 1].scatter(matched_pairs, stability_scores, s=100, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Number of Matched Pairs')
        axes[1, 1].set_ylabel('Stability Score (1 - E[BP]/Matches)')
        axes[1, 1].set_title('Stability Score vs Group Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall stability score
        axes[1, 1].axhline(y=self.overall_stats['stability_ratio'], color='red', 
                          linestyle='--', linewidth=2,
                          label=f'Overall Stability: {self.overall_stats["stability_ratio"]:.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_file = os.path.join(self.base_path, 'stability_analysis_english.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Stability analysis plot saved to: {output_file}")
    
    def compare_models(self):
        """Compare stability under exponential and sigmoid models"""
        # Analyze with exponential model
        exp_stats = self.analyze_all_groups(model='exponential')
        exp_results = self.group_results.copy()
        
        # Reset and analyze with sigmoid model
        self.group_results = {}
        sig_stats = self.analyze_all_groups(model='sigmoid')
        sig_results = self.group_results.copy()
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compare expected blocking pairs
        groups = sorted(exp_results.keys())
        exp_bp = [exp_results[g]['expected_blocking_pairs'] for g in groups]
        sig_bp = [sig_results[g]['expected_blocking_pairs'] for g in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        axes[0].bar(x - width/2, exp_bp, width, label='Exponential', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, sig_bp, width, label='Sigmoid', color='green', alpha=0.7)
        axes[0].set_xlabel('Group')
        axes[0].set_ylabel('Expected Blocking Pairs')
        axes[0].set_title('Expected Blocking Pairs: Model Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(groups, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_comparison = {
            'Metric': ['Total Expected BP', 'Average BP/Group', 'Overall Stability'],
            'Exponential': [exp_stats['total_expected_blocking_pairs'],
                           exp_stats['average_blocking_pairs_per_group'],
                           exp_stats['stability_ratio']],
            'Sigmoid': [sig_stats['total_expected_blocking_pairs'],
                       sig_stats['average_blocking_pairs_per_group'],
                       sig_stats['stability_ratio']]
        }
        
        # Create bar chart for summary
        metrics = stats_comparison['Metric']
        exp_values = stats_comparison['Exponential']
        sig_values = stats_comparison['Sigmoid']
        
        x = np.arange(len(metrics))
        axes[1].bar(x - width/2, exp_values[:2] + [exp_values[2]*100], width, 
                   label='Exponential', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, sig_values[:2] + [sig_values[2]*100], width, 
                   label='Sigmoid', color='green', alpha=0.7)
        axes[1].set_xlabel('Metric')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Stability Metrics Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Total E[BP]', 'Avg BP/Group', 'Stability (%)'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.base_path, 'model_comparison_stability.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        return exp_stats, sig_stats
    
    def save_results(self):
        """Save detailed results to JSON"""
        output = {
            'overall_statistics': self.overall_stats,
            'group_summaries': {
                g: {
                    'group': r['group'],
                    'num_men': r['num_men'],
                    'num_women': r['num_women'],
                    'num_matched_pairs': r['num_matched_pairs'],
                    'expected_blocking_pairs': r['expected_blocking_pairs'],
                    'stability_score': 1 - (r['expected_blocking_pairs'] / r['num_matched_pairs']) 
                                     if r['num_matched_pairs'] > 0 else 1
                }
                for g, r in self.group_results.items()
            }
        }
        
        json_file = os.path.join(self.base_path, 'stability_analysis_results.json')
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {json_file}")

def main():
    """Run complete stability analysis"""
    analyzer = StabilityAnalyzer()
    
    # Compare both models
    print("Comparing stability under different models...")
    exp_stats, sig_stats = analyzer.compare_models()
    
    print("\n" + "="*60)
    print("Stability Analysis Results - English Groups")
    print("="*60)
    
    print("\nExponential Model:")
    print(f"  Total Expected Blocking Pairs: {exp_stats['total_expected_blocking_pairs']:.2f}")
    print(f"  Average per Group: {exp_stats['average_blocking_pairs_per_group']:.2f}")
    print(f"  Overall Stability Score: {exp_stats['stability_ratio']:.3f}")
    
    print("\nSigmoid Model:")
    print(f"  Total Expected Blocking Pairs: {sig_stats['total_expected_blocking_pairs']:.2f}")
    print(f"  Average per Group: {sig_stats['average_blocking_pairs_per_group']:.2f}")
    print(f"  Overall Stability Score: {sig_stats['stability_ratio']:.3f}")
    
    # Plot results
    analyzer.plot_stability_analysis()
    
    # Save results
    analyzer.save_results()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()



# ============================================================
# Stability Analysis Results - English Groups
# ============================================================

# Exponential Model:
#   Total Expected Blocking Pairs: 8586.00
#   Average per Group: 408.86
#   Overall Stability Score: -20.358

# Sigmoid Model:
#   Total Expected Blocking Pairs: 2147.42
#   Average per Group: 102.26
#   Overall Stability Score: -4.342