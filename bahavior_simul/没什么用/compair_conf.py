import pandas as pd
import numpy as np
import json
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class MatchingComparison:
    def __init__(self, base_path="/home/lsy/match/bahavior_simul/0627_gpt4_eng/"):
        self.base_path = base_path
        self.lambda_exp = 0.500  # From previous analysis
        self.lambda_sig = 0.010  # From previous analysis
        
    def load_gpt_matching(self, group_num):
        """Load GPT-generated matching from JSON"""
        json_path = f"{self.base_path}0618_gpt4_turbo_random_group{group_num}.json"
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def parse_scores_from_csv(self, group_num):
        """Parse scores from CSV"""
        csv_path = f"{self.base_path}0618_gpt4_turbo_random_group{group_num}.csv"
        scores_dict = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 4:
                        try:
                            target_id = str(row[2])
                            proposer_id = str(row[3])
                            
                            if not target_id or not proposer_id:
                                continue
                            
                            prompt = row[0]
                            score = self.extract_overall_score(prompt)
                            
                            if proposer_id not in scores_dict:
                                scores_dict[proposer_id] = {}
                            scores_dict[proposer_id][target_id] = score
                            
                        except:
                            continue
                            
        except Exception as e:
            print(f"Error parsing CSV for group {group_num}: {e}")
        
        return scores_dict
    
    def extract_overall_score(self, prompt):
        """Extract weighted score from prompt"""
        import re
        
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
        
        total_score = 0
        total_weight = 0
        
        for attr in scores:
            if attr in weights:
                total_score += scores[attr] * weights[attr]
                total_weight += weights[attr]
        
        if total_weight > 0:
            return total_score / total_weight
        return 5.0
    
    def identify_men_and_women(self, matching):
        """Identify men and women from matching"""
        men = set()
        women = set()
        
        for person, partner in matching.items():
            if partner != 'rejected' and partner is not None:
                try:
                    if int(person) % 2 == 1:
                        men.add(person)
                        women.add(partner)
                    else:
                        women.add(person)
                        men.add(partner)
                except:
                    men.add(person)
                    women.add(partner)
        
        return men, women
    
    def run_gale_shapley(self, men, women, scores_dict):
        """Run standard Gale-Shapley algorithm"""
        # Build preference lists
        men_prefs = {}
        for m in men:
            if m in scores_dict:
                women_scores = [(w, scores_dict[m].get(w, 0)) for w in women]
                women_scores.sort(key=lambda x: x[1], reverse=True)
                men_prefs[m] = [w for w, _ in women_scores]
            else:
                men_prefs[m] = list(women)
        
        women_prefs = {}
        for w in women:
            if w in scores_dict:
                men_scores = [(m, scores_dict[w].get(m, 0)) for m in men]
                men_scores.sort(key=lambda x: x[1], reverse=True)
                women_prefs[w] = [m for m, _ in men_scores]
            else:
                women_prefs[w] = list(men)
        
        # Run algorithm
        free_men = list(men)
        current_match = {}
        next_proposal = {m: 0 for m in men}
        
        while free_men:
            man = free_men[0]
            
            if next_proposal[man] >= len(men_prefs[man]):
                free_men.remove(man)
                continue
            
            woman = men_prefs[man][next_proposal[man]]
            next_proposal[man] += 1
            
            if woman not in current_match:
                current_match[woman] = man
                free_men.remove(man)
            else:
                current_partner = current_match[woman]
                
                if woman in women_prefs:
                    try:
                        current_rank = women_prefs[woman].index(current_partner)
                        new_rank = women_prefs[woman].index(man)
                        
                        if new_rank < current_rank:
                            current_match[woman] = man
                            free_men.remove(man)
                            free_men.append(current_partner)
                    except ValueError:
                        pass
        
        # Convert to bidirectional matching
        matching = {}
        for w, m in current_match.items():
            matching[m] = w
            matching[w] = m
        
        for person in men | women:
            if person not in matching:
                matching[person] = 'rejected'
        
        return matching
    
    def calculate_expected_blocking_pairs(self, matching, scores_dict, men, women, model='exponential'):
        """Calculate expected blocking pairs for a given matching"""
        expected_bp = 0
        blocking_details = []
        
        for m in men:
            for w in women:
                if matching.get(m) == w:
                    continue
                
                m_partner = matching.get(m)
                w_partner = matching.get(w)
                
                if m_partner == 'rejected' or w_partner == 'rejected':
                    continue
                
                try:
                    # Calculate score differences
                    score_m_current = scores_dict.get(m, {}).get(m_partner, 5.0)
                    score_m_w = scores_dict.get(m, {}).get(w, 5.0)
                    diff_m = score_m_w - score_m_current
                    
                    score_w_current = scores_dict.get(w, {}).get(w_partner, 5.0)
                    score_w_m = scores_dict.get(w, {}).get(m, 5.0)
                    diff_w = score_w_m - score_w_current
                    
                    # Calculate probabilities based on absolute differences
                    # P(switch) based on |diff|, regardless of sign
                    if model == 'exponential':
                        prob_m = min(1.0, np.exp(self.lambda_exp * abs(diff_m)))
                        prob_w = min(1.0, np.exp(self.lambda_exp * abs(diff_w)))
                    else:  # sigmoid
                        prob_m = 1 / (1 + np.exp(-self.lambda_sig * abs(diff_m)))
                        prob_w = 1 / (1 + np.exp(-self.lambda_sig * abs(diff_w)))
                    
                    p_bp = prob_m * prob_w
                    expected_bp += p_bp
                    
                    if p_bp > 0.01:
                        blocking_details.append({
                            'man': m,
                            'woman': w,
                            'diff_m': diff_m,
                            'diff_w': diff_w,
                            'p_blocking': p_bp
                        })
                
                except:
                    continue
        
        return expected_bp, blocking_details
    
    def compare_group(self, group_num, model='exponential'):
        """Compare GPT and Gale-Shapley for one group"""
        # Load GPT matching
        gpt_matching = self.load_gpt_matching(group_num)
        if not gpt_matching:
            return None
        
        # Parse scores
        scores_dict = self.parse_scores_from_csv(group_num)
        if not scores_dict:
            return None
        
        # Identify men and women
        men, women = self.identify_men_and_women(gpt_matching)
        
        # Run Gale-Shapley
        gs_matching = self.run_gale_shapley(men, women, scores_dict)
        
        # Calculate expected blocking pairs for both
        gpt_bp, gpt_details = self.calculate_expected_blocking_pairs(
            gpt_matching, scores_dict, men, women, model)
        gs_bp, gs_details = self.calculate_expected_blocking_pairs(
            gs_matching, scores_dict, men, women, model)
        
        # Count matched pairs
        gpt_matched = sum(1 for p in men if gpt_matching.get(p) != 'rejected')
        gs_matched = sum(1 for p in men if gs_matching.get(p) != 'rejected')
        
        return {
            'group': group_num,
            'gpt_expected_bp': gpt_bp,
            'gs_expected_bp': gs_bp,
            'gpt_matched_pairs': gpt_matched,
            'gs_matched_pairs': gs_matched,
            'bp_difference': gpt_bp - gs_bp,
            'improvement_ratio': (gpt_bp - gs_bp) / gpt_bp if gpt_bp > 0 else 0
        }
    
    def analyze_all_groups(self, model='exponential'):
        """Analyze all groups"""
        results = {}
        
        print(f"Comparing GPT vs Gale-Shapley matchings ({model} model)...")
        for group_num in tqdm(range(1, 22)):
            result = self.compare_group(group_num, model)
            if result:
                results[group_num] = result
        
        # Calculate overall statistics
        total_gpt_bp = sum(r['gpt_expected_bp'] for r in results.values())
        total_gs_bp = sum(r['gs_expected_bp'] for r in results.values())
        total_gpt_matched = sum(r['gpt_matched_pairs'] for r in results.values())
        total_gs_matched = sum(r['gs_matched_pairs'] for r in results.values())
        
        overall_stats = {
            'total_gpt_expected_bp': total_gpt_bp,
            'total_gs_expected_bp': total_gs_bp,
            'average_gpt_bp': total_gpt_bp / len(results),
            'average_gs_bp': total_gs_bp / len(results),
            'total_bp_reduction': total_gpt_bp - total_gs_bp,
            'reduction_percentage': ((total_gpt_bp - total_gs_bp) / total_gpt_bp * 100) if total_gpt_bp > 0 else 0,
            'gpt_stability_score': 1 - (total_gpt_bp / total_gpt_matched) if total_gpt_matched > 0 else 1,
            'gs_stability_score': 1 - (total_gs_bp / total_gs_matched) if total_gs_matched > 0 else 1
        }
        
        return results, overall_stats
    
    def plot_comparison(self, results, overall_stats):
        """Visualize comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        groups = sorted(results.keys())
        gpt_bp = [results[g]['gpt_expected_bp'] for g in groups]
        gs_bp = [results[g]['gs_expected_bp'] for g in groups]
        differences = [results[g]['bp_difference'] for g in groups]
        
        # 1. Side-by-side comparison
        x = np.arange(len(groups))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, gpt_bp, width, label='GPT Matching', color='red', alpha=0.7)
        axes[0, 0].bar(x + width/2, gs_bp, width, label='Gale-Shapley', color='green', alpha=0.7)
        axes[0, 0].set_xlabel('Group')
        axes[0, 0].set_ylabel('Expected Blocking Pairs')
        axes[0, 0].set_title('Expected Blocking Pairs: GPT vs Gale-Shapley')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(groups, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Difference plot
        colors = ['red' if d > 0 else 'green' for d in differences]
        axes[0, 1].bar(groups, differences, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].set_xlabel('Group')
        axes[0, 1].set_ylabel('BP Difference (GPT - GS)')
        axes[0, 1].set_title('Blocking Pair Difference by Group')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot: GPT vs GS
        if len(gs_bp) > 0 and len(gpt_bp) > 0 and not all(x == 0 for x in gs_bp + gpt_bp):
            axes[1, 0].scatter(gs_bp, gpt_bp, s=100, alpha=0.7, color='blue')
            
            # Add diagonal line
            max_bp = max(max(gpt_bp) if gpt_bp else 1, max(gs_bp) if gs_bp else 1)
            axes[1, 0].plot([0, max_bp], [0, max_bp], 'k--', alpha=0.5, label='Equal')
            
            # Add trend line only if there's variation
            if np.std(gs_bp) > 0 and np.std(gpt_bp) > 0:
                try:
                    z = np.polyfit(gs_bp, gpt_bp, 1)
                    p = np.poly1d(z)
                    axes[1, 0].plot(sorted(gs_bp), p(sorted(gs_bp)), "r-", alpha=0.8, 
                                   label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                except:
                    pass
            
            axes[1, 0].set_xlabel('Gale-Shapley Expected BP')
            axes[1, 0].set_ylabel('GPT Expected BP')
            axes[1, 0].set_title('GPT vs Gale-Shapley Expected Blocking Pairs')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No blocking pairs detected\nin either matching', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            axes[1, 0].set_xlabel('Gale-Shapley Expected BP')
            axes[1, 0].set_ylabel('GPT Expected BP')
            axes[1, 0].set_title('GPT vs Gale-Shapley Expected Blocking Pairs')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        summary_text = f"""Comparison Summary:
        
GPT Matching:
  Total Expected BP: {overall_stats['total_gpt_expected_bp']:.2f}
  Average per Group: {overall_stats['average_gpt_bp']:.2f}
  Stability Score: {overall_stats['gpt_stability_score']:.3f}

Gale-Shapley:
  Total Expected BP: {overall_stats['total_gs_expected_bp']:.2f}
  Average per Group: {overall_stats['average_gs_bp']:.2f}
  Stability Score: {overall_stats['gs_stability_score']:.3f}

Improvement:
  Total BP Reduction: {overall_stats['total_bp_reduction']:.2f}
  Reduction %: {overall_stats['reduction_percentage']:.1f}%

Winner: {'Gale-Shapley' if overall_stats['total_gs_expected_bp'] < overall_stats['total_gpt_expected_bp'] else 'GPT'}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(self.base_path, 'gpt_vs_gs_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison plot saved to: {output_file}")
    
    def compare_both_models(self):
        """Compare using both exponential and sigmoid models"""
        # Exponential model
        exp_results, exp_stats = self.analyze_all_groups('exponential')
        
        # Sigmoid model
        sig_results, sig_stats = self.analyze_all_groups('sigmoid')
        
        print("\n" + "="*60)
        print("GPT vs Gale-Shapley Comparison Results")
        print("="*60)
        
        print("\n--- Exponential Model (λ = 0.500) ---")
        print(f"GPT Total Expected BP: {exp_stats['total_gpt_expected_bp']:.2f}")
        print(f"G-S Total Expected BP: {exp_stats['total_gs_expected_bp']:.2f}")
        print(f"Reduction: {exp_stats['reduction_percentage']:.1f}%")
        print(f"Winner: {'Gale-Shapley' if exp_stats['total_gs_expected_bp'] < exp_stats['total_gpt_expected_bp'] else 'GPT'}")
        
        print("\n--- Sigmoid Model (λ = 0.010) ---")
        print(f"GPT Total Expected BP: {sig_stats['total_gpt_expected_bp']:.2f}")
        print(f"G-S Total Expected BP: {sig_stats['total_gs_expected_bp']:.2f}")
        print(f"Reduction: {sig_stats['reduction_percentage']:.1f}%")
        print(f"Winner: {'Gale-Shapley' if sig_stats['total_gs_expected_bp'] < sig_stats['total_gpt_expected_bp'] else 'GPT'}")
        
        # Plot for exponential model
        self.plot_comparison(exp_results, exp_stats)
        
        # Save detailed results
        output = {
            'exponential_model': {
                'overall_stats': exp_stats,
                'group_results': exp_results
            },
            'sigmoid_model': {
                'overall_stats': sig_stats,
                'group_results': sig_results
            },
            'parameters': {
                'lambda_exponential': self.lambda_exp,
                'lambda_sigmoid': self.lambda_sig
            }
        }
        
        json_file = os.path.join(self.base_path, 'gpt_vs_gs_comparison_results.json')
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {json_file}")
        
        return exp_results, sig_results, exp_stats, sig_stats

def main():
    """Run complete comparison"""
    comparator = MatchingComparison()
    
    # Run comparison with both models
    exp_results, sig_results, exp_stats, sig_stats = comparator.compare_both_models()
    
    # Create summary table
    print("\n" + "="*60)
    print("Summary Table: Expected Blocking Pairs")
    print("="*60)
    print(f"{'Model':<15} {'GPT':<15} {'Gale-Shapley':<15} {'Difference':<15} {'Winner'}")
    print("-"*60)
    print(f"{'Exponential':<15} {exp_stats['total_gpt_expected_bp']:<15.2f} {exp_stats['total_gs_expected_bp']:<15.2f} "
          f"{exp_stats['total_bp_reduction']:<15.2f} {'G-S' if exp_stats['total_gs_expected_bp'] < exp_stats['total_gpt_expected_bp'] else 'GPT'}")
    print(f"{'Sigmoid':<15} {sig_stats['total_gpt_expected_bp']:<15.2f} {sig_stats['total_gs_expected_bp']:<15.2f} "
          f"{sig_stats['total_bp_reduction']:<15.2f} {'G-S' if sig_stats['total_gs_expected_bp'] < sig_stats['total_gpt_expected_bp'] else 'GPT'}")
    
    return comparator

if __name__ == "__main__":
    comparator = main()