import pandas as pd
import numpy as np
import json
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class GaleShapleyStabilityAnalyzer:
    def __init__(self, base_path="/home/lsy/match/bahavior_simul/0627_gpt4_eng/"):
        self.base_path = base_path
        self.lambda_exp = 0.500  # From previous analysis
        self.lambda_sig = 0.010  # From previous analysis
        self.group_results = {}
        
    def parse_scores_from_csv(self, group_num):
        """Parse scores from CSV to build preference matrix"""
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
                            
                        except Exception as e:
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
    
    def identify_men_and_women(self, scores_dict):
        """Identify men and women from the scores dictionary"""
        # Use the original matching to identify genders
        json_path = f"{self.base_path}0618_gpt4_turbo_random_group1.json"
        try:
            with open(json_path, 'r') as f:
                original_matching = json.load(f)
        except:
            original_matching = {}
        
        men = set()
        women = set()
        
        # First identify from successful matches in original
        for person, partner in original_matching.items():
            if partner != 'rejected' and partner is not None:
                # Use pattern: typically men have odd IDs, women have even IDs
                try:
                    if int(person) % 2 == 1:
                        men.add(person)
                        women.add(partner)
                    else:
                        women.add(person)
                        men.add(partner)
                except:
                    # If not numeric, use first appearance as men
                    men.add(person)
                    women.add(partner)
        
        # Add any remaining people based on who they evaluated
        all_people = set(scores_dict.keys()) | set([p for scores in scores_dict.values() for p in scores.keys()])
        for person in all_people:
            if person not in men and person not in women:
                # Default: assign based on ID pattern or arbitrarily
                try:
                    if int(person) % 2 == 1:
                        men.add(person)
                    else:
                        women.add(person)
                except:
                    men.add(person)
        
        return men, women
    
    def run_gale_shapley(self, men, women, scores_dict):
        """Run traditional Gale-Shapley algorithm"""
        # Build preference lists
        men_prefs = {}
        for m in men:
            if m in scores_dict:
                # Sort women by score (highest first)
                women_scores = [(w, scores_dict[m].get(w, 0)) for w in women]
                women_scores.sort(key=lambda x: x[1], reverse=True)
                men_prefs[m] = [w for w, _ in women_scores]
            else:
                men_prefs[m] = list(women)
        
        women_prefs = {}
        for w in women:
            if w in scores_dict:
                # Sort men by score (highest first)
                men_scores = [(m, scores_dict[w].get(m, 0)) for m in men]
                men_scores.sort(key=lambda x: x[1], reverse=True)
                women_prefs[w] = [m for m, _ in men_scores]
            else:
                women_prefs[w] = list(men)
        
        # Run Gale-Shapley (men proposing)
        free_men = list(men)
        current_match = {}  # woman -> man
        next_proposal = {m: 0 for m in men}
        
        while free_men:
            man = free_men[0]
            
            if next_proposal[man] >= len(men_prefs[man]):
                free_men.remove(man)
                continue
            
            woman = men_prefs[man][next_proposal[man]]
            next_proposal[man] += 1
            
            if woman not in current_match:
                # Woman is free
                current_match[woman] = man
                free_men.remove(man)
            else:
                # Woman is already matched
                current_partner = current_match[woman]
                
                # Check woman's preference
                if woman in women_prefs:
                    try:
                        current_rank = women_prefs[woman].index(current_partner)
                        new_rank = women_prefs[woman].index(man)
                        
                        if new_rank < current_rank:  # Prefers new man
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
        
        # Add unmatched people
        for person in men | women:
            if person not in matching:
                matching[person] = 'rejected'
        
        return matching
    
    def calculate_switch_probability(self, score_diff, model='exponential'):
        """Calculate probability of switching based on score difference"""
        abs_diff = abs(score_diff)
        
        if model == 'exponential':
            prob = min(1.0, np.exp(self.lambda_exp * abs_diff))
        else:  # sigmoid
            prob = 1 / (1 + np.exp(-self.lambda_sig * abs_diff))
        
        return prob
    
    def calculate_expected_blocking_pairs(self, matching, scores_dict, model='exponential'):
        """Calculate expected number of blocking pairs using probability model"""
        # Extract matched pairs
        men = set()
        women = set()
        
        for person, partner in matching.items():
            if partner != 'rejected' and partner is not None:
                # Determine gender (using pattern from identify_men_and_women)
                try:
                    if int(person) % 2 == 1:
                        men.add(person)
                        women.add(partner)
                except:
                    # Alternate assignment
                    if person not in women:
                        men.add(person)
                    if partner not in men:
                        women.add(partner)
        
        # Calculate blocking pair probabilities
        expected_bp = 0
        blocking_pair_details = []
        
        for m in men:
            for w in women:
                # Skip if already matched
                if matching.get(m) == w:
                    continue
                
                # Get current partners
                m_partner = matching.get(m)
                w_partner = matching.get(w)
                
                if m_partner == 'rejected' or w_partner == 'rejected':
                    continue
                
                # Calculate score differences
                try:
                    # m's perspective: score(w) - score(current_partner)
                    score_m_current = scores_dict.get(m, {}).get(m_partner, 5.0)
                    score_m_w = scores_dict.get(m, {}).get(w, 5.0)
                    diff_m = score_m_w - score_m_current
                    
                    # w's perspective: score(m) - score(current_partner)
                    score_w_current = scores_dict.get(w, {}).get(w_partner, 5.0)
                    score_w_m = scores_dict.get(w, {}).get(m, 5.0)
                    diff_w = score_w_m - score_w_current
                    
                    # Calculate probabilities (using absolute value of differences)
                    prob_m = self.calculate_switch_probability(abs(diff_m), model)
                    prob_w = self.calculate_switch_probability(abs(diff_w), model)
                    
                    # Probability of being blocking pair
                    p_bp = prob_m * prob_w
                    expected_bp += p_bp
                    
                    if p_bp > 0.01:  # Record significant blocking pairs
                        blocking_pair_details.append({
                            'man': m,
                            'woman': w,
                            'diff_m': diff_m,
                            'diff_w': diff_w,
                            'prob_m': prob_m,
                            'prob_w': prob_w,
                            'p_blocking': p_bp
                        })
                
                except Exception as e:
                    continue
        
        return expected_bp, blocking_pair_details
    
    def analyze_group(self, group_num, model='exponential'):
        """Analyze a single group with Gale-Shapley matching"""
        # Parse scores
        scores_dict = self.parse_scores_from_csv(group_num)
        if not scores_dict:
            return None
        
        # Identify men and women
        men, women = self.identify_men_and_women(scores_dict)
        
        # Run Gale-Shapley to get stable matching
        gs_matching = self.run_gale_shapley(men, women, scores_dict)
        
        # Calculate expected blocking pairs
        expected_bp, bp_details = self.calculate_expected_blocking_pairs(gs_matching, scores_dict, model)
        
        # Count matched pairs
        matched_pairs = sum(1 for p, partner in gs_matching.items() 
                          if partner != 'rejected' and p in men) # Count each pair once
        
        return {
            'group': group_num,
            'num_men': len(men),
            'num_women': len(women),
            'num_matched_pairs': matched_pairs,
            'expected_blocking_pairs': expected_bp,
            'blocking_pair_details': bp_details
        }
    
    def analyze_all_groups(self, model='exponential'):
        """Analyze all groups using Gale-Shapley matching"""
        print(f"Analyzing stability with Gale-Shapley matching using {model} model...")
        
        for group_num in tqdm(range(1, 22)):
            result = self.analyze_group(group_num, model)
            if result:
                self.group_results[group_num] = result
        
        # Calculate overall statistics
        total_expected = sum([r['expected_blocking_pairs'] for r in self.group_results.values()])
        total_pairs = sum([r['num_matched_pairs'] for r in self.group_results.values()])
        
        self.overall_stats = {
            'total_expected_blocking_pairs': total_expected,
            'total_matched_pairs': total_pairs,
            'average_blocking_pairs_per_group': total_expected / len(self.group_results) if self.group_results else 0,
            'stability_score': 1 - (total_expected / total_pairs) if total_pairs > 0 else 1,
            'model': model
        }
        
        return self.overall_stats
    
    def plot_results(self):
        """Visualize results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        groups = sorted(self.group_results.keys())
        expected_bp = [self.group_results[g]['expected_blocking_pairs'] for g in groups]
        matched_pairs = [self.group_results[g]['num_matched_pairs'] for g in groups]
        
        # 1. Expected blocking pairs by group
        axes[0, 0].bar(groups, expected_bp, color='red', alpha=0.7)
        axes[0, 0].axhline(y=np.mean(expected_bp), color='black', linestyle='--',
                          label=f'Average: {np.mean(expected_bp):.2f}')
        axes[0, 0].set_xlabel('Group')
        axes[0, 0].set_ylabel('Expected Blocking Pairs')
        axes[0, 0].set_title('Expected Blocking Pairs by Group (Gale-Shapley Matching)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Matched pairs by group
        axes[0, 1].bar(groups, matched_pairs, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Group')
        axes[0, 1].set_ylabel('Number of Matched Pairs')
        axes[0, 1].set_title('Matched Pairs by Group')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stability score by group
        stability_scores = [1 - (expected_bp[i] / matched_pairs[i]) if matched_pairs[i] > 0 else 1
                           for i in range(len(groups))]
        
        axes[1, 0].plot(groups, stability_scores, 'o-', color='blue', markersize=8)
        axes[1, 0].axhline(y=self.overall_stats['stability_score'], color='red', linestyle='--',
                          label=f'Overall: {self.overall_stats["stability_score"]:.3f}')
        axes[1, 0].set_xlabel('Group')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].set_title('Stability Score by Group (1 - E[BP]/Matches)')
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary text
        summary_text = f"""Gale-Shapley Matching Stability Analysis
        
Total Expected Blocking Pairs: {self.overall_stats['total_expected_blocking_pairs']:.2f}
Total Matched Pairs: {self.overall_stats['total_matched_pairs']}
Average BP per Group: {self.overall_stats['average_blocking_pairs_per_group']:.2f}
Overall Stability Score: {self.overall_stats['stability_score']:.3f}

Model: {self.overall_stats['model']}
λ_exp = {self.lambda_exp}, λ_sig = {self.lambda_sig}

Note: Gale-Shapley produces stable matchings,
but expected BP > 0 due to probabilistic
switching behavior."""
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(self.base_path, 'gale_shapley_stability_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to: {output_file}")
    
    def compare_models(self):
        """Compare exponential and sigmoid models"""
        # Analyze with exponential
        exp_stats = self.analyze_all_groups(model='exponential')
        exp_results = self.group_results.copy()
        
        # Reset and analyze with sigmoid
        self.group_results = {}
        sig_stats = self.analyze_all_groups(model='sigmoid')
        sig_results = self.group_results.copy()
        
        print("\n" + "="*60)
        print("Model Comparison - Gale-Shapley Matching")
        print("="*60)
        
        print("\nExponential Model:")
        print(f"  Total Expected BP: {exp_stats['total_expected_blocking_pairs']:.2f}")
        print(f"  Stability Score: {exp_stats['stability_score']:.3f}")
        
        print("\nSigmoid Model:")
        print(f"  Total Expected BP: {sig_stats['total_expected_blocking_pairs']:.2f}")
        print(f"  Stability Score: {sig_stats['stability_score']:.3f}")
        
        return exp_stats, sig_stats

def main():
    """Run Gale-Shapley stability analysis"""
    analyzer = GaleShapleyStabilityAnalyzer()
    
    # Compare both models
    exp_stats, sig_stats = analyzer.compare_models()
    
    # Plot results (using exponential model)
    analyzer.analyze_all_groups(model='exponential')
    analyzer.plot_results()
    
    # Save results
    results = {
        'exponential_model': exp_stats,
        'sigmoid_model': sig_stats,
        'parameters': {
            'lambda_exponential': analyzer.lambda_exp,
            'lambda_sigmoid': analyzer.lambda_sig
        }
    }
    
    json_file = os.path.join(analyzer.base_path, 'gale_shapley_stability_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_file}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()