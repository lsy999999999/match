#!/usr/bin/env python3
"""
Script to fix malformed CSV files from the Gale-Shapley simulation
"""

import csv
import sys
import os

def fix_csv_file(input_file, output_file=None):
    """
    Fix malformed CSV file structure
    
    Args:
        input_file: Path to the malformed CSV file
        output_file: Path for the fixed CSV file (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file + '.fixed'
    
    fixed_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        for row_num, row in enumerate(reader):
            try:
                if len(row) == 0:
                    continue
                    
                function_type = row[0]
                
                if 'single' in function_type and 'non_single' not in function_type:
                    # Single proposal format
                    if len(row) >= 8 and row[2] == 'API_ERROR':
                        # Current malformed format: [type, score, 'API_ERROR', decision, target, proposer, '', result]
                        # We need to expand this to include all score fields
                        fixed_row = [
                            row[0],  # function_type
                            row[1],  # overall_score
                            'N/A',   # attractive (not available due to API error)
                            'N/A',   # sincere
                            'N/A',   # intelligence
                            'N/A',   # funny
                            'N/A',   # ambition
                            'N/A',   # shared_interests
                            f"API_ERROR: {row[3]}",  # decision with API_ERROR prefix
                            row[4],  # target
                            row[5],  # proposer
                            row[6],  # empty for single
                            row[7]   # result
                        ]
                        fixed_rows.append(fixed_row)
                    else:
                        # Already correct format or different error
                        fixed_rows.append(row)
                        
                elif 'non_single' in function_type:
                    # Non-single proposal format
                    if len(row) >= 7 and row[1] == 'API_ERROR':
                        # Current malformed format: [type, 'API_ERROR', decision, target, proposer, current, result]
                        # Need to fix this format
                        fixed_row = [
                            row[0],  # function_type
                            'current:N/A',  # current partner score not available
                            'proposer:N/A', # proposer score not available
                            'attr_diff:N/A',  # attribute differences not available
                            'sinc_diff:N/A',
                            'intel_diff:N/A',
                            f"API_ERROR: {row[2]}",  # decision with API_ERROR prefix
                            row[3],  # target
                            row[4],  # proposer
                            row[5],  # current partner
                            row[6] if len(row) > 6 else '0'  # result
                        ]
                        fixed_rows.append(fixed_row)
                    else:
                        # Check if this is the other malformed format seen in the data
                        # Where columns are shifted
                        if len(row) >= 4 and isinstance(row[1], str) and 'current:' not in str(row[1]):
                            # This appears to be shifted data
                            # Try to reconstruct
                            fixed_row = [
                                row[0],  # function_type
                                'current:N/A',
                                'proposer:N/A',
                                'attr_diff:N/A',
                                'sinc_diff:N/A',
                                'intel_diff:N/A',
                                'API_ERROR: Data corrupted',
                                row[1] if len(row) > 1 else '',  # target
                                row[2] if len(row) > 2 else '',  # proposer  
                                row[3] if len(row) > 3 else '',  # current
                                row[4] if len(row) > 4 else '0'  # result
                            ]
                            fixed_rows.append(fixed_row)
                        else:
                            # Already correct format
                            fixed_rows.append(row)
                else:
                    # Unknown format, keep as is
                    fixed_rows.append(row)
                    
            except Exception as e:
                print(f"Error processing row {row_num}: {e}")
                print(f"Row content: {row}")
                # Keep problematic rows as-is
                fixed_rows.append(row)
    
    # Write the fixed CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    
    print(f"Fixed CSV written to: {output_file}")
    print(f"Total rows processed: {len(fixed_rows)}")
    
    # Print sample of fixed data
    print("\nSample of fixed data (first 5 rows):")
    for i, row in enumerate(fixed_rows[:5]):
        print(f"Row {i}: {row}")
    
    return output_file


def fix_all_csv_files(directory, pattern="*_group*.csv"):
    """
    Fix all CSV files in a directory matching a pattern
    
    Args:
        directory: Directory containing CSV files
        pattern: Glob pattern for CSV files
    """
    import glob
    
    csv_files = glob.glob(os.path.join(directory, pattern))
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        try:
            fix_csv_file(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_csv.py <csv_file> [output_file]")
        print("   or: python fix_csv.py --directory <directory>")
        sys.exit(1)
    
    if sys.argv[1] == "--directory":
        if len(sys.argv) < 3:
            print("Please provide a directory path")
            sys.exit(1)
        fix_all_csv_files(sys.argv[2])
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fix_csv_file(input_file, output_file)