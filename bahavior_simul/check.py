import pandas as pd
import numpy as np

def check_available_groups(excel_path):
    """Check which groups are available in the dataset"""
    print("Checking available groups in the dataset...")
    print("="*50)
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Get unique groups
        unique_groups = sorted(df['group'].unique())
        
        print(f"Total unique groups found: {len(unique_groups)}")
        print(f"Group numbers: {unique_groups}")
        
        # Check each group's data
        print("\nDetailed group information:")
        print("-"*50)
        
        for group in unique_groups:
            group_data = df[df['group'] == group]
            men_count = len(group_data[group_data['gender'] == 1])
            women_count = len(group_data[group_data['gender'] == 0])
            
            print(f"Group {group}: {len(group_data)} total entries, {men_count} men, {women_count} women")
        
        # Check if groups 22-50 exist
        print("\n" + "="*50)
        print("Checking for groups 22-50:")
        groups_22_50 = [g for g in unique_groups if 22 <= g <= 50]
        if groups_22_50:
            print(f"Found groups: {groups_22_50}")
        else:
            print("No groups found in range 22-50!")
            
        return unique_groups
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def run_gale_shapley_for_existing_groups(excel_path):
    """Run Gale-Shapley only for groups that exist in the dataset"""
    from read_file_random import read_file, read_file_attr
    from gale_shapley_random import gale_shapley
    
    # First check which groups exist
    available_groups = check_available_groups(excel_path)
    
    if not available_groups:
        print("No groups found in the dataset!")
        return
    
    print("\n" + "="*50)
    print("Running Gale-Shapley for available groups...")
    print("="*50)
    
    # Run only for existing groups
    for i in available_groups:
        try:
            print(f"\nProcessing group {i}...")
            l1, l2 = read_file(excel_path, i)
            dict1, dict2 = read_file_attr(excel_path, i)
            
            if len(l1) > 0 and len(l2) > 0:
                gale_shapley(l1, l2, dict1, dict2, i)
                print(f"✓ Group {i} processed successfully")
            else:
                print(f"✗ Group {i} has no valid data")
                
        except Exception as e:
            print(f"✗ Error processing group {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Check the dataset
    excel_path = '/home/lsy/match/dataset/save_merge_select_null_3.xlsx'
    
    # Option 1: Just check what groups are available
    print("CHECKING DATASET...")
    available_groups = check_available_groups(excel_path)
    
    # Option 2: Run Gale-Shapley for all available groups
    # Uncomment the line below to run the algorithm for all existing groups
    # run_gale_shapley_for_existing_groups(excel_path)