import logging
import pandas as pd
import openpyxl
from gooster import Gooster
from simulator import Simulator
from config import GOO_VALUES,IDOL_MULT
import json
import re

def humanize_number(value, force_m=True, return_string=False):
    """Formats large numbers into a readable format (e.g., 2K, 12M)."""
    if force_m or value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M" if return_string else round(value / 1_000_000,3)
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K" if return_string else round(value / 1_000,3)
    return str(value)

def convert_to_winrate_df(results_df):
    """
    Converts the 3D results DataFrame into a 2D DataFrame with win rates and goo values.
    """
    return pd.DataFrame({
        attacker: {
            enemy: [
                round(stats['wins'] / stats['n'], 3),   # Winrate
                stats['n'],                              # Number of battles
                humanize_number(stats['goo'],  force_m=True, return_string=False)           # Human-readable goo
            ] if stats['n'] else None
            for enemy, stats in enemy_results.items()
        }
        for attacker, enemy_results in results_df.to_dict().items()
    }).T  # Transpose to make attackers rows and enemies columns
    
def export_results_to_excel(results_df, filename="simulation_results.xlsx"):
    """
    Converts the 3D results DataFrame into separate 2D matrices for win rates, battle counts, 
    and goo earnings, then saves them as sheets in an Excel file.
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame from simulate_random_samples.
    - filename (str): Name of the output Excel file.
    """
    winrate_matrix = {}
    battle_count_matrix = {}
    goo_matrix = {}
    gpm_matrix = {}

    for attacker, enemy_results in results_df.to_dict().items():
        winrate_matrix[attacker] = {}
        battle_count_matrix[attacker] = {}
        goo_matrix[attacker] = {}
        gpm_matrix[attacker] = {}

        for enemy, stats in enemy_results.items():
            if isinstance(stats, dict) and stats['n']:  # Avoid division by zero
                winrate_matrix[attacker][enemy] = round(stats['wins'] / stats['n'], 3)
                battle_count_matrix[attacker][enemy] = stats['n']
                goo_matrix[attacker][enemy] = humanize_number(stats['goo'])
                gpm_matrix[attacker][enemy] = humanize_number(stats['gpm'])
            else:
                winrate_matrix[attacker][enemy] = None
                battle_count_matrix[attacker][enemy] = None
                goo_matrix[attacker][enemy] = None
                gpm_matrix[attacker][enemy] = None

    # Convert to DataFrame
    winrate_df = pd.DataFrame(winrate_matrix).T
    battle_count_df = pd.DataFrame(battle_count_matrix).T
    goo_df = pd.DataFrame(goo_matrix).T
    gpm_df = pd.DataFrame(gpm_matrix).T

    # Export to Excel
    with pd.ExcelWriter(filename) as writer:
        winrate_df.to_excel(writer, sheet_name="Win Rates")
        battle_count_df.to_excel(writer, sheet_name="Battle Counts")
        goo_df.to_excel(writer, sheet_name="Goo Earnings")
        gpm_df.to_excel(writer, sheet_name="GPM")

    print(f"Results exported to {filename}")

import pandas as pd

def get_v2_df(results_df):
    """
    Converts the 3D results DataFrame into a single structured DataFrame,
    pivoting enemy types for Goo Earned, Win Rate, and Encounter Rate while keeping Total KPIs.

    Parameters:
    - results_df (pd.DataFrame): The results DataFrame from simulate_random_samples.
    - filename (str): Name of the output Excel file.
    """
    processed_data = []

    for attacker, enemy_results in results_df.to_dict().items():
        total_goo = sum(stats['goo'] for stats in enemy_results.values() if isinstance(stats, dict))
        total_n = sum(stats['n'] for stats in enemy_results.values() if isinstance(stats, dict))
        gpm = total_goo / (total_n / 1000) if total_n else 0  # Correct GPM formula

        row_data = {
            "Attacker": attacker,
            "Total Goo": humanize_number(total_goo), 
            "Total Battles": total_n, 
            "GPM": humanize_number(gpm)
        }

        for enemy, stats in enemy_results.items():
            if isinstance(stats, dict) and stats['n']:  # Avoid division by zero
                row_data[f"{enemy} Goo"] = humanize_number(stats['goo'])
                row_data[f"{enemy} Win Rate"] = f"{round(stats['wins'] / stats['n'], 3):.1%}"
                row_data[f"{enemy} Encounter Rate"] = f"{round(stats['n'] / total_n, 3):.1%}"

        processed_data.append(row_data)

    # Convert to DataFrame
    output_df = pd.DataFrame(processed_data)

    # Ensure GPM is sorted numerically
    output_df["GPM"] = pd.to_numeric(output_df["GPM"], errors="coerce")
    output_df = output_df.sort_values(by="GPM", ascending=False)

    # Reorder columns: KPI columns first, then grouped enemy columns by metric
    kpi_columns = ["Attacker", "Total Goo", "Total Battles", "GPM"]

    enemy_types = sorted({col.split()[0] for col in output_df.columns if " " in col})  # Extract unique enemy names
    goo_columns = [f"{enemy} Goo" for enemy in enemy_types if f"{enemy} Goo" in output_df.columns]
    win_rate_columns = [f"{enemy} Win Rate" for enemy in enemy_types if f"{enemy} Win Rate" in output_df.columns]
    encounter_rate_columns = [f"{enemy} Encounter Rate" for enemy in enemy_types if f"{enemy} Encounter Rate" in output_df.columns]

    ordered_columns = kpi_columns + goo_columns + win_rate_columns + encounter_rate_columns
    output_df = output_df[ordered_columns]

    return output_df
    # Export to Excel
    output_df.to_excel(filename, index=False)




def sim_all_builds_cross_product():
    """
    Main entry point for the simulation.
    """
    test_level = 30
    permutations = Simulator.generate_permutations(test_level)
    print(f"Generated {len(permutations)} permutations.")

    battle_results = Simulator.simulate_matrix(permutations,permutations, n=100)
    file_name = f'g_matrix_{test_level}.csv'
    battle_results.to_csv(file_name)
    print(f'Simulation completed. Results saved to {file_name}')

def sim_set_builds():
    attackers = [
        Gooster(upgrade_counts=[9,10,0,10]), 
        Gooster(upgrade_counts=[12,10,1,9]),
        Gooster(upgrade_counts=[10,10,1,11]),
        Gooster(upgrade_counts=[1,15,16,0]),
        Gooster(upgrade_counts=[6,10,16,0]),
        Gooster(upgrade_counts=[1,10,16,5]),
        Gooster(stats=[190,20,21,13]),
    ]

    df = Simulator.simulate_random_samples(attackers, 10000)
    # df2 = Simulator.convert_to_winrate_df(df)
    # file_name = f'g_sample.csv'
    # df2.to_csv(file_name)
    return df



def sim_all_good_builds(level, bosstype=None):
    required_upgrades = [5,10,5,0]
    target_level = level
    remaining_levels = target_level - sum(required_upgrades) - 1
    ## get all combos but only include attack ends with 2 or 5 also total attack <=35
    next_upgrades = [p for p in Simulator._partitions(remaining_levels, 4) if ((p[1] % 10 == 2 or p[1] % 5 == 0)) ]
    print(f'Created {len(next_upgrades)} builds for level {level} {'normal' if bosstype==None else bosstype} gooster.')

    attackers = []
    for upgrade in next_upgrades:
        g = Gooster(upgrade_counts=required_upgrades)
        g.apply_upgrades(*upgrade)
        g.boss_type=bosstype
        attackers.append(g)
    
    df = Simulator.simulate_random_samples(attackers, n=10000)
    return df

def sim_blitz(level, bosstype=None):
    required_upgrades = [10,10,10,0]
    target_level = level
    remaining_levels = target_level - sum(required_upgrades) - 1
    ## get all combos but only include attack ends with 2 or 5 also total attack <=35
    next_upgrades = [p for p in Simulator._partitions(remaining_levels, 4) if ((p[1] % 10 == 2 or p[1] % 5 == 0)) ]
    print(f'Created {len(next_upgrades)} builds for level {level} {'normal' if bosstype==None else bosstype} gooster.')

    attackers = []
    for upgrade in next_upgrades:
        g = Gooster(upgrade_counts=required_upgrades)
        g.apply_upgrades(*upgrade)
        g.boss_type=bosstype
        attackers.append(g)
    
    df = Simulator.simulate_random_samples(attackers, fixed_level=level, n=3000)
    return df

def sim_1v1(attacker_stats,defender_stats,n):
    attacker = Gooster(stats=attacker_stats)
    attackers = [attacker]
    print(attacker)
    df = Simulator.simulate_random_samples(attackers, defender_stats=defender_stats, n=n)
    return df


import pandas as pd
import json

def build_paths(filepath, n=5, m=3):
    '''
    Parameters:
    n = 5  # Number of top builds to keep per level
    m = 3  # Max number of viable builds per upgrade path
    '''

    # Read CSV into DataFrame
    df = pd.read_excel(filepath)

    # Ensure 'Attacker' column is correctly formatted before splitting
    df[['level', 'stats']] = df['Attacker'].str.split('|', expand=True, n=1)
    df['level'] = df['level'].astype(int)

    # Split stats into individual components
    df[['hp', 'atk', 'spd', 'ddg']] = df['stats'].str.split(',', expand=True).astype(float).astype(pd.Int64Dtype())
    df.drop(columns=['stats'], inplace=True)

    # Calculate rank and efficiency within each level
    df['Rank'] = df.groupby('level')['GPM'].rank(ascending=False, method='dense').astype(int)
    df['Max_GPM'] = df.groupby('level')['GPM'].transform('max')
    df['Efficiency'] = df['GPM'] / df['Max_GPM']
    df.drop(columns=['Max_GPM'], inplace=True)

    # Get the highest level in the dataset
    max_level = df['level'].max()
    min_level = df['level'].min()

    # Function to check if a build is viable based on the new criteria
    def is_viable_upgrade(current, previous):
        """Returns True if exactly one stat in `previous` is 1 less than `current`, and others are the same."""
        diffs = (current[['hp', 'atk', 'spd', 'ddg']] - previous[['hp', 'atk', 'spd', 'ddg']])
        return (diffs == 1).sum() == 1 and (diffs == 0).sum() == 3

    def build_upgrade_tree(build, df, m, levels_remaining, efficiency_sum=0, weight_sum=0):
        """Recursively builds a tree of upgrade paths and tracks average efficiency with weighted average calculation."""
        weight = 2**levels_remaining
        prev_level = build['level'] - 1

        if prev_level < df['level'].min():
            # If we reached the bottom, return a leaf with weighted average efficiency
            avg_efficiency = efficiency_sum / weight_sum
            return {"Average Efficiency": round(avg_efficiency, 4)}

        # Get all potential builds from the previous level
        prev_builds = df[df['level'] == prev_level]
        # Filter viable upgrade paths
        viable_upgrades = prev_builds[prev_builds.apply(lambda x: is_viable_upgrade(build, x), axis=1)]
        # Keep only the top `m` based on GPM
        viable_upgrades = viable_upgrades.nlargest(m, 'GPM')

        # If no viable upgrades, return the leaf node with weighted average efficiency
        if viable_upgrades.empty:
            avg_efficiency = efficiency_sum / weight_sum
            return {"Average Efficiency": round(avg_efficiency, 4)}

        # Recursively build the tree
        return {
            f"Build {row['Attacker']} (GPM: {row['GPM']}, Efficiency: {row['Efficiency']:.2f})": 
            build_upgrade_tree(
                row, df, m, levels_remaining-1,
                efficiency_sum + (build['Efficiency']*weight), weight_sum + weight
            )
            for _, row in viable_upgrades.iterrows()
        }

    n = 5
    m = 3
    # Find the top `n` builds at the highest level
    top_builds = df[df['level'] == max_level].nlargest(n, 'GPM')

    # Construct the full upgrade tree
    upgrade_tree = {
        f"BUILD {row['Attacker']} (GPM: {row['GPM']}, Efficiency: {row['Efficiency']:.2f})": 
        build_upgrade_tree(row, df, m, levels_remaining=max_level-min_level)
        for _, row in top_builds.iterrows()
    }

    def print_tree(tree, level=0, prefix=""):
        """Recursively prints the tree with better readability."""
        for i, (key, sub_tree) in enumerate(tree.items()):
            is_last = (i == len(tree) - 1)
            branch = "└── " if is_last else "├── "
            
            if isinstance(sub_tree, float):  # If it's a leaf node (float value)
                print(prefix + branch + f"{key}: {sub_tree*100:.1f}%")
            else:
                print(prefix + branch + key)
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(sub_tree, level + 1, new_prefix)

    def flatten_upgrade_tree(tree, path="", results=None):
        """Flattens the upgrade tree into a DataFrame format."""
        if results is None:
            results = []
        
        for key, sub_tree in tree.items():
            if key == "Average Efficiency":
                # Store the final path and efficiency at the leaf node
                results.append({"average_efficiency": sub_tree, "build_path": path.strip(" -> ")})
            else:                    
                # Append to the path and recurse
                build_name = key.replace("Build ","").replace("Efficiency ","eff")
                new_path = f"{build_name} -> {path}" if path else f"{key}"
                flatten_upgrade_tree(sub_tree, new_path, results)
        
        return pd.DataFrame(results)


    # Save the tree to a JSON file
    with open("upgrade_tree.json", "w") as f:
        json.dump(upgrade_tree, f, indent=4)

    print("Upgrade tree saved to upgrade_tree.json")

    # Print final tree structure
    print_tree(upgrade_tree)

    upgrade_tree_df = flatten_upgrade_tree(upgrade_tree).sort_values(by="average_efficiency", ascending=False)
    print(upgrade_tree_df.head(10))
    upgrade_tree_df.to_excel("upgrade_tree_flattened.xlsx", index=False)




if __name__ == "__main__":

    filename="simulation_results_v2.xlsx"
    logging.basicConfig(level=logging.WARNING)
    # sim_all_builds_cross_product()
    # test_angel_gen()
    # sim_set_builds()
    build_paths(filename,5,3) ## top 5 builds with 10 branches each

    ############################## BUILDS1   ##############################
    # df = pd.DataFrame()
    # for level in range(33, 39+1):
    #     # df1 = get_v2_df(sim_all_good_builds(level=level))
    #     # print(df1[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    #     # df = pd.concat([df, df1], ignore_index=True)  # Append df1

    #     # df2 = get_v2_df(sim_all_good_builds(level=level, bosstype='angel'))
    #     # print(df2[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    #     # df = pd.concat([df, df2], ignore_index=True)  # Append df2

    #     df3 = get_v2_df(sim_all_good_builds(level=level, bosstype='ultima'))
    #     print(df3[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    #     df = pd.concat([df, df3], ignore_index=True)  # Append df3
    # df.to_excel(filename, index=False)

    ############################## BUILDS2  ##############################
    # level = 37
    # df1 = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    # print(df1[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    # level = 38
    # df2 = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    # print(df2[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    # level = 39
    # df3 = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    # print(df2[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    # # Union the two DataFrames
    # df_union = pd.concat([df1, df2,df3], ignore_index=True)
    # df_union.to_excel(filename, index=False)


    ############################## BLITZ    ##############################
    # level = 60
    # df = get_v2_df(sim_blitz
    #                (level=level))
    # print(df.head(n=15).to_string(index=False))

    ############################## 1v1      ##############################
    # sim_1v1([100,10,11,10],[110,10,10,10],100000)
    # sim_1v1([140,14,11,10],[140,13,10,12],100000)

    ############################## OUTPUT   ##############################
    # output_df.to_excel(filename, index=False)
    # print(f"Results exported to {filename}")

## GPM = goo(in millions) / (n/1000) 
## Angel builds
# 31: 210,20,18,11 54.491
# 32: 210,20,18,12 59.765 (5.3/41M)
# 33: 210,20,19,12 68.774 (9/82M)
# 34: 220,20,19,12 71.430 (34|210,20,21,11 73.438) (2.5/160M)

## Ultima builds
# 33|210,20,18,13 79.293
# 34|210,20,20,12 99.663

## early ULTIMA
# 33|210,20,19,12
# 34|210,20,19,13
# 35|210,20,19,14
# 36|220,20,19,14

## 22 Branch  (not optimized)
# 36|230,22,18,12 124.220
# 37|230,22,19,12 140.294 (16/1300M)
# 38|230,22,19,13 149.038 (9/2620M)
# 39|240,22,19,13 149.095

## Best Branch
# 37|210,25,18,12 133.171
# 38|190,25,19,14 143.472
# 39|220,25,18,13 160.041

## 20 Branch (optimized for 39)
# 37|220,20,22,12 132.690
# 38|220,20,23,12 139.260
# 39|220,20,24,12 152.210

# Created 397 builds for level 37 ultima gooster.
#        Attacker  Total Goo  Total Goo     GPM
# 37|210,25,18,12   1331.714   1331.714 133.171
# 37|220,20,22,12   1326.905   1326.905 132.690
# 37|190,25,20,12   1309.444   1309.444 130.944
# 37|240,22,18,12   1309.081   1309.081 130.908
# 37|210,25,19,11   1303.137   1303.137 130.314
# 37|250,20,18,13   1288.403   1288.403 128.840
# 37|230,20,21,12   1286.816   1286.816 128.682
# 37|240,20,20,12   1281.816   1281.816 128.182
# 37|230,25,16,12   1281.419   1281.419 128.142
# 37|230,22,19,12   1278.235   1278.235 127.823
# Created 461 builds for level 38 ultima gooster.
#        Attacker  Total Goo  Total Goo     GPM
# 38|210,22,22,12   1442.563   1442.563 144.256
# 38|190,25,19,14   1434.715   1434.715 143.472
# 38|210,20,22,14   1428.468   1428.468 142.847
# 38|220,20,22,13   1420.653   1420.653 142.065
# 38|220,25,17,13   1414.952   1414.952 141.495
# 38|230,22,20,12   1408.637   1408.637 140.864
# 38|230,20,21,13   1398.210   1398.210 139.821
# 38|190,25,20,13   1397.450   1397.450 139.745
# 38|240,20,21,12   1393.668   1393.668 139.367
# 38|220,20,23,12   1392.604   1392.604 139.260
# Created 531 builds for level 39 ultima gooster.
#        Attacker  Total Goo  Total Goo     GPM
# 39|220,25,18,13   1600.414   1600.414 160.041
# 39|220,20,24,12   1522.098   1522.098 152.210
# 39|230,20,24,11   1519.916   1519.916 151.992
# 39|220,25,19,12   1519.678   1519.678 151.968
# 39|210,22,22,13   1508.874   1508.874 150.887
# 39|240,20,21,13   1495.741   1495.741 149.574
# 39|230,20,23,12   1491.619   1491.619 149.162
# 39|220,20,21,15   1490.206   1490.206 149.021
# 39|190,25,22,12   1487.185   1487.185 148.718
# 39|230,25,16,14   1480.284   1480.284 148.028

# Upgrade tree saved to upgrade_tree.json
# ├── BUILD 39|240,20,21,13 (GPM: 154.449, Efficiency: 1.00)
# │   ├── Build 38|240,20,21,12 (GPM: 133.933, Efficiency: 0.93)
# │   │   ├── Build 37|240,20,20,12 (GPM: 126.977, Efficiency: 0.95)
# │   │   │   ├── Build 36|240,20,19,12 (GPM: 115.392, Efficiency: 0.92)
# │   │   │   │   ├── Build 35|240,20,18,12 (GPM: 101.278, Efficiency: 0.92)
# │   │   │   │   │   ├── Build 34|240,20,17,12 (GPM: 93.375, Efficiency: 0.94)
# │   │   │   │   │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │   │   │   │   │   │   │   └── Average Efficiency: 96.9%
# │   │   │   │   │   │   └── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │   │   │   │   │   │       └── Average Efficiency: 96.9%
# │   │   │   │   │   └── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │   │   │   │   │       ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │   │   │   │   │       │   └── Average Efficiency: 96.9%
# │   │   │   │   │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │   │   │           └── Average Efficiency: 96.9%
# │   │   │   │   └── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │   │   │   │       ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │   │   │   │       │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │   │   │   │       │   │   └── Average Efficiency: 96.8%
# │   │   │   │       │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │   │       │       └── Average Efficiency: 96.8%
# │   │   │   │       └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │   │   │           └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │   │               └── Average Efficiency: 96.7%
# │   │   │   └── Build 36|240,20,20,11 (GPM: 107.899, Efficiency: 0.86)
# │   │   │       ├── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │   │   │       │   ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │   │   │       │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │   │   │       │   │   │   └── Average Efficiency: 96.4%
# │   │   │       │   │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │       │   │       └── Average Efficiency: 96.4%
# │   │   │       │   └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │   │       │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │       │           └── Average Efficiency: 96.4%
# │   │   │       └── Build 35|240,20,20,10 (GPM: 96.773, Efficiency: 0.88)
# │   │   │           └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │   │               └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │   │                   └── Average Efficiency: 96.3%
# │   │   └── Build 37|240,20,21,11 (GPM: 118.978, Efficiency: 0.89)
# │   │       ├── Build 36|240,20,20,11 (GPM: 107.899, Efficiency: 0.86)
# │   │       │   ├── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │   │       │   │   ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │   │       │   │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │   │       │   │   │   │   └── Average Efficiency: 95.7%
# │   │       │   │   │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │       │   │   │       └── Average Efficiency: 95.7%
# │   │       │   │   └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │       │   │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │       │   │           └── Average Efficiency: 95.6%
# │   │       │   └── Build 35|240,20,20,10 (GPM: 96.773, Efficiency: 0.88)
# │   │       │       └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │       │           └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │       │               └── Average Efficiency: 95.5%
# │   │       └── Build 36|240,20,21,10 (GPM: 102.756, Efficiency: 0.82)
# │   │           └── Build 35|240,20,20,10 (GPM: 96.773, Efficiency: 0.88)
# │   │               └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │   │                   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │   │                       └── Average Efficiency: 95.2%
# │   └── Build 38|240,20,20,13 (GPM: 132.346, Efficiency: 0.92)
# │       ├── Build 37|240,20,20,12 (GPM: 126.977, Efficiency: 0.95)
# │       │   ├── Build 36|240,20,19,12 (GPM: 115.392, Efficiency: 0.92)
# │       │   │   ├── Build 35|240,20,18,12 (GPM: 101.278, Efficiency: 0.92)
# │       │   │   │   ├── Build 34|240,20,17,12 (GPM: 93.375, Efficiency: 0.94)
# │       │   │   │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │       │   │   │   │   │   └── Average Efficiency: 96.6%
# │       │   │   │   │   └── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │       │   │   │   │       └── Average Efficiency: 96.6%
# │       │   │   │   └── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │       │   │   │       ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │       │   │   │       │   └── Average Efficiency: 96.6%
# │       │   │   │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │   │   │           └── Average Efficiency: 96.6%
# │       │   │   └── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │       │   │       ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │       │   │       │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │       │   │       │   │   └── Average Efficiency: 96.5%
# │       │   │       │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │   │       │       └── Average Efficiency: 96.5%
# │       │   │       └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │       │   │           └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │   │               └── Average Efficiency: 96.5%
# │       │   └── Build 36|240,20,20,11 (GPM: 107.899, Efficiency: 0.86)
# │       │       ├── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │       │       │   ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │       │       │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │       │       │   │   │   └── Average Efficiency: 96.1%
# │       │       │   │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │       │   │       └── Average Efficiency: 96.1%
# │       │       │   └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │       │       │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │       │           └── Average Efficiency: 96.1%
# │       │       └── Build 35|240,20,20,10 (GPM: 96.773, Efficiency: 0.88)
# │       │           └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │       │               └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │       │                   └── Average Efficiency: 96.0%
# │       └── Build 37|240,20,19,13 (GPM: 121.455, Efficiency: 0.91)
# │           ├── Build 36|240,20,18,13 (GPM: 116.682, Efficiency: 0.94)
# │           │   ├── Build 35|240,20,17,13 (GPM: 109.68, Efficiency: 0.99)
# │           │   │   ├── Build 34|240,20,17,12 (GPM: 93.375, Efficiency: 0.94)
# │           │   │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │           │   │   │   │   └── Average Efficiency: 96.4%
# │           │   │   │   └── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │           │   │   │       └── Average Efficiency: 96.4%
# │           │   │   └── Build 34|240,20,16,13 (GPM: 90.277, Efficiency: 0.91)
# │           │   │       ├── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │           │   │       │   └── Average Efficiency: 96.3%
# │           │   │       └── Build 33|240,20,15,13 (GPM: 59.729, Efficiency: 0.79)
# │           │   │           └── Average Efficiency: 96.3%
# │           │   └── Build 35|240,20,18,12 (GPM: 101.278, Efficiency: 0.92)
# │           │       ├── Build 34|240,20,17,12 (GPM: 93.375, Efficiency: 0.94)
# │           │       │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │           │       │   │   └── Average Efficiency: 96.2%
# │           │       │   └── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │           │       │       └── Average Efficiency: 96.2%
# │           │       └── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │           │           ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │           │           │   └── Average Efficiency: 96.1%
# │           │           └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │           │               └── Average Efficiency: 96.1%
# │           └── Build 36|240,20,19,12 (GPM: 115.392, Efficiency: 0.92)
# │               ├── Build 35|240,20,18,12 (GPM: 101.278, Efficiency: 0.92)
# │               │   ├── Build 34|240,20,17,12 (GPM: 93.375, Efficiency: 0.94)
# │               │   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │               │   │   │   └── Average Efficiency: 96.1%
# │               │   │   └── Build 33|240,20,16,12 (GPM: 59.985, Efficiency: 0.79)
# │               │   │       └── Average Efficiency: 96.1%
# │               │   └── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │               │       ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │               │       │   └── Average Efficiency: 96.0%
# │               │       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │               │           └── Average Efficiency: 96.0%
# │               └── Build 35|240,20,19,11 (GPM: 99.361, Efficiency: 0.90)
# │                   ├── Build 34|240,20,18,11 (GPM: 91.533, Efficiency: 0.92)
# │                   │   ├── Build 33|240,20,17,11 (GPM: 64.493, Efficiency: 0.85)
# │                   │   │   └── Average Efficiency: 96.0%
# │                   │   └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │                   │       └── Average Efficiency: 96.0%
# │                   └── Build 34|240,20,19,10 (GPM: 86.584, Efficiency: 0.87)
# │                       └── Build 33|240,20,18,10 (GPM: 63.506, Efficiency: 0.83)
# │                           └── Average Efficiency: 95.9%
# ├── BUILD 39|270,20,19,12 (GPM: 152.6, Efficiency: 0.99)
# │   ├── Build 38|270,20,18,12 (GPM: 133.15, Efficiency: 0.93)
# │   │   ├── Build 37|270,20,17,12 (GPM: 118.008, Efficiency: 0.89)
# │   │   │   ├── Build 36|270,20,17,11 (GPM: 109.45, Efficiency: 0.88)
# │   │   │   │   ├── Build 35|270,20,16,11 (GPM: 92.287, Efficiency: 0.84)
# │   │   │   │   │   ├── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │   │   │   │   │   └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │   │   │   │       └── Average Efficiency: 94.5%
# │   │   │   │   │   └── Build 34|270,20,15,11 (GPM: 75.554, Efficiency: 0.76)
# │   │   │   │   │       └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │   │   │           └── Average Efficiency: 94.4%
# │   │   │   │   └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │   │   │   │       └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │   │   │           └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │   │               └── Average Efficiency: 94.4%
# │   │   │   └── Build 36|270,20,16,12 (GPM: 100.972, Efficiency: 0.81)
# │   │   │       ├── Build 35|270,20,16,11 (GPM: 92.287, Efficiency: 0.84)
# │   │   │       │   ├── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │   │       │   │   └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │       │   │       └── Average Efficiency: 94.1%
# │   │   │       │   └── Build 34|270,20,15,11 (GPM: 75.554, Efficiency: 0.76)
# │   │   │       │       └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │       │           └── Average Efficiency: 94.0%
# │   │   │       └── Build 35|270,20,15,12 (GPM: 90.454, Efficiency: 0.82)
# │   │   │           └── Build 34|270,20,15,11 (GPM: 75.554, Efficiency: 0.76)
# │   │   │               └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │   │                   └── Average Efficiency: 94.0%
# │   │   └── Build 37|270,20,18,11 (GPM: 113.639, Efficiency: 0.85)
# │   │       ├── Build 36|270,20,17,11 (GPM: 109.45, Efficiency: 0.88)
# │   │       │   ├── Build 35|270,20,16,11 (GPM: 92.287, Efficiency: 0.84)
# │   │       │   │   ├── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │       │   │   │   └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │       │   │   │       └── Average Efficiency: 94.1%
# │   │       │   │   └── Build 34|270,20,15,11 (GPM: 75.554, Efficiency: 0.76)
# │   │       │   │       └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │       │   │           └── Average Efficiency: 94.0%
# │   │       │   └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │   │       │       └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │       │           └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │       │               └── Average Efficiency: 94.0%
# │   │       └── Build 36|270,20,18,10 (GPM: 105.341, Efficiency: 0.84)
# │   │           └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │   │               └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │   │                   └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │   │                       └── Average Efficiency: 93.8%
# │   └── Build 38|270,20,19,11 (GPM: 122.351, Efficiency: 0.85)
# │       ├── Build 37|270,20,18,11 (GPM: 113.639, Efficiency: 0.85)
# │       │   ├── Build 36|270,20,17,11 (GPM: 109.45, Efficiency: 0.88)
# │       │   │   ├── Build 35|270,20,16,11 (GPM: 92.287, Efficiency: 0.84)
# │       │   │   │   ├── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │       │   │   │   │   └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │       │   │   │   │       └── Average Efficiency: 92.2%
# │       │   │   │   └── Build 34|270,20,15,11 (GPM: 75.554, Efficiency: 0.76)
# │       │   │   │       └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │       │   │   │           └── Average Efficiency: 92.1%
# │       │   │   └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │       │   │       └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │       │   │           └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │       │   │               └── Average Efficiency: 92.1%
# │       │   └── Build 36|270,20,18,10 (GPM: 105.341, Efficiency: 0.84)
# │       │       └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │       │           └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │       │               └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │       │                   └── Average Efficiency: 91.8%
# │       └── Build 37|270,20,19,10 (GPM: 111.415, Efficiency: 0.84)
# │           └── Build 36|270,20,18,10 (GPM: 105.341, Efficiency: 0.84)
# │               └── Build 35|270,20,17,10 (GPM: 88.38, Efficiency: 0.80)
# │                   └── Build 34|270,20,16,10 (GPM: 79.184, Efficiency: 0.80)
# │                       └── Build 33|270,20,15,10 (GPM: 46.811, Efficiency: 0.62)
# │                           └── Average Efficiency: 91.6%
# ├── BUILD 39|230,20,21,14 (GPM: 152.555, Efficiency: 0.99)
# │   ├── Build 38|230,20,21,13 (GPM: 140.21, Efficiency: 0.98)
# │   │   ├── Build 37|230,20,20,13 (GPM: 127.736, Efficiency: 0.96)
# │   │   │   ├── Build 36|230,20,19,13 (GPM: 117.427, Efficiency: 0.94)
# │   │   │   │   ├── Build 35|230,20,18,13 (GPM: 105.017, Efficiency: 0.95)
# │   │   │   │   │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │   │   │   │   │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │   │   │   │   │   └── Average Efficiency: 97.7%
# │   │   │   │   │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │   │   │   │   │   │       └── Average Efficiency: 97.7%
# │   │   │   │   │   └── Build 34|230,20,17,13 (GPM: 91.47, Efficiency: 0.92)
# │   │   │   │   │       ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │   │   │   │   │       │   └── Average Efficiency: 97.6%
# │   │   │   │   │       └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │   │   │   │   │           └── Average Efficiency: 97.6%
# │   │   │   │   └── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │   │   │   │       ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │   │   │   │       │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │   │       │   │   └── Average Efficiency: 97.7%
# │   │   │   │       │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │   │   │   │       │       └── Average Efficiency: 97.7%
# │   │   │   │       └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │   │   │           ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │   │           │   └── Average Efficiency: 97.6%
# │   │   │   │           └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │   │   │               └── Average Efficiency: 97.6%
# │   │   │   └── Build 36|230,20,20,12 (GPM: 113.469, Efficiency: 0.91)
# │   │   │       ├── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │   │   │       │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │   │   │       │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │       │   │   │   └── Average Efficiency: 97.5%
# │   │   │       │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │   │   │       │   │       └── Average Efficiency: 97.5%
# │   │   │       │   └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │   │       │       ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │       │       │   └── Average Efficiency: 97.4%
# │   │   │       │       └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │   │       │           └── Average Efficiency: 97.4%
# │   │   │       └── Build 35|230,20,20,11 (GPM: 104.285, Efficiency: 0.95)
# │   │   │           ├── Build 34|230,20,20,10 (GPM: 90.798, Efficiency: 0.91)
# │   │   │           │   └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │   │           │       └── Average Efficiency: 97.4%
# │   │   │           └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │   │               ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │   │               │   └── Average Efficiency: 97.4%
# │   │   │               └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │   │                   └── Average Efficiency: 97.4%
# │   │   └── Build 37|230,20,21,12 (GPM: 126.112, Efficiency: 0.95)
# │   │       ├── Build 36|230,20,20,12 (GPM: 113.469, Efficiency: 0.91)
# │   │       │   ├── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │   │       │   │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │   │       │   │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │       │   │   │   │   └── Average Efficiency: 97.4%
# │   │       │   │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │   │       │   │   │       └── Average Efficiency: 97.4%
# │   │       │   │   └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │       │   │       ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │       │   │       │   └── Average Efficiency: 97.2%
# │   │       │   │       └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │       │   │           └── Average Efficiency: 97.2%
# │   │       │   └── Build 35|230,20,20,11 (GPM: 104.285, Efficiency: 0.95)
# │   │       │       ├── Build 34|230,20,20,10 (GPM: 90.798, Efficiency: 0.91)
# │   │       │       │   └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │       │       │       └── Average Efficiency: 97.2%
# │   │       │       └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │       │           ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │       │           │   └── Average Efficiency: 97.2%
# │   │       │           └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │       │               └── Average Efficiency: 97.2%
# │   │       └── Build 36|230,20,21,11 (GPM: 110.416, Efficiency: 0.88)
# │   │           ├── Build 35|230,20,20,11 (GPM: 104.285, Efficiency: 0.95)
# │   │           │   ├── Build 34|230,20,20,10 (GPM: 90.798, Efficiency: 0.91)
# │   │           │   │   └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │           │   │       └── Average Efficiency: 97.1%
# │   │           │   └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │   │           │       ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │   │           │       │   └── Average Efficiency: 97.1%
# │   │           │       └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │           │           └── Average Efficiency: 97.1%
# │   │           └── Build 35|230,20,21,10 (GPM: 102.118, Efficiency: 0.93)
# │   │               └── Build 34|230,20,20,10 (GPM: 90.798, Efficiency: 0.91)
# │   │                   └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │   │                       └── Average Efficiency: 97.0%
# │   └── Build 38|230,20,20,14 (GPM: 132.391, Efficiency: 0.92)
# │       ├── Build 37|230,20,20,13 (GPM: 127.736, Efficiency: 0.96)
# │       │   ├── Build 36|230,20,19,13 (GPM: 117.427, Efficiency: 0.94)
# │       │   │   ├── Build 35|230,20,18,13 (GPM: 105.017, Efficiency: 0.95)
# │       │   │   │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │       │   │   │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │   │   │   │   │   └── Average Efficiency: 96.3%
# │       │   │   │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │       │   │   │   │       └── Average Efficiency: 96.3%
# │       │   │   │   └── Build 34|230,20,17,13 (GPM: 91.47, Efficiency: 0.92)
# │       │   │   │       ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │       │   │   │       │   └── Average Efficiency: 96.2%
# │       │   │   │       └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │       │   │   │           └── Average Efficiency: 96.2%
# │       │   │   └── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │       │   │       ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │       │   │       │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │   │       │   │   └── Average Efficiency: 96.3%
# │       │   │       │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │       │   │       │       └── Average Efficiency: 96.3%
# │       │   │       └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │       │   │           ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │   │           │   └── Average Efficiency: 96.2%
# │       │   │           └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │       │   │               └── Average Efficiency: 96.2%
# │       │   └── Build 36|230,20,20,12 (GPM: 113.469, Efficiency: 0.91)
# │       │       ├── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │       │       │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │       │       │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │       │   │   │   └── Average Efficiency: 96.1%
# │       │       │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │       │       │   │       └── Average Efficiency: 96.1%
# │       │       │   └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │       │       │       ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │       │       │   └── Average Efficiency: 96.0%
# │       │       │       └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │       │       │           └── Average Efficiency: 96.0%
# │       │       └── Build 35|230,20,20,11 (GPM: 104.285, Efficiency: 0.95)
# │       │           ├── Build 34|230,20,20,10 (GPM: 90.798, Efficiency: 0.91)
# │       │           │   └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │       │           │       └── Average Efficiency: 96.0%
# │       │           └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │       │               ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │       │               │   └── Average Efficiency: 96.0%
# │       │               └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │       │                   └── Average Efficiency: 96.0%
# │       └── Build 37|230,20,19,14 (GPM: 120.009, Efficiency: 0.90)
# │           ├── Build 36|230,20,19,13 (GPM: 117.427, Efficiency: 0.94)
# │           │   ├── Build 35|230,20,18,13 (GPM: 105.017, Efficiency: 0.95)
# │           │   │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │           │   │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │           │   │   │   │   └── Average Efficiency: 95.6%
# │           │   │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │           │   │   │       └── Average Efficiency: 95.6%
# │           │   │   └── Build 34|230,20,17,13 (GPM: 91.47, Efficiency: 0.92)
# │           │   │       ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │           │   │       │   └── Average Efficiency: 95.5%
# │           │   │       └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │           │   │           └── Average Efficiency: 95.5%
# │           │   └── Build 35|230,20,19,12 (GPM: 104.589, Efficiency: 0.95)
# │           │       ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │           │       │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │           │       │   │   └── Average Efficiency: 95.6%
# │           │       │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │           │       │       └── Average Efficiency: 95.6%
# │           │       └── Build 34|230,20,19,11 (GPM: 89.558, Efficiency: 0.90)
# │           │           ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │           │           │   └── Average Efficiency: 95.5%
# │           │           └── Build 33|230,20,19,10 (GPM: 59.899, Efficiency: 0.79)
# │           │               └── Average Efficiency: 95.5%
# │           └── Build 36|230,20,18,14 (GPM: 112.11, Efficiency: 0.90)
# │               ├── Build 35|230,20,18,13 (GPM: 105.017, Efficiency: 0.95)
# │               │   ├── Build 34|230,20,18,12 (GPM: 97.282, Efficiency: 0.98)
# │               │   │   ├── Build 33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88)
# │               │   │   │   └── Average Efficiency: 95.3%
# │               │   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │               │   │       └── Average Efficiency: 95.3%
# │               │   └── Build 34|230,20,17,13 (GPM: 91.47, Efficiency: 0.92)
# │               │       ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │               │       │   └── Average Efficiency: 95.2%
# │               │       └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │               │           └── Average Efficiency: 95.2%
# │               └── Build 35|230,20,17,14 (GPM: 101.862, Efficiency: 0.92)
# │                   ├── Build 34|230,20,17,13 (GPM: 91.47, Efficiency: 0.92)
# │                   │   ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │                   │   │   └── Average Efficiency: 95.2%
# │                   │   └── Build 33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)
# │                   │       └── Average Efficiency: 95.2%
# │                   └── Build 34|230,20,16,14 (GPM: 89.985, Efficiency: 0.90)
# │                       ├── Build 33|230,20,16,13 (GPM: 66.559, Efficiency: 0.87)
# │                       │   └── Average Efficiency: 95.1%
# │                       └── Build 33|230,20,15,14 (GPM: 64.673, Efficiency: 0.85)
# │                           └── Average Efficiency: 95.1%
# ├── BUILD 39|230,25,18,12 (GPM: 151.755, Efficiency: 0.98)
# │   ├── Build 38|230,25,17,12 (GPM: 135.251, Efficiency: 0.94)
# │   │   ├── Build 37|230,25,17,11 (GPM: 115.973, Efficiency: 0.87)
# │   │   │   ├── Build 36|230,25,17,10 (GPM: 108.479, Efficiency: 0.87)
# │   │   │   │   └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │   │   │   │       └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │   │   │           └── Average Efficiency: 94.6%
# │   │   │   └── Build 36|230,25,16,11 (GPM: 104.123, Efficiency: 0.83)
# │   │   │       ├── Build 35|230,25,15,11 (GPM: 97.534, Efficiency: 0.88)
# │   │   │       │   └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │   │       │       └── Average Efficiency: 94.5%
# │   │   │       └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │   │   │           └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │   │               └── Average Efficiency: 94.4%
# │   │   └── Build 37|230,25,16,12 (GPM: 115.722, Efficiency: 0.87)
# │   │       ├── Build 36|230,25,15,12 (GPM: 107.433, Efficiency: 0.86)
# │   │       │   └── Build 35|230,25,15,11 (GPM: 97.534, Efficiency: 0.88)
# │   │       │       └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │       │           └── Average Efficiency: 94.7%
# │   │       └── Build 36|230,25,16,11 (GPM: 104.123, Efficiency: 0.83)
# │   │           ├── Build 35|230,25,15,11 (GPM: 97.534, Efficiency: 0.88)
# │   │           │   └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │           │       └── Average Efficiency: 94.5%
# │   │           └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │   │               └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │   │                   └── Average Efficiency: 94.4%
# │   └── Build 38|230,25,18,11 (GPM: 130.927, Efficiency: 0.91)
# │       ├── Build 37|230,25,17,11 (GPM: 115.973, Efficiency: 0.87)
# │       │   ├── Build 36|230,25,17,10 (GPM: 108.479, Efficiency: 0.87)
# │       │   │   └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │       │   │       └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │       │   │           └── Average Efficiency: 93.9%
# │       │   └── Build 36|230,25,16,11 (GPM: 104.123, Efficiency: 0.83)
# │       │       ├── Build 35|230,25,15,11 (GPM: 97.534, Efficiency: 0.88)
# │       │       │   └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │       │       │       └── Average Efficiency: 93.7%
# │       │       └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │       │           └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │       │               └── Average Efficiency: 93.6%
# │       └── Build 37|230,25,18,10 (GPM: 113.265, Efficiency: 0.85)
# │           └── Build 36|230,25,17,10 (GPM: 108.479, Efficiency: 0.87)
# │               └── Build 35|230,25,16,10 (GPM: 94.136, Efficiency: 0.85)
# │                   └── Build 34|230,25,15,10 (GPM: 81.095, Efficiency: 0.81)
# │                       └── Average Efficiency: 93.6%
# └── BUILD 39|220,22,20,14 (GPM: 151.518, Efficiency: 0.98)
#     ├── Build 38|220,22,20,13 (GPM: 140.105, Efficiency: 0.98)
#     │   ├── Build 37|220,22,20,12 (GPM: 133.172, Efficiency: 1.00)
#     │   │   ├── Build 36|220,22,19,12 (GPM: 118.004, Efficiency: 0.95)
#     │   │   │   ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#     │   │   │   │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │   │   │   │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │   │   │   │   │   └── Average Efficiency: 97.9%
#     │   │   │   │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │   │   │   │   │       └── Average Efficiency: 97.9%
#     │   │   │   │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#     │   │   │   │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#     │   │   │   │       │   └── Average Efficiency: 97.9%
#     │   │   │   │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │   │   │   │           └── Average Efficiency: 97.9%
#     │   │   │   └── Build 35|220,22,19,11 (GPM: 103.627, Efficiency: 0.94)
#     │   │   │       ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │   │   │       │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │   │       │   │   └── Average Efficiency: 97.9%
#     │   │   │       │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │   │   │       │       └── Average Efficiency: 97.9%
#     │   │   │       └── Build 34|220,22,19,10 (GPM: 86.851, Efficiency: 0.87)
#     │   │   │           └── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │   │               └── Average Efficiency: 97.7%
#     │   │   └── Build 36|220,22,20,11 (GPM: 110.511, Efficiency: 0.89)
#     │   │       ├── Build 35|220,22,19,11 (GPM: 103.627, Efficiency: 0.94)
#     │   │       │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │   │       │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │       │   │   │   └── Average Efficiency: 97.5%
#     │   │       │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │   │       │   │       └── Average Efficiency: 97.5%
#     │   │       │   └── Build 34|220,22,19,10 (GPM: 86.851, Efficiency: 0.87)
#     │   │       │       └── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │       │           └── Average Efficiency: 97.3%
#     │   │       └── Build 35|220,22,20,10 (GPM: 102.086, Efficiency: 0.93)
#     │   │           └── Build 34|220,22,19,10 (GPM: 86.851, Efficiency: 0.87)
#     │   │               └── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │   │                   └── Average Efficiency: 97.3%
#     │   └── Build 37|220,22,19,13 (GPM: 128.724, Efficiency: 0.97)
#     │       ├── Build 36|220,22,19,12 (GPM: 118.004, Efficiency: 0.95)
#     │       │   ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#     │       │   │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │       │   │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │       │   │   │   │   └── Average Efficiency: 97.5%
#     │       │   │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │       │   │   │       └── Average Efficiency: 97.5%
#     │       │   │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#     │       │   │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#     │       │   │       │   └── Average Efficiency: 97.5%
#     │       │   │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │       │   │           └── Average Efficiency: 97.5%
#     │       │   └── Build 35|220,22,19,11 (GPM: 103.627, Efficiency: 0.94)
#     │       │       ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │       │       │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │       │       │   │   └── Average Efficiency: 97.5%
#     │       │       │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │       │       │       └── Average Efficiency: 97.5%
#     │       │       └── Build 34|220,22,19,10 (GPM: 86.851, Efficiency: 0.87)
#     │       │           └── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │       │               └── Average Efficiency: 97.3%
#     │       └── Build 36|220,22,18,13 (GPM: 114.529, Efficiency: 0.92)
#     │           ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#     │           │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#     │           │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#     │           │   │   │   └── Average Efficiency: 97.3%
#     │           │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │           │   │       └── Average Efficiency: 97.3%
#     │           │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#     │           │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#     │           │       │   └── Average Efficiency: 97.3%
#     │           │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │           │           └── Average Efficiency: 97.3%
#     │           └── Build 35|220,22,17,13 (GPM: 102.982, Efficiency: 0.93)
#     │               ├── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#     │               │   ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#     │               │   │   └── Average Efficiency: 97.2%
#     │               │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#     │               │       └── Average Efficiency: 97.2%
#     │               └── Build 34|220,22,16,13 (GPM: 82.09, Efficiency: 0.82)
#     │                   ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#     │                   │   └── Average Efficiency: 97.0%
#     │                   └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#     │                       └── Average Efficiency: 97.0%
#     └── Build 38|220,22,19,14 (GPM: 136.58, Efficiency: 0.95)
#         ├── Build 37|220,22,19,13 (GPM: 128.724, Efficiency: 0.97)
#         │   ├── Build 36|220,22,19,12 (GPM: 118.004, Efficiency: 0.95)
#         │   │   ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#         │   │   │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#         │   │   │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#         │   │   │   │   │   └── Average Efficiency: 96.9%
#         │   │   │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │   │   │   │       └── Average Efficiency: 96.9%
#         │   │   │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#         │   │   │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#         │   │   │       │   └── Average Efficiency: 96.9%
#         │   │   │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │   │   │           └── Average Efficiency: 96.9%
#         │   │   └── Build 35|220,22,19,11 (GPM: 103.627, Efficiency: 0.94)
#         │   │       ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#         │   │       │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#         │   │       │   │   └── Average Efficiency: 96.8%
#         │   │       │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │   │       │       └── Average Efficiency: 96.8%
#         │   │       └── Build 34|220,22,19,10 (GPM: 86.851, Efficiency: 0.87)
#         │   │           └── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#         │   │               └── Average Efficiency: 96.7%
#         │   └── Build 36|220,22,18,13 (GPM: 114.529, Efficiency: 0.92)
#         │       ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#         │       │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#         │       │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#         │       │   │   │   └── Average Efficiency: 96.7%
#         │       │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │       │   │       └── Average Efficiency: 96.7%
#         │       │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#         │       │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#         │       │       │   └── Average Efficiency: 96.7%
#         │       │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │       │           └── Average Efficiency: 96.7%
#         │       └── Build 35|220,22,17,13 (GPM: 102.982, Efficiency: 0.93)
#         │           ├── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#         │           │   ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#         │           │   │   └── Average Efficiency: 96.6%
#         │           │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#         │           │       └── Average Efficiency: 96.6%
#         │           └── Build 34|220,22,16,13 (GPM: 82.09, Efficiency: 0.82)
#         │               ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#         │               │   └── Average Efficiency: 96.4%
#         │               └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#         │                   └── Average Efficiency: 96.4%
#         └── Build 37|220,22,18,14 (GPM: 127.012, Efficiency: 0.95)
#             ├── Build 36|220,22,18,13 (GPM: 114.529, Efficiency: 0.92)
#             │   ├── Build 35|220,22,18,12 (GPM: 105.59, Efficiency: 0.96)
#             │   │   ├── Build 34|220,22,18,11 (GPM: 97.237, Efficiency: 0.98)
#             │   │   │   ├── Build 33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86)
#             │   │   │   │   └── Average Efficiency: 96.5%
#             │   │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#             │   │   │       └── Average Efficiency: 96.5%
#             │   │   └── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#             │   │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#             │   │       │   └── Average Efficiency: 96.5%
#             │   │       └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#             │   │           └── Average Efficiency: 96.5%
#             │   └── Build 35|220,22,17,13 (GPM: 102.982, Efficiency: 0.93)
#             │       ├── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#             │       │   ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#             │       │   │   └── Average Efficiency: 96.5%
#             │       │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#             │       │       └── Average Efficiency: 96.5%
#             │       └── Build 34|220,22,16,13 (GPM: 82.09, Efficiency: 0.82)
#             │           ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#             │           │   └── Average Efficiency: 96.2%
#             │           └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#             │               └── Average Efficiency: 96.2%
#             └── Build 36|220,22,17,14 (GPM: 109.886, Efficiency: 0.88)
#                 ├── Build 35|220,22,17,13 (GPM: 102.982, Efficiency: 0.93)
#                 │   ├── Build 34|220,22,17,12 (GPM: 96.29, Efficiency: 0.97)
#                 │   │   ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#                 │   │   │   └── Average Efficiency: 96.2%
#                 │   │   └── Build 33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83)
#                 │   │       └── Average Efficiency: 96.2%
#                 │   └── Build 34|220,22,16,13 (GPM: 82.09, Efficiency: 0.82)
#                 │       ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#                 │       │   └── Average Efficiency: 96.0%
#                 │       └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#                 │           └── Average Efficiency: 96.0%
#                 └── Build 35|220,22,16,14 (GPM: 95.098, Efficiency: 0.86)
#                     ├── Build 34|220,22,15,14 (GPM: 83.574, Efficiency: 0.84)
#                     │   └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#                     │       └── Average Efficiency: 95.8%
#                     └── Build 34|220,22,16,13 (GPM: 82.09, Efficiency: 0.82)
#                         ├── Build 33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87)
#                         │   └── Average Efficiency: 95.8%
#                         └── Build 33|220,22,15,13 (GPM: 64.113, Efficiency: 0.84)
#                             └── Average Efficiency: 95.8%
#      average_efficiency                                         build_path
# 125              0.9793  33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83...
# 124              0.9793  33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86...
# 127              0.9791  33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83...
# 126              0.9791  33|220,22,16,12 (GPM: 66.253, Efficiency: 0.87...
# 129              0.9787  33|220,22,17,11 (GPM: 63.115, Efficiency: 0.83...
# 128              0.9787  33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86...
# 57               0.9772  33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88...
# 58               0.9772  33|230,20,17,12 (GPM: 51.64, Efficiency: 0.68)...
# 130              0.9771  33|220,22,18,10 (GPM: 65.148, Efficiency: 0.86...
# 61               0.9771  33|230,20,18,11 (GPM: 66.758, Efficiency: 0.88...