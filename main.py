import logging
import pandas as pd
import openpyxl
from gooster import Gooster
from simulator import Simulator
from config import GOO_VALUES,IDOL_MULT

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
    
    df = Simulator.simulate_random_samples(attackers, n=5000)
    return df

def sim_1v1(attacker_stats,defender_stats,n):
    attacker = Gooster(stats=attacker_stats)
    attackers = [attacker]
    print(attacker)
    df = Simulator.simulate_random_samples(attackers, defender_stats=defender_stats, n=n)
    return df



if __name__ == "__main__":
    filename="simulation_results_v2.xlsx"

    logging.basicConfig(level=logging.WARNING)
    # sim_all_builds_cross_product()
    # test_angel_gen()
    # sim_set_builds()
    level = 31
    df = get_v2_df(sim_all_good_builds(level=level))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level, bosstype='angel'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    level = 32
    df = get_v2_df(sim_all_good_builds(level=level))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level, bosstype='angel'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    level = 33
    df = get_v2_df(sim_all_good_builds(level=level))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level, bosstype='angel'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))
    df = get_v2_df(sim_all_good_builds(level=level,  bosstype='ultima'))
    print(df[['Attacker', 'Total Goo', 'GPM']].head(n=10).to_string(index=False))

    # sim_1v1([100,10,11,10],[110,10,10,10],100000)
    # sim_1v1([140,14,11,10],[140,13,10,12],100000)

    # output_df.to_excel(filename, index=False)
    # print(f"Results exported to {filename}")

# Angel builds
# 31: 210,20,18,11
# 32: 210,20,18,12
# 33: 220,20,18,12