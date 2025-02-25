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

    for attacker, enemy_results in results_df.to_dict().items():
        winrate_matrix[attacker] = {}
        battle_count_matrix[attacker] = {}
        goo_matrix[attacker] = {}

        for enemy, stats in enemy_results.items():
            if isinstance(stats, dict) and stats['n']:  # Avoid division by zero
                winrate_matrix[attacker][enemy] = round(stats['wins'] / stats['n'], 3)
                battle_count_matrix[attacker][enemy] = stats['n']
                goo_matrix[attacker][enemy] = humanize_number(stats['goo'])
            else:
                winrate_matrix[attacker][enemy] = None
                battle_count_matrix[attacker][enemy] = None
                goo_matrix[attacker][enemy] = None

    # Convert to DataFrame
    winrate_df = pd.DataFrame(winrate_matrix).T
    battle_count_df = pd.DataFrame(battle_count_matrix).T
    goo_df = pd.DataFrame(goo_matrix).T

    # Export to Excel
    with pd.ExcelWriter(filename) as writer:
        winrate_df.to_excel(writer, sheet_name="Win Rates")
        battle_count_df.to_excel(writer, sheet_name="Battle Counts")
        goo_df.to_excel(writer, sheet_name="Goo Earnings")

    print(f"Results exported to {filename}")


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


def sim_all_good_builds():
    required_upgrades = [5,10,5,0]
    target_level = 31
    remaining_levels = target_level - sum(required_upgrades) - 1
    ## get all combos but only include attack ends with 2 or 5 also total attack <=35
    next_upgrades = [p for p in Simulator._partitions(remaining_levels, 4) if ((p[1] % 10 == 2 or p[1] % 5 == 0)) ]
    print(len(next_upgrades))

    attackers = []
    for upgrade in next_upgrades:
        g = Gooster(upgrade_counts=required_upgrades)
        g.apply_upgrades(*upgrade)
        g.is_omega = True
        g.is_ultima = True
        attackers.append(g)

    # required_upgrades = [5,10,0,0]
    # target_level = 34
    # remaining_levels = target_level - sum(required_upgrades) - 1
    # ## get all combos but only include attack ends with 2 or 5 also total attack <=35
    # next_upgrades = [p for p in Simulator._partitions(remaining_levels, 4) if ((p[1] % 10 == 2 or p[1] % 5 == 0)) ]
    # print(len(next_upgrades))

    # for upgrade in next_upgrades:
    #     g = Gooster(upgrade_counts=required_upgrades)
    #     g.apply_upgrades(*upgrade)
    #     g.is_omega = True
    #     attackers.append(g)
    
    df = Simulator.simulate_random_samples(attackers, n=5000)
    return df

def sim_1v1(attacker_stats,defender_stats,n):
    attacker = Gooster(stats=attacker_stats)
    attackers = [attacker]
    print(attacker)
    df = Simulator.simulate_random_samples(attackers, defender_stats=defender_stats, n=n)
    return df



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # sim_all_builds_cross_product()
    # test_angel_gen()
    # sim_set_builds()
    df = sim_all_good_builds()
    # sim_1v1([100,10,11,10],[110,10,10,10],100000)
    # sim_1v1([140,14,11,10],[140,13,10,12],100000)
    export_results_to_excel(df)

