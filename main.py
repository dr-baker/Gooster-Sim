import logging
from gooster import Gooster
from simulator import Simulator

def test_angel_gen():
    for i in range(0,20):
        g = Simulator.create_enemy(30, 'angel')
        print(g)

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
    df2 = Simulator.convert_to_winrate_df(df)
    file_name = f'g_sample.csv'
    df2.to_csv(file_name)
    Simulator.export_results_to_excel(df)

    

def sim_all_good_builds():
    required_upgrades = [5,10,5,0]
    target_level = 39
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
    Simulator.export_results_to_excel(df)

def sim_1v1(attacker_stats,defender_stats,n):
    attacker = Gooster(stats=attacker_stats)
    attackers = [attacker]
    print(attacker)
    df = Simulator.simulate_random_samples(attackers, defender_stats=defender_stats, n=n)
    print(df)



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # sim_all_builds_cross_product()
    # test_angel_gen()
    # sim_set_builds()
    sim_all_good_builds()
    # sim_1v1([100,10,11,10],[110,10,10,10],100000)
    # sim_1v1([140,14,11,10],[140,13,10,12],100000)

