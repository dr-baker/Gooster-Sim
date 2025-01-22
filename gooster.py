import pandas as pd
import random
import logging
import multiprocessing
# import copy
import itertools

GOO_VALUES = {
    "angel": 20000,
    "omega": 10000,
    "elemental": 5000,
    "level_20": 1000
}

class Gooster:
    """
    Represents a Goo character with attributes like health, attack, speed, dodge, and level.
    """

    def __init__(self, stat_upgrades=None, level=None):
        self.hp = 100
        self.atk = 10
        self.spd = 10
        self.ddg = 10
        self.level = 1
        # TODO
        self.is_elem = True
        self.is_omega = True
        
        if stat_upgrades and level:
            raise Exception("Invalid input. Init goosters with stats array and level will be set automatically. \
                            Assign with level to assign stats randomly. Do not set both.")
        if stat_upgrades:
            self.set_stats(*stat_upgrades)
        if level:
            self.rand_level_to(level)
            
    def set_stats(self, hp, atk, spd, ddg):
        """
        Applies multiple upgrades to the Goo's stats.
        """
        self.increment_stat("hp", hp)
        self.increment_stat("atk", atk)
        self.increment_stat("spd", spd)
        self.increment_stat("ddg", ddg)

    def increment_stat(self, stat_name, amount=1):
        """
        Increments a specific stat by the given amount.
        """
        if stat_name == "hp":
            self.hp += 10 * amount
        elif stat_name == "atk":
            self.atk += amount
        elif stat_name == "spd":
            self.spd += amount
        elif stat_name == "ddg":
            self.ddg += amount
        self.level += amount

    def rand_level_to(self, level):
        """
        Randomly assigns stat upgrades to reach a given level.
        """
        for _ in range(0, level-self.level):
            self._increment_random_stat()

    def _increment_random_stat(self):
        """
        Randomly increments one of the Goo's stats.
        """
        stat_options = ['hp', 'atk', 'spd', 'ddg']
        chosen_stat = random.choice(stat_options)
        self.increment_stat(chosen_stat)

    def to_dataframe_row(self):
        """
        Converts the Goo's attributes into a DataFrame row.
        """
        return pd.DataFrame([self.to_dict()])

    def to_dict(self):
        """
        Converts the Goo's attributes into a dictionary.
        """
        return {
            'level': self.level,
            'hp': self.hp,
            'atk': self.atk,
            'spd': self.spd,
            'ddg': self.ddg
        }

    def __str__(self):
        """
        Returns a string representation of the Goo.
        """
        return f'{self.level}|{int((self.hp - 100) / 10)},{self.atk - 10},{self.spd - 10},{self.ddg - 10}'


class BattleInstance:
    """
    A battle wrapper that maintains a Goo's battle-specific state without modifying the original object.
    """

    def __init__(self, goo):
        self.goo = goo
        self.current_hp = goo.hp

    def take_damage(self, damage):
        """
        Reduces the current HP during a battle.
        """
        self.current_hp -= damage
        return self.current_hp > 0


class Simulator:
    """
    Handles simulation logic for battles between Goo instances.
    """

    @staticmethod
    def battle(goo1, goo2):
        """
        Simulates a battle between two Goo instances.
        Returns True if goo1 wins, False if goo2 wins.

        Assumptions:
        - Faster gooster gets a free hit start of battle that can not crit
        - Crit fully bypasses dodge
        - Win is evaluated after each damage event and not in rounds (no ties)
        - In speed tie: first gooster is decided randomly at start of battle
        """
        # Determine turn order
        goo1_first = goo1.spd > goo2.spd or (goo1.spd == goo2.spd and random.random() < 0.5)

        # Wrap Goos in BattleInstance
        g1_battle = BattleInstance(goo1)
        g2_battle = BattleInstance(goo2)

        attacker, defender = (g1_battle, g2_battle) if goo1_first else (g2_battle, g1_battle)

        # Free hit
        defender.take_damage(attacker.goo.atk)
        if defender.current_hp <= 0:
            return attacker == g1_battle

        # Battle loop
        while True:
            Simulator.attack(attacker, defender)
            if defender.current_hp <= 0:
                return attacker == g1_battle

            Simulator.attack(defender, attacker)
            if attacker.current_hp <= 0:
                return defender == g1_battle

    @staticmethod
    def attack(attacker, defender):
        """
        Simulates an attack from one Goo to another.
        """
        dh_chance = max(0, (attacker.goo.spd - defender.goo.ddg) * 0.05)
        ddg_chance = max(0, (defender.goo.ddg * 0.02) - (attacker.goo.spd * 0.01) )
        roll = random.random()

        if roll <= dh_chance:
            # Critical hit
            defender.take_damage(attacker.goo.atk * 2)
            return 'crit'
        elif roll <= dh_chance + ddg_chance:
            # Attack dodged
            return 'dodge'
        else:
            # Normal hit
            defender.take_damage(attacker.goo.atk)
            return 'hit'

    # @staticmethod
    # def n_battles(goo1, goo2, n=100):
    #     """
    #     Simulates n battles between two Goo instances and calculates the win rate for goo1.
    #     """
    #     pool = multiprocessing.Pool(processes=4)
    #     results = pool.starmap(Simulator.battle, [(goo1, goo2) for _ in range(n)])
    #     pool.close()
    #     return sum(results) / n
    
    @staticmethod
    def n_battles(goo1, goo2, n=100):
        wins = 0
        losses = 0
        
        for _ in range(n):
            wins += Simulator.battle(goo1, goo2)

        win_rate = wins / n
        return n, wins, win_rate
    
    @staticmethod
    def _partitions(n, k):
        # https://stackoverflow.com/questions/28965734/general-bars-and-stars
        for c in itertools.combinations(range(n+k-1), k-1):
            yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

    @staticmethod
    def generate_permutations(level):
        """
        Generates all possible Goo stat combinations up to a given level.
        """
        permutations = []
        for stats in Simulator._partitions(level,4):
            goo_instance = Gooster(stat_upgrades=stats)
            permutations.append(goo_instance)
        return permutations

    @staticmethod
    def simulate_matrix(attackers, defenders, n=300):
        """
        Simulates a battle matrix for Goo permutations.
        """
        results = {}
        for attacker in attackers:
            results[attacker] = {}
            for defender in defenders:
                results[attacker][defender] = Simulator.n_battles(attacker, defender, n)
        return pd.DataFrame(results)
    
    @staticmethod
    def get_random_enemy(attacker_level, is_elem=True, is_omega=True):
        """
        Assumptions:
        - Using fairly small sample size for rates, round up to nearest whole %
        - Assume rarest options are top of the waterfall
        """
        roll = random.random()
        enemy_probabilities = [
            ("sshiny", roll < 0.005 and attacker_level >= 25),
            ("ninja", roll < 0.01 and attacker_level >= 25),
            ("angel", roll < 0.02 and is_omega and attacker_level >= 30),
            ("omega", roll < 0.35 and is_elem and attacker_level >= 25),
            ("elemental", roll < 0.8 and attacker_level >= 20),
            ("level_20", roll < 0.10 and attacker_level >= 20),
            ("shiny", roll < 0.1)
        ]

        # Return first matching enemy
        for enemy, condition in enemy_probabilities:
            if condition:
                return enemy, Simulator.get_enemy_by_type(attacker_level, enemy)
            
        # Default enemy type if no special conditions are met
        return 'normal', Simulator.get_enemy_by_type(attacker_level, 'normal')  
    
    @staticmethod
    def get_enemy_by_type(attacker_level, enemy_type):
        if enemy_type == 'ninja':
            return Gooster(stat_upgrades=[0,0,0,49])
        if enemy_type == 'sshiny':
            return Gooster(level=attacker_level+2)
        if enemy_type == 'angel':
            # Assume angel is created as 240/24/10/10 with random distribution up to level+5
            gooster = Gooster(stat_upgrades=[14,14,0,0])
            gooster.rand_level_to(attacker_level+5)
            return gooster
        if enemy_type == 'omega':
            return Gooster(level=30)
        if enemy_type == 'elemental':
            return Gooster(level=25)
        if enemy_type == 'level_20':
            return Gooster(level=20)
        if enemy_type == 'shiny':
            return Gooster(level=attacker_level+1)
        
        # Otherwise return a regular gooster in range [level-3, level]
        opponent_level = random.choice( range(attacker_level-3,attacker_level+1) )
        return Gooster(level=opponent_level)




    @staticmethod
    def simulate_random_samples(attackers, n=3000):
        """
        Simulates battles between attackers and randomly generated enemies.

        Parameters:
        - attackers (list): List of attacker objects.
        - n (int): Number of simulations per attacker (default: 3000).

        Returns:
        - pd.DataFrame: A summary of the simulation results.
        """
        results = {}

        for attacker in attackers:
            results[attacker] = {}

            for _ in range(n):
                enemy_type, enemy = Simulator.get_random_enemy(
                    attacker.level, is_elem=attacker.is_elem, is_omega=attacker.is_omega
                )

                # Ensure the enemy_type key exists in results[attacker]
                if enemy_type not in results[attacker]:
                    results[attacker][enemy_type] = {'n': 0, 'wins': 0, 'goo':0}

                # Update counts
                results[attacker][enemy_type]['n'] += 1
                battle_won = Simulator.battle(attacker, enemy)
                results[attacker][enemy_type]['wins'] += battle_won
                if battle_won:
                    results[attacker][enemy_type]['goo'] += GOO_VALUES[enemy_type]

        return pd.DataFrame(results)
    
    @staticmethod
    def convert_to_winrate_df(results_df):
        """
        Converts the 3D results DataFrame into a 2D DataFrame with win rates.
        """
        return pd.DataFrame({
            attacker: {enemy: [round(stats['wins'] / stats['n'],3), stats['n']] if stats['n'] else None 
                    for enemy, stats in enemy_results.items()}
            for attacker, enemy_results in results_df.to_dict().items()
        }).T  # Transpose to make attackers rows and enemies columns




def test_angel_gen():
    for i in range(0,20):
        print(Simulator.get_enemy_by_type(30, 'angel'))

def main():
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

def main2():
    tommy = Gooster(stat_upgrades=[9,10,0,10])
    tomas = Gooster(stat_upgrades=[12,10,1,9])
    pirate = Gooster(stat_upgrades=[1,15,16,0])
    attackers = [tommy, tomas,pirate]

    df = Simulator.simulate_random_samples(attackers, 10000)
    df = Simulator.convert_to_winrate_df(df)
    file_name = f'g_sample.csv'
    df.to_csv(file_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # main()
    # test_angel_gen()
    main2()
