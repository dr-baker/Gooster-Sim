import pandas as pd
import random
import logging
import multiprocessing
# import copy
import itertools

GOO_VALUES = {
    "ultima": 80000,
    "angel2": 40000,
    "angel": 20000,
    "omega": 10000,
    "elem": 5000,
    "level_20": 1000
}
IDOL_MULT = 16

# ENEMY_PROBS = {
#     "shiny":    0.1,
#     "level_20":       0.1,
#     "elem":     0.1,
#     "omega":    0.05,
#     # "angel":    0, # test <30
#     "angel":    0.025, # was halved at one point
#     "sshiny":   0.01,
#     "ninja":    0.01,
# }
# ENEMY_CHECK_ORDER = ['shiny','level_20','elem','omega','angel','ninja','sshiny'] # ~introduction order

###################################################################### GOOSTER DEFINITION
class Gooster:
    """
    Represents a Goo character with attributes like health, attack, speed, dodge, and level.
    """

    def __init__(self, is_elem = True, is_omega = True, is_ultima = True, is_shiny = False, 
                 boss_type=None, stats=None, upgrade_counts=None, level=None):
        self.hp = 100
        self.atk = 10
        self.spd = 10
        self.ddg = 10
        self.is_elem = is_omega or is_elem ## relevant as attacker
        self.is_omega = is_omega ## relevant as attacker
        self.is_ultima = is_ultima ## relevant as attacker
        self.boss_type = boss_type ## relevant for goo value
        self.is_shiny = is_shiny ## relevant for goo value
        
        if (stats or upgrade_counts) and level:
            raise Exception("Invalid input. Level input expects no stats or upgrades.")
        
        if stats:
            self.set_stats(*stats)
        if upgrade_counts:
            self.apply_upgrades(*upgrade_counts)
        if level:
            self.rand_level_to(level)

    def set_stats(self, hp, atk, spd, ddg):
        """
        Applies multiple upgrades to the Goo's stats.
        """
        self.hp = hp
        self.atk = atk
        self.spd = spd
        self.ddg = ddg

    @property
    def level(self):
        return 1 + int((self.hp-100)/10) + self.atk-10 + self.spd-10 + self.ddg-10
    
    @property
    def goo(self):
        """
        Returns goo value when this gooster is killed as an enemy.
        """
        base_value = self.level*10
        boss_bonus = GOO_VALUES[self.boss_type] if self.boss_type in GOO_VALUES else 0
        mult = 2 if self.is_shiny else 1
        mult *= IDOL_MULT
        return (base_value+boss_bonus)*mult
            
    def apply_upgrades(self, hp, atk, spd, ddg):
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
        # return f'{self.level}|{int((self.hp - 100) / 10)},{self.atk - 10},{self.spd - 10},{self.ddg - 10}'
        return f'{self.level}|{self.hp},{self.atk},{self.spd},{self.ddg}'

###################################################################### SIMULATOR

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
        - First hit can not crit or be dodged
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

        first_hit = True # First hit has special logic (no dodge/crit chance)
        # Battle loop
        while True: 
            Simulator.attack(attacker, defender, first_hit)
            if defender.current_hp <= 0:
                return attacker == g1_battle
            
            first_hit = False

            Simulator.attack(defender, attacker, first_hit)
            if attacker.current_hp <= 0:
                return defender == g1_battle


    @staticmethod
    def attack(attacker, defender, first_hit):
        """
        Simulates an attack from one Goo to another.
        """
        # Special logic for first hit (no dodge/crit chance)
        if first_hit:
            defender.take_damage(attacker.goo.atk)
            return 'first hit'

        dh_chance = max(0, (attacker.goo.spd - defender.goo.ddg) * 0.05)
        ddg_chance = max(0, (defender.goo.ddg * 0.02) - (attacker.goo.spd * 0.01) )
        roll = random.random()

        if roll <= dh_chance:
            # Critical hit
            defender.take_damage(attacker.goo.atk * 2) # Note no dodge roll
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
            goo_instance = Gooster(upgrade_counts=stats)
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
    def get_random_enemy(level, is_elem=False,is_omega=False,shiny_streak=-1,is_ultima=False):
        """
        Returns a random enemy based on the player's level.
        Bosses become available every 5 levels, with their appearance rates increasing.
        """
        ## Special case, no random
        if shiny_streak == 8:
            return "angel2", Simulator.create_enemy(level, "angel2")
        elif shiny_streak > -1:
            return "angelshiny", Simulator.create_enemy(level, "angelshiny")

        bosses = {
            35: ("ultima", 1/320),
            30: ("angel", 1/160),
            25: ("omega", 1/320),
            20: ("elem", 1/160),
            19: ("level_20", 1/160),
        }
        fixed_bosses = {
            19: ("ninja", 1/80),
            20: ("sshiny", 1/160),
            25: ("shiny", 1/10),
        }
        
        # Determine available bosses
        available_bosses = {lvl: boss for lvl, boss in bosses.items() if level >= lvl}
        
        boss_probabilities = {}
        # Adjust probabilities
        for start_level, (boss_name, base_rate) in available_bosses.items():
            # Calculate rate increase based on levels since unlock (caps at 5 levels)
            levels_since_unlock = min(level - start_level, 4)  # 0 to 4
            # Bonus for matching type
            if boss_name == 'omega' and is_omega:
                levels_since_unlock = 4
            if boss_name == 'elem' and is_elem:
                levels_since_unlock = 4
            if boss_name == 'angel' and is_omega: # TODO
                levels_since_unlock = 4
            if boss_name == 'ultima' and is_ultima: # TODO
                levels_since_unlock = 4
            boss_probabilities[boss_name] = base_rate * (2 ** levels_since_unlock)
        for start_level, (boss_name, base_rate) in fixed_bosses.items():
            boss_probabilities[boss_name] = base_rate
        
        # Do random
        roll = random.random()
        cumulative_probability = 0
        for boss_name, probability in boss_probabilities.items():
            cumulative_probability += probability
            if roll < cumulative_probability:
                return boss_name, Simulator.create_enemy(level, boss_name)
        return 'normal', Simulator.create_enemy(level, 'normal')

    @staticmethod
    def create_enemy(attacker_level, enemy_type):
        if enemy_type == 'ninja':
            return Gooster(stats=[100,10,10,59], boss_type=enemy_type)
        if enemy_type == 'sshiny':
            return Gooster(level=attacker_level+2) ## no shiny bonus on these
        if enemy_type == 'ultima':
            gooster = Gooster(stats=[240,24,10,10], boss_type=enemy_type)
            gooster.rand_level_to(40)
            return gooster
        if enemy_type == 'angel' or enemy_type == 'angel2':
            # Assume angel is created as 240/24/10/10 with random distribution up to level+5
            gooster = Gooster(stats=[240,24,10,10], boss_type=enemy_type)
            gooster.rand_level_to(35)
            return gooster
        if enemy_type == 'omega':
            return Gooster(level=30, boss_type=enemy_type)
        if enemy_type == 'elem':
            return Gooster(level=25, boss_type=enemy_type)
        if enemy_type == 'level_20':
            return Gooster(level=20, boss_type=enemy_type)
        if enemy_type == 'shiny':
            return Gooster(level=attacker_level+1, is_shiny=True)
        if enemy_type == 'angelshiny':
            return Gooster(level=35, is_shiny=True)
        
        # Otherwise return a regular gooster in range [level-3, level]
        opponent_level = random.choice(range(int(attacker_level) - 3, int(attacker_level) + 1))
        return Gooster(level=opponent_level)




    @staticmethod
    def simulate_random_samples(attackers, defender_stats=None, n=3000):
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

            is_shiny_streak = -1
            for _ in range(n):
                if defender_stats:
                    enemy_type = 'normal'
                    enemy = Gooster(stats=defender_stats)
                else: ## Get random enemy if none is passed
                    enemy_type, enemy = Simulator.get_random_enemy(
                        attacker.level, is_elem=attacker.is_elem, 
                        is_omega=attacker.is_omega, shiny_streak=is_shiny_streak, 
                        is_ultima=attacker.is_ultima
                    )

                # Ensure the enemy_type key exists in results[attacker]
                if enemy_type not in results[attacker]:
                    results[attacker][enemy_type] = {'n': 0, 'wins': 0, 'goo':0}

                # Update counts
                results[attacker][enemy_type]['n'] += 1
                battle_won = Simulator.battle(attacker, enemy)
                results[attacker][enemy_type]['wins'] += battle_won
                if battle_won:
                    results[attacker][enemy_type]['goo'] += enemy.goo

                # Start or continue shiny streak. Else end it.
                if battle_won and (enemy_type=='angel' or enemy_type=='angelshiny'):
                    is_shiny_streak += 1
                else:
                    is_shiny_streak = -1

        return pd.DataFrame(results)
    
    @staticmethod
    def humanize_number(value):
        """Formats large numbers into a readable format (e.g., 2K, 12M)."""
        return float(f"{value / 1_000_000:.1f}")
        # if value >= 1_000_000:
        #     return f"{value / 1_000_000:.1f}M"
        # elif value >= 1_000:
        #     return f"{value / 1_000:.1f}K"
        # return str(value)

    @staticmethod
    def convert_to_winrate_df(results_df):
        """
        Converts the 3D results DataFrame into a 2D DataFrame with win rates and goo values.
        """
        return pd.DataFrame({
            attacker: {
                enemy: [
                    round(stats['wins'] / stats['n'], 3),   # Winrate
                    stats['n'],                              # Number of battles
                    Simulator.humanize_number(stats['goo'])           # Human-readable goo
                ] if stats['n'] else None
                for enemy, stats in enemy_results.items()
            }
            for attacker, enemy_results in results_df.to_dict().items()
        }).T  # Transpose to make attackers rows and enemies columns
    
    @staticmethod
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
                    goo_matrix[attacker][enemy] = Simulator.humanize_number(stats['goo'])
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


