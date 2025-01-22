import pandas as pd
import random
import logging
import multiprocessing
import copy


class Gooster:
    """
    Represents a Goo character with attributes like health, attack, speed, dodge, and level.
    """

    def __init__(self, hp=100, atk=10, spd=10, ddg=10, level=1):
        self.hp = hp
        self.atk = atk
        self.spd = spd
        self.ddg = ddg
        self.level = level

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
        for _ in range(1, level):
            self._increment_random_stat()

    def _increment_random_stat(self):
        """
        Randomly increments one of the Goo's stats.
        """
        stat_options = ['hp', 'atk', 'spd', 'ddg']
        chosen_stat = random.choice(stat_options)
        self.increment_stat(chosen_stat)

    def set_stats(self, hp, atk, spd, ddg):
        """
        Applies multiple upgrades to the Goo's stats.
        """
        self.increment_stat("hp", hp)
        self.increment_stat("atk", atk)
        self.increment_stat("spd", spd)
        self.increment_stat("ddg", ddg)

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
        """
        # Determine turn order
        goo1_first = goo1.spd > goo2.spd or (goo1.spd == goo2.spd and random.random() < 0.5)

        # Wrap Goos in BattleInstance
        g1_battle = BattleInstance(goo1)
        g2_battle = BattleInstance(goo2)


        attacker, defender = (g1_battle, g2_battle) if goo1_first else (g2_battle, g1_battle)

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
        ddg_chance = defender.goo.ddg * 0.02
        roll = random.random()

        if roll <= dh_chance:
            # Critical hit
            defender.take_damage(attacker.goo.atk * 2)
        elif roll <= dh_chance + ddg_chance:
            # Attack dodged
            pass
        else:
            # Normal hit
            defender.take_damage(attacker.goo.atk)

    @staticmethod
    def n_battles(goo1, goo2, n=100):
        """
        Simulates n battles between two Goo instances and calculates the win rate for goo1.
        """
        pool = multiprocessing.Pool(processes=4)
        results = pool.starmap(Simulator.battle, [(goo1, goo2) for _ in range(n)])
        pool.close()
        return sum(results) / n

    @staticmethod
    def generate_permutations(level):
        """
        Generates all possible Goo stat combinations up to a given level.
        """
        permutations = []
        for hp in range(level):
            for atk in range(level - hp):
                for spd in range(level - hp - atk):
                    ddg = level - hp - atk - spd
                    goo_instance = Gooster()
                    goo_instance.set_stats(hp, atk, spd, ddg)
                    permutations.append(goo_instance)
        return permutations

    @staticmethod
    def simulate_matrix(perms, n=300):
        """
        Simulates a battle matrix for Goo permutations.
        """
        results = {}
        for attacker in perms:
            results[attacker] = {}
            for defender in perms:
                results[attacker][defender] = Simulator.n_battles(attacker, defender, n)
        return pd.DataFrame(results)


def main():
    """
    Main entry point for the simulation.
    """
    # test_level = 10
    # permutations = Simulator.generate_permutations(test_level)
    # print(f"Generated {len(permutations)} permutations.")

    # battle_results = Simulator.simulate_matrix(permutations, n=150)
    # battle_results.to_csv('g_matrix.csv')
    # print("Simulation completed. Results saved to g_matrix.csv")
    goo = Gooster()
    print(goo)
    goo.rand_level_to(1)
    print(goo)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
