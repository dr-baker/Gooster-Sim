import random
import pandas as pd
from config import GOO_VALUES, IDOL_MULT

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


