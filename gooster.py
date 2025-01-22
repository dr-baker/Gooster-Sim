import pandas as pd
import random
import logging
import multiprocessing



class goo:
    def __init__(self, hp=100, atk=10, spd=10, ddg=10, level=1):
        self.hp = hp
        self.atk = atk
        self.spd = spd
        self.ddg = ddg
        self.level = level
        self.bhp = hp

    def rand_level(self, level):
        for i in range(1, level):
            self.increment_random_stat()
             
    def increment_random_stat(self):
        random_stat = random.choice(['increment_hp', 'increment_spd', 'increment_atk', 'increment_ddg'])
        getattr(self, random_stat)()

    def increment_hp(self):
        self.hp += 10
        self.level += 1

    def increment_atk(self):
        self.atk += 1
        self.level += 1

    def increment_spd(self):
        self.spd += 1
        self.level += 1

    def increment_ddg(self):
        self.ddg += 1
        self.level += 1
        
    def reset_bhp(self):
        self.bhp = self.hp
        
    def updgrade(self, hp, atk, spd, ddg):
        for _ in range(hp):
            self.increment_hp()
        for _ in range(atk):  
            self.increment_atk()
        for _ in range(spd):
            self.increment_spd()
        for _ in range(ddg):
            self.increment_ddg()

    def to_dataframe_row(self):
        return pd.DataFrame([{
            'level': self.level,
            'hp': self.hp,
            'atk': self.atk,
            'spd': self.spd,
            'ddg': self.ddg
        }])
        
    def to_dict(self):
        return {
            'level': self.level,
            'hp': self.hp,
            'atk': self.atk,
            'spd': self.spd,
            'ddg': self.ddg
        }
        
    def __copy__(self, memo):
        return goo(self.hp, self.atk, self.spd, self.ddg, self.level)
    
    def __str__(self):
        return f'{self.level}|{int((self.hp-100)/10)},{self.atk-10},{self.spd-10},{self.ddg-10}'
        # return f'{self.level}|{self.hp},{self.atk},{self.spd},{self.ddg}'
        
class sim:
    def battle(goo1, goo2):
        '''Simulate a battle between two goos. Returns true if goo1 wins, false if goo2 wins. Handles goo1==goo2.'''
        goo1_first = False
        ## Get turn order
        if goo1.spd > goo2.spd:
            f, s = goo1, goo2
            goo1_first = True
        elif goo1.spd == goo2.spd:
            ## Speed tie, assign randomly
            r = random.random()
            if r < 0.5:
                f,s = goo1, goo2
                goo1_first = True
            else:
                f,s = goo2, goo1
        else:
            f,s = goo2, goo1
            
        ## Use copies to avoid issues and keep track for winner
        f,s = f.__copy__(None), s.__copy__(None)

        winner = None
        logging.debug(f'{goo1} vs {goo2}')
        while True:
            logging.debug(f'{f.bhp} | {s.bhp}')
            ## First attacker
            sim.attack(f, s)
            if s.bhp <= 0:
                winner = f
                break
            ## Second attacker
            sim.attack(s, f)
            if f.bhp <= 0:
                winner = s
                break
        
        return (winner == f and goo1_first) or (winner == s and not goo1_first)
    
    def attack(attacker, defender):
        dh_chance =  max( 0, (attacker.spd - defender.ddg) * 0.05 )
        ddg_chance =  defender.ddg * 0.02
        logging.debug(f'{attacker.spd} | {defender.ddg} | {dh_chance} | {ddg_chance}')
        
        ## dh_chance chance of crit, ddg_chance chance of dodge, else normal attack
        r = random.random()
        if r <= dh_chance:
            logging.debug(f'crit')
            defender.bhp -= attacker.atk * 2 ## crit
        elif r + dh_chance < ddg_chance:
            logging.debug(f'dodge')
            pass ## dodge
        else:
            logging.debug(f'hit')
            defender.bhp -= attacker.atk ## normal attack
        

    def n_battles(goo1, goo2, n):
        wins = 0
        losses = 0
        
        for _ in range(n):
            wins += sim.battle(goo1, goo2)

        win_rate = wins / n
        return n, wins, win_rate
    
    def n_battles2(goo1, goo2, n):
        wins = 0
        losses = 0
        
        pool = multiprocessing.Pool(16)
        results = pool.starmap(sim.battle, [(goo1, goo2) for _ in range(n)])
        wins = sum(results)

        win_rate = wins / n
        return n, wins, win_rate
    
    def gen_permutations(test_level):
        ## create a list of goos with all possible stats
        perms = []
        stats = [0,0,0,0]
        for h in range(test_level):
            stats[0] = h
            for a in range(test_level - sum(stats)):
                stats[1] = a
                for s in range(test_level - sum(stats)):
                    stats[2] = s
                    for d in range(test_level - sum(stats)):
                        stats[3] = d
                        g = goo()
                        g.updgrade(stats[0], stats[1], stats[2], stats[3])
                        perms.append(g)
                        stats[3]=0
                    stats[2]=0
                stats[1]=0
        return perms
    
    def create_matrix(perms, n=300, test_level = None, skip_low_levels = True, skip_duplicates = True):
        ## Simulate n battles for each perm vs each perm 
        
        ## if passed a test_level, only test goos of that level
        if test_level is None:
            matrix = {g: {g: None for g in perms} for g in perms} ## test all levels
        else:
            matrix = {g: {g: None for g in perms} for g in perms if g.level == test_level} ## only test one level attacker
        
        i=0
        for g1 in matrix.keys():
            print(f'{i}\tSimulating {g1} vs {len(matrix[g1])}')
            for g2 in matrix[g1].keys():
                if skip_low_levels and g2.level < g1.level-3:
                    logging.info(f'{g2} is too low level for {g1}')
                    continue
                if skip_duplicates and g2 in matrix and matrix[g2][g1] is not None:
                    logging.info(f'{g2} vs {g1} already simulated')
                    matrix[g1][g2] = 1 - matrix[g2][g1]
                    continue
                
                # print(f'Simulating {g1} vs {g2}')
                n, wins, win_rate = sim.n_battles(g1, g2, 300)
                
                matrix[g1][g2] = win_rate
            i += 1
        ## Rows are attacker, columns are defender
        return pd.DataFrame(matrix).T
    
    def create_rows(perms, n=300, test_level=None, skip_low_levels=True, skip_duplicates=True):
        # Initialize DataFrame with MultiIndex
        index = pd.MultiIndex.from_tuples([], names=['attacker', 'defender'])
        df = pd.DataFrame(index=index, columns=['win_rate'])
        
        i = 0
        for g1 in perms:
            if test_level is not None and g1.level != test_level:
                continue
            
            if skip_low_levels:
                viable_opponents = [g for g in perms if g.level >= g1.level - 3 and g.level <= g1.level]
                logging.info(f'{i}/{len(perms)}\tRemoving {len(perms) - len(viable_opponents)} opponents not in level range')
            else:
                viable_opponents = perms
            
            print(f'{i}/{len(perms)}\tSimulating {g1} vs {len(viable_opponents)}')
            
            dupes = 0
            for g2 in viable_opponents:
                if skip_duplicates and (g2, g1) in df.index:
                    df.loc[(g1, g2), 'win_rate'] = 1 - df.loc[(g2, g1), 'win_rate']
                    dupes += 1
                    continue
                n, wins, win_rate = sim.n_battles(g1, g2, n)
                
                df.loc[(g1, g2), 'win_rate'] = win_rate
            if skip_duplicates: 
                logging.info(f'Skipped {dupes} duplicates')
            i += 1
        
        df.sort_index(level=[0,1], ascending=[False, False], inplace=True)
        return df

    
def main():
    test_level = 15
    perms = sim.gen_permutations(test_level)         
    print(f'Number of permutations: {len(perms)}')

    # df = sim.create_matrix(perms, n=300, test_level = None, skip_low_levels = True, skip_duplicates = True)
    df = sim.create_rows(perms, n=150, test_level = 15, skip_low_levels = True, skip_duplicates = True)
    
    try:
        df.to_csv('g_matrix.csv')
    except Exception as e:
        logging.error(e)
        input()
        df.to_csv('g_matrix.csv')
    print(df)

def test():
    g1 = goo(100,10,10,19,10)
    # g2 = goo(190,10,10,10,10)
    g2 = g1
    print(sim.battle(g1, g2))
    print(sim.battle(g1, g2))
    print(sim.battle(g1, g2))
    print(sim.battle(g1, g2))
    print(sim.battle(g1, g2))
    print(sim.battle(g1, g2))
    print(sim.n_battles(g1, g2, 20))
 
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    # test()
    main()
    
    