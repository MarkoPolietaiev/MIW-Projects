import numpy as np

#  Chance of meeting an enemy or search more
t1 = ["Alice", "John", "Florence", "Search"]
p_t1 = [0.2, 0.5, 0.1, 0.2]


#  Chance of fight or run away
t2 = ["Start fighting", "Run"]
p_t2 = [[0.7, 0.3],  # Fight/Run for Alice
        [1, 0],  # Fight/Run for John
        [0.8, 0.2]]  # Fight/Run for Florence


#  Chance of win,loose or draw with each enemy in fight
t3 = ["Win", "Loose", "Draw"]
p_t3 = [[0.8, 0.1, 0.1],  # Win/Loose/Draw for Alice
        [0.25, 0.25, 0.5],  # Win/Loose/Draw for John
        [0.005, 0.98, 0.015]]  # Win/Loose/Draw for Florence


#  Chance of run away successfully or loose
t4 = ["Run Away", "Loose"]
p_t4 = [0.7, 0.3]  # Successfully Run Away/Loose


print("How many games you want to simulate?")
games_number = int(input())


def game(number):
    counter = 1
    while number >= counter:
        i = 1
        enemy = np.random.choice(t1, p=p_t1)  # pick initial enemy
        print("\n")
        print("Game #" + str(counter))
        print("-------------------------------")
        print(str(i) + ": " + "I am searching for enemy")
        run_simulation(enemy, i+1)
        counter += 1


def run_simulation(enemy, i):
    while True:  # iterate till loose
        if enemy == "Alice":
            print(str(i) + ": " + "I found " + enemy)
            i = i + 1
            activity = np.random.choice(t2, p=p_t2[0])  # fight or run
            if activity == "Start fighting":
                print(str(i) + ": " + "I fight against " + enemy)
                result = np.random.choice(t3, p=p_t3[0])  # result of fight
                i = i + 1
                if result == "Win":
                    won(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif result == "Loose":
                    lost(i)
                    break
                elif result == "Draw":
                    draw(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
            elif activity == "Run":  # result of running away
                run_result = np.random.choice(t4, p=p_t4)
                i = i + 1
                if run_result == "Run Away":
                    ran_away(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif run_result == "Loose":
                    lost(i)
                    break
        elif enemy == "John":
            print(str(i) + ": " + "I found " + enemy)
            i = i + 1
            activity = np.random.choice(t2, p=p_t2[1])
            if activity == "Start fighting":
                print(str(i) + ": " + "I fight against " + enemy)
                result = np.random.choice(t3, p=p_t3[1])
                i = i + 1
                if result == "Win":
                    won(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif result == "Loose":
                    lost(i)
                    break
                elif result == "Draw":
                    draw(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
            elif activity == "Run":
                run_result = np.random.choice(t4, p=p_t4)
                i = i + 1
                if run_result == "Run Away":
                    ran_away(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif run_result == "Loose":
                    lost(i)
                    break
        elif enemy == "Florence":
            print(str(i) + ": " + "I found " + enemy)
            i = i + 1
            activity = np.random.choice(t2, p=p_t2[2])
            if activity == "Start fighting":
                print(str(i) + ": " + "I fight against " + enemy)
                result = np.random.choice(t3, p=p_t3[2])
                i = i + 1
                if result == "Win":
                    won(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif result == "Loose":
                    lost(i)
                    break
                elif result == "Draw":
                    draw(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i+1
            elif activity == "Run":
                run_result = np.random.choice(t4, p=p_t4)
                i = i + 1
                if run_result == "Run Away":
                    ran_away(i)
                    enemy = np.random.choice(t1, p=p_t1)
                    i = i + 1
                elif run_result == "Loose":
                    lost(i)
                    break
        elif enemy == "Search":
            print(str(i) + ": " + "I am searching for enemy")
            i = i + 1
            enemy = np.random.choice(t1, p=p_t1)


def lost(i):
    print(str(i) + ": " + "I lost")
    print("Simulation ended")


def draw(i):
    print(str(i) + ": " + "We got draw")


def won(i):
    print(str(i) + ": " + "I won")


def ran_away(i):
    print(str(i) + ": " + "I ran away")


game(games_number)
