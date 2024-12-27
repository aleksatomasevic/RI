import random
import numpy as np
from pathlib import Path
import json

# ucitavanje iz json fajla
# kroisticemo apsolutne putanje do fajla
current_dir = Path(__file__).parent  # direktorijum u kojem se nalazi trenutni .py fajl
json_path = current_dir / "parameters.json"

with open(json_path, "r") as f:
    datasets = json.load(f)

# biramo skup podatak s kojim radimo
selected_dataset = "dataset4"  
params = datasets[selected_dataset]


# postavljanje vrednosti promenljivama
R = params["R"] # Broj ruta
T = params["T"] # Broj tipova aviona
A_t = {int(k): v for k, v in params["A_t"].items()}  # Broj aviona po tipu
P_r = params["P_r"] # Broj putnika za svaku rutu
F_t_r = params["F_t_r"] # Matrica troska konkretnog tipa aviona na konkretnoj ruti
C_t = params["C_t"] # Kapacitet aviona po tipu
T_r = params["T_r"] # Vreme trajanja leta za svaku rutu
H_t = params["H_t"] # Vreme provedeno na zemlji za svaki tip aviona
alpha = params["alpha"]
beta = params["beta"]
max_hours = params["max_hours"] # Maksimalno vreme rada aviona
price_per_passenger = params["price_per_passenger"] # Cena karte po putniku za svaku rutu

all_routes = range(R) #Sve rute
all_types = range(T) #Svi tipovi
specific_planes = {t: range(A_t[t]) for t in all_types}  # Skup specifičnih aviona unutar svakog tipa

class Individual:
    def __init__(self, R, T):
        self.code = np.random.randint(0, 2, size=(R, T))
        self.fitness = None  # Fitnes se računa kasnije

    def calculate_fitness(self):
        self.fitness, _, _ = self.evaluate_solution(self.code)

    def mutate(self):
        r = random.randint(0, self.code.shape[0] - 1)
        t = random.randint(0, self.code.shape[1] - 1)
        self.code[r, t] = 1 - self.code[r, t]  # Flip bit

    def evaluate_solution(self, x_rt):
        total_cost = 0
        total_profit = 0
        available_times = {t: [0] * A_t[t] for t in all_types}
        flight_hours = {t: [0] * A_t[t] for t in all_types}
        current_time = 0 
        previous_time = 0 #Prethodno vreme za racunanje proteklog vremena izmedju njega i trenutnog

        # print(f"Evaluacija kombinacije: \n{x_rt}") 

        for r in all_routes:
            route_cost = 0
            route_profit = 0
            valid_assignments = 0  # Broj validnih dodela aviona za rutu

            # Pronalazi prvi slobodan avion unutar svih tipova
            earliest_time = min([min(available_times[t]) for t in all_types])


            for t in all_types:
                if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?
                    valid_assignments += 1
                    
                    for a in specific_planes[t]:
                        if (
                            available_times[t][a] <= earliest_time and
                            flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                        ):
                            cost = F_t_r[r][t]
                            # print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                            # f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                            route_cost += cost
                            route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]

                            # Ažuriraj dostupnost i radne sate za trenutno avion
                            available_times[t][a] = earliest_time + T_r[r] + H_t[t]
                            flight_hours[t][a] += T_r[r] + H_t[t]

                            break

            if valid_assignments > 1:
                route_cost += 1000000

            if  not valid_assignments:  #Ruta nije pokrivena
                # print(f'Ruta {t} nije pokrivena')
                route_cost += 1000000  # Penal

            # Dodaj ukupne troškove i profit za rutu
            total_cost += route_cost
            total_profit += route_profit
            
            previous_time = current_time
            current_time += 1 #Simulacija vremena kroz svaki korak

        # F-ja koju minimizujemo
        objective = alpha * total_cost - beta * total_profit
        # print(f"Ukupan trošak: {total_cost}, Ukupan profit: {total_profit}, Ciljna funkcija: {objective}\n")
        return objective, total_cost, total_profit

def crossover(parent1, parent2, child1, child2):
    mask = np.random.randint(0, 2, size=(R, T))
    child1.code = parent1.code * mask + parent2.code * (1 - mask)
    child2.code = parent2.code * mask + parent1.code * (1 - mask)

def selection(population):
    TOURNAMENT_SIZE = 5
    selected = random.sample(population, TOURNAMENT_SIZE)
    best = min(selected, key=lambda ind: ind.fitness)
    return best

def genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE):
    # Inicijalizacija populacije
    population = [Individual(R, T) for _ in range(POPULATION_SIZE)]
    newPopulation = [Individual(R, T) for _ in range(POPULATION_SIZE)]
    for individual in population:
        individual.calculate_fitness()

    for generation in range(NUM_GENERATIONS):
        population.sort(key=lambda x: x.fitness, reverse=True)
        newPopulation[:ELITISIM_SIZE] = population[:ELITISIM_SIZE]

        for i in range(ELITISIM_SIZE, POPULATION_SIZE, 2):
            parent1 = selection(population)
            parent2 = selection(population)

            # Ukrštanje
            crossover(parent1, parent2, newPopulation[i], newPopulation[i+1])

            # Mutacija
            newPopulation[i].mutate()
            newPopulation[i+1].mutate()

            # Računanje fitnesa
            newPopulation[i].calculate_fitness()
            newPopulation[i+1].calculate_fitness()

        population = newPopulation

        # Debugging: Prikaz najboljeg u generaciji
        # print(f"Generacija {generation}: Najbolji fitnes = {best_individual.fitness}")

    return min(population, key=lambda ind: ind.fitness)

# Parametri za genetski algoritam
NUM_GENERATIONS = 1000
POPULATION_SIZE = 200
ELITISIM_SIZE = POPULATION_SIZE // 100
if ELITISIM_SIZE % 2 == 1:
    ELITISIM_SIZE -= 1 

best_individual = genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE)
print("Najbolja jedinka:")
print(best_individual.code)
print(f"Fitnes: {best_individual.fitness}")

