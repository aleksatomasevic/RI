from copy import deepcopy
import random
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta

# ucitavanje iz json fajla
# kroisticemo apsolutne putanje do fajla
current_dir = Path(__file__).parent  # direktorijum u kojem se nalazi trenutni .py fajl
json_path = current_dir / "parameters.json"

with open(json_path, "r") as f:
    datasets = json.load(f)

# biramo skup podatak s kojm radimo
selected_dataset = "dataset7"  
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
destinations = params["destinations"]  # Lista destinacija
plane_types = params["plane_types"]  # Lista tipova aviona
base_time = params["base_time"]  # Početno vreme

all_routes = range(R) #Sve rute
all_types = range(T) #Svi tipovi
specific_planes = {t: range(A_t[t]) for t in all_types}  # Skup specifičnih aviona unutar svakog tipa

#izmena, proverava da li postoji bar jedno validno resenje unutar dataseta
def validate_dataset(R, T, A_t, T_r, H_t, max_hours):
    uncovered_routes = []
    available_times = {t: [0] * A_t[t] for t in range(T)}  # Vremena dostupnosti za svaki avion
    flight_hours = {t: [0] * A_t[t] for t in range(T)}  # Ukupno radno vreme za svaki avion

    for r in range(R):
        assigned = False
        earliest_time = min([min(available_times[t]) for t in range(T)])  # Pronalazi najraniji avion

        for t in range(T):
            for a in range(A_t[t]):
                if (
                    available_times[t][a] <= earliest_time and  # Proverava da li avion može da poleti
                    flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours  # Proverava ukupno radno vreme
                ):
                    # Ažuriraj dostupnost i radne sate aviona
                    available_times[t][a] = earliest_time + T_r[r] + H_t[t]
                    flight_hours[t][a] += T_r[r] + H_t[t]
                    assigned = True
                    break  # Dodeljen avion za ovu rutu
            if assigned:
                break  # Ruta je pokrivena, ne traži dalje

        if not assigned:
            uncovered_routes.append(r)  # Ako nijedan avion nije mogao da pokrije rutu

    return uncovered_routes


# Pozovi validate_dataset
uncovered_routes = validate_dataset(R, T, A_t, T_r, H_t, max_hours)
if uncovered_routes:
    print(f"Nepokrivene rute u dataset-u: {uncovered_routes}")
else:
    print("Sve rute u dataset-u mogu biti pokrivene.")
#-----------------------------------

class Individual:
    def __init__(self, R, T):
        self.code = self.initialize_valid_solution(R, T)
        self.fitness = None  # Fitnes se računa kasnije
        self.uncovered_routes = []

    # izmena
    # ovo je funkcija koja ce da nam osigura da imamo tacno jednu jedinicu u svakoj vrti tj za svaku rutu
    def initialize_valid_solution(self, R, T, epsilon=0.01):
        code = np.zeros((R, T), dtype=int)
        for r in range(R):
            t = random.randint(0, T-1)
            code[r, t] = 1
        return code
   

    def calculate_fitness(self):
        # self.fitness , _, _, _ = self.evaluate_solution(self.code)
        self.fitness = self.evaluate_solution(self.code)

    #izmena, pametniji mutate
    def mutate(self, generation, max_generations):

        mutation_rate = 0.05

        for r in range(self.code.shape[0]):  # Iteracija kroz sve rute
            if random.random() < mutation_rate:  # Primeni mutaciju sa verovatnoćom mutation_rate
                # Nađi trenutni tip aviona za rutu
                t_old = np.argmax(self.code[r])
                # Izaberi novi tip aviona koji nije isti kao trenutni
                t_new = random.choice([t for t in range(self.code.shape[1]) if t != t_old])
                # Ažuriraj dodelu
                self.code[r, t_old] = 0
                self.code[r, t_new] = 1




    def evaluate_solution(self, x_rt):
        total_cost = 0
        total_profit = 0
        available_times = {t: [0] * A_t[t] for t in all_types}
        flight_hours = {t: [0] * A_t[t] for t in all_types}
        current_time = 0 
        previous_time = 0 #Prethodno vreme za racunanje proteklog vremena izmedju njega i trenutnog
        uncovered_routes = []

        for r in all_routes:
            route_cost = 0
            route_profit = 0
            valid_assignments = 0  # Broj validnih dodela aviona za rutu
            assigned = False

            # Pronalazi prvi slobodan avion unutar svih tipova
            earliest_time = min([min(available_times[t]) for t in range(len(all_types))])


            for t in all_types:
                if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?
                    #valid_assignments += 1
                    
                    for a in specific_planes[t]:
                        if (
                            available_times[t][a] <= earliest_time and
                            flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                        ):
                            cost = F_t_r[r][t]
                            assigned = True
                            # print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                            # f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                            route_cost += cost
                            route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]

                            # Ažuriraj dostupnost i radne sate za trenutno avion
                            available_times[t][a] = earliest_time + T_r[r] + H_t[t]
                            flight_hours[t][a] += T_r[r] + H_t[t]

                            valid_assignments += 1

                            break

            if  not assigned:  #Ruta nije pokrivena
                uncovered_routes.append(r)

            # Dodaj ukupne troškove i profit za rutu
            total_cost += route_cost
            total_profit += route_profit

        # F-ja koju minimizujemo
        objective = alpha * total_cost - beta * total_profit

        return (len(uncovered_routes), objective)


def crossover(parent1, parent2, child1, child2):
    # Odaberi slučajnu tačku za ukrštanje (red matrice)
    crossover_point = random.randint(1, R - 1)  # Mora biti između prvog i poslednjeg reda

    # Kreiraj potomke kombinovanjem roditeljskih matrica
    temp1 = np.vstack((parent1.code[:crossover_point], parent2.code[crossover_point:]))
    temp2 = np.vstack((parent2.code[:crossover_point], parent1.code[crossover_point:]))

    # Ispravi rešenja tako da svaka ruta ima tačno jednu jedinicu
    for r in range(R):
        if temp1[r].sum() != 1:
            t = random.randint(0, T - 1)
            temp1[r] = np.zeros(T, dtype=int)
            temp1[r, t] = 1
        if temp2[r].sum() != 1:
            t = random.randint(0, T - 1)
            temp2[r] = np.zeros(T, dtype=int)
            temp2[r, t] = 1

    # Ažuriraj kodove potomaka
    child1.code = temp1
    child2.code = temp2


def selection(population, tournament_size=5):
    # Izaberi slučajne jedinke za turnir
    tournament = random.sample(population, tournament_size)
    # Pronađi jedinku sa najmanjim fitnesom (jer tražimo minimum)
    winner = min(tournament, key=lambda x: x.fitness)
    return winner



def heuristic_solution():
    code = np.zeros((R, T), dtype=int)
    available_times = {t: [0] * A_t[t] for t in range(T)}
    for r in range(R):
        best_t = -1
        earliest_time = float('inf')
        for t in range(T):
            for a in range(A_t[t]):
                if available_times[t][a] + T_r[r] + H_t[t] <= max_hours and available_times[t][a] < earliest_time:
                    best_t = t
                    earliest_time = available_times[t][a]
        if best_t != -1:
            code[r, best_t] = 1
            for a in range(A_t[best_t]):
                if available_times[best_t][a] <= earliest_time:
                    available_times[best_t][a] += T_r[r] + H_t[best_t]
                    break
    return code


def genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE):
    # Inicijalizacija populacije
    population = [Individual(R, T) for _ in range(POPULATION_SIZE - 10)]
    
    for _ in range(10):  # Generiši heuristički jedinke
        individual = Individual(R, T)
        individual.code = heuristic_solution()
        individual.calculate_fitness()
        population.append(individual)
    
    newPopulation = [Individual(R, T) for _ in range(POPULATION_SIZE)]
    
    for individual in population:
        individual.calculate_fitness()

    for generation in range(NUM_GENERATIONS):
        population.sort(key=lambda x: x.fitness)

        # Debugging: Prikaz najboljeg u generaciji
        print(f"Generacija {generation}: Najbolji fitnes = {population[0].fitness}")

        newPopulation[:ELITISIM_SIZE] = population[:ELITISIM_SIZE]

        for i in range(ELITISIM_SIZE, POPULATION_SIZE, 2):
            parent1 = selection(population)
            parent2 = selection(population)

            # Ukrštanje
            crossover(parent1, parent2, newPopulation[i], newPopulation[i+1])

            # Mutacija
            newPopulation[i].mutate(generation, NUM_GENERATIONS)
            newPopulation[i+1].mutate(generation, NUM_GENERATIONS)

            # Računanje fitnesa
            newPopulation[i].calculate_fitness()
            newPopulation[i+1].calculate_fitness()

        population = deepcopy(newPopulation)


    return min(population, key=lambda ind: ind.fitness)

def generate_flight_schedule(best_code, destinations, plane_types, specific_planes, T_r, base_time):
    flight_schedule = []
    flight_count = 1  # Brojač letova

    # Pretvaranje base_time u datetime ako je prosleđeno kao string
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%H:%M")

    available_times = {t: [base_time] * A_t[t] for t in range(len(all_types))}
    flight_hours = {t: [0] * A_t[t] for t in range(len(all_types))}  # Početno radno vreme za sve avione

    # Globalno vreme za raspoređivanje polazaka
    # added
    global_time = base_time

    for r, row in enumerate(best_code):
        # Pronađi najranije dostupno vreme za sve avione
        #earliest_time = min([min(available_times[t]) for t in range(len(plane_types))])
        assigned = False # Da li je ruta pokrivena

        for t, assigned in enumerate(row):
            if assigned == 1:  # Ako je ruta pokrivena tipom aviona
                for a in specific_planes[t]:
                    if (
                        available_times[t][a] <= global_time and  # Proveri dostupnost
                        flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours  # Proveri maksimalno radno vreme
                    ):
                        # added
                        departure_time = max(global_time, available_times[t][a])  # Najraniji mogući polazak
                        available_times[t][a] = departure_time + timedelta(minutes=T_r[r] + H_t[t])  # Ažuriraj dostupnost

                        #available_times[t][a] += departure_time + T_r[r] + H_t[t]  # Ažuriranje vremena dostupnosti
                        flight_hours[t][a] += T_r[r] + H_t[t]  # Ažuriranje radnog vremena

                        flight_schedule.append({
                            "Flight Number": flight_count,
                            "Destination": destinations[r],
                            "Plane Type": plane_types[t],
                            "Plane ID": f"T{t+1}-{a+1}",
                            "Departure Time": departure_time.strftime("%H:%M"),
                        })
                        flight_count += 1
                        # added
                        global_time += timedelta(minutes=5)  # Povećaj globalno vreme za 5 minuta
                        assigned = True
                        break  # Prvi slobodan avion unutar tipa

    # Kreiraj i prikaži tabelu
    df = pd.DataFrame(flight_schedule)
    print(df)
    return df

# Parametri za genetski algoritam
NUM_GENERATIONS = 1500
POPULATION_SIZE = 100
ELITISIM_SIZE = POPULATION_SIZE // 10
if ELITISIM_SIZE % 2 == 1:
    ELITISIM_SIZE -= 1 


best_individual = genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE)
print("Najbolja jedinka:")
print(best_individual.code)

#izmena aleksa
uncovered_routes, fitness  = best_individual.evaluate_solution(best_individual.code)
print(f"Fitnes: {fitness}")
if uncovered_routes:
    print(f"Nepokrivene rute: {uncovered_routes}")
    print(f"Broj nepokrivenih ruta: {uncovered_routes}")
else:
    print("Sve rute su pokrivene!")
#-----------------------

flight_schedule_df = generate_flight_schedule(
    best_individual.code, destinations, plane_types, specific_planes, T_r, base_time
)
