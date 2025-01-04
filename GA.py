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

    # izmena
    # ovo je funkcija koja ce da nam osigura da imamo tacno jednu jedinicu u svakoj vrti tj za svaku rutu
    # def initialize_valid_solution(self, R, T, epsilon=0.01):
    #     code = np.zeros((R, T), dtype=int)
    #     for r in range(R):
    #         t = random.randint(0, T-1)
    #         code[r, t] = 1
    #     return code
    def initialize_valid_solution(self, R, T):
        code = np.zeros((R, T), dtype=int)

        for r in range(R):
            # Heuristički pokušaj pronalaska najboljeg aviona
            best_type = None
            best_score = float('inf')

            for t in range(T):
                cost = F_t_r[r][t]
                capacity = C_t[t]
                passengers = P_r[r]

                # Proveri validnost
                if capacity >= passengers:  # Avion može prevesti sve putnike
                    if cost < best_score:  # Pronađi minimalni trošak
                        best_score = cost
                        best_type = t

            # Ako heuristika ne pronađe validan avion
            if best_type is None:
                # Nasumično dodeli bilo koji avion
                random_type = np.random.randint(0, T)
                best_type = random_type

            # Dodeli avion za rutu
            code[r, best_type] = 1

        return code



    # funkcija koja ce da nam proverava validnost jedinke nakon mutacije i ukrstanja
    # def is_valid(self):
    #     return all(self.code[r].sum() == 1 for r in range(self.code.shape[0]))
    #--------------------------------------------------------------------------

    def calculate_fitness(self):
        self.fitness, _, _, _ = self.evaluate_solution(self.code)

    # izmena, pametniji mutate
    def mutate(self, generation, max_generations):
        # Dinamička stopa mutacije
        mutation_rate = 0.2 * (1 - generation / max_generations)
        r = random.randint(0, self.code.shape[0] - 1)
        if random.random() < mutation_rate:  # Više jedinica
            self.code[r] = np.zeros(self.code.shape[1], dtype=int)
            self.code[r, random.randint(0, self.code.shape[1] - 1)] = 1
        else:
            t_old = np.argmax(self.code[r])
            t_new = random.choice([t for t in range(self.code.shape[1]) if t != t_old])
            self.code[r, t_old] = 0
            self.code[r, t_new] = 1
    #---------------------------------------------------------



    def evaluate_solution(self, x_rt):
        total_cost = 0
        total_profit = 0
        available_times = {t: [0] * A_t[t] for t in all_types}
        flight_hours = {t: [0] * A_t[t] for t in all_types}
        current_time = 0 
        previous_time = 0 #Prethodno vreme za racunanje proteklog vremena izmedju njega i trenutnog
        uncovered_routes = []
        # print(f"Evaluacija kombinacije: \n{x_rt}") 

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
            if valid_assignments > 1:
                 penalty = 20000 * valid_assignments  # Penal po jedinici iznad jedne
                 route_cost += penalty

            if  not assigned:  #Ruta nije pokrivena
                #print(f'Ruta {r} nije pokrivena')
                route_cost += 100000  # Penal
                uncovered_routes.append(r)

            # Dodaj ukupne troškove i profit za rutu
            total_cost += route_cost
            total_profit += route_profit
            
            previous_time = current_time
            current_time += 1 #Simulacija vremena kroz svaki korak


        # F-ja koju minimizujemo
        #print(f"Broj nepokrivenih ruta: {len(uncovered_routes)}")
        objective = alpha * total_cost - beta * total_profit
        # print(f"Ukupan trošak: {total_cost}, Ukupan profit: {total_profit}, Ciljna funkcija: {objective}\n")
        return objective, total_cost, total_profit, uncovered_routes

def crossover(parent1, parent2, child1, child2, crossover_epsilon=0.1):
    mask = np.random.randint(0, 2, size=(R, T))
    temp1 = parent1.code * mask + parent2.code * (1 - mask)
    temp2 = parent2.code * mask + parent1.code * (1 - mask)

    # izmena
    # Ispravi rešenja tako da svaka ruta ima tačno jednu jedinicu
    for r in range(R):
        if temp1[r].sum() != 1:
            t = random.randint(0, T - 1)
            temp1[r] = np.zeros(T, dtype=int)
            temp1[r, t] = 1
        # Popravi za dete 2
        if temp2[r].sum() != 1:
            t = random.randint(0, T - 1)
            temp2[r] = np.zeros(T, dtype=int)
            temp2[r, t] = 1

    child1.code = temp1
    child2.code = temp2
    #--------------------------------------------------------------------------

# def selection(population):
#     TOURNAMENT_SIZE = 5
#     selected = random.sample(population, TOURNAMENT_SIZE)
    
#     # Računaj diverzitet unutar turnira
#     diversity_bonus = [
#         sum(np.sum(np.abs(ind.code - other.code)) for other in selected) for ind in selected
#     ]
    
#     # Kombinujte fitnes i bonus za diverzitet
#     fitness_with_diversity = [
#         ind.fitness - 0.1 * diversity for ind, diversity in zip(selected, diversity_bonus)
#     ]
    
#     # Izaberi jedinku sa najboljim kombinovanim fitnesom
#     best = selected[np.argmin(fitness_with_diversity)]
#     return best

#izmena, pametniji selection 
def selection(population):
    # Rangiraj populaciju prema fitnesu (manji fitnes je bolji)
    population.sort(key=lambda x: x.fitness)
    
    # Izračunaj šanse za selekciju
    total_rank = sum(range(1, len(population) + 1))
    probabilities = [rank / total_rank for rank in range(1, len(population) + 1)]
    
    # Selektuj jedinku prema verovatnoći
    selected = np.random.choice(population, p=probabilities)
    return selected
#------------------------------------


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
            newPopulation[i].mutate(generation, NUM_GENERATIONS)
            newPopulation[i+1].mutate(generation, NUM_GENERATIONS)

            # Računanje fitnesa
            newPopulation[i].calculate_fitness()
            newPopulation[i+1].calculate_fitness()

        population = newPopulation

        # Debugging: Prikaz najboljeg u generaciji
        # print(f"Generacija {generation}: Najbolji fitnes = {best_individual.fitness}")

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
NUM_GENERATIONS = 5000
POPULATION_SIZE = 200
ELITISIM_SIZE = POPULATION_SIZE // 20
if ELITISIM_SIZE % 2 == 1:
    ELITISIM_SIZE -= 1 


best_individual = genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE)
print("Najbolja jedinka:")
print(best_individual.code)

#izmena aleksa
fitness, total_cost, total_profit, uncovered_routes = best_individual.evaluate_solution(best_individual.code)
print(f"Fitnes: {fitness}")
print(f"Ukupan trošak: {total_cost}, Ukupan profit: {total_profit}")
if uncovered_routes:
    print(f"Nepokrivene rute: {uncovered_routes}")
    print(f"Broj nepokrivenih ruta: {len(uncovered_routes)}")
else:
    print("Sve rute su pokrivene!")
#-----------------------

flight_schedule_df = generate_flight_schedule(
    best_individual.code, destinations, plane_types, specific_planes, T_r, base_time
)
