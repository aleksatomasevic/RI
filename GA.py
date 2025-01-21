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
selected_dataset = "dataset10"  
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
distance_r = params["distance_r"]
range_t = params["range_t"]

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
        self.fitness = self.evaluate_solution(self.code, base_time)

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




    def evaluate_solution(self, x_rt, base_time):
        total_cost = 0
        total_profit = 0
        turnaround_time = 45

         # Pretvaranje base_time u datetime ako je prosleđeno kao string
        if isinstance(base_time, str):
            base_time = datetime.strptime(base_time, "%H:%M")

        available_times = {t: [base_time] * A_t[t] for t in all_types}
        flight_hours = {t: [0] * A_t[t] for t in all_types}
        global_time = base_time
        uncovered_routes = []

        for r in all_routes:
            route_cost = 0
            route_profit = 0
            valid_assignments = 0  # Broj validnih dodela aviona za rutu
            assigned = False

            # Pronalazi prvi slobodan avion unutar svih tipova
            earliest_time = max(global_time, min([min(available_times[t]) for t in range(len(all_types))]))


            for t in all_types:
                if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?
                    #valid_assignments += 1

                    # # Provera da li avion tog tipa ima dovoljan domet za rutu
                    # if distance_r[r] > 8000 and range_t[t] < 12000:
                    #     continue  # Dugodometni avioni su obavezni za duge rute


                    for a in specific_planes[t]:
                        if (
                            available_times[t][a] <= earliest_time and
                            flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                        ):
                            penalty = max(0, range_t[t] - distance_r[r])  # Penalizacija ako avion ima mnogo veći domet od potrebnog
                            cost = F_t_r[r][t] + penalty
                            assigned = True
                            # print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                            # f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                            route_cost += cost
                            route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]

                            # Ažuriraj vreme polaska i povratka
                            departure_time = earliest_time
                            return_time = departure_time + timedelta(hours=T_r[r] + H_t[t])
                            
                            # Ažuriraj dostupnost i radne sate za trenutno avion
                            available_times[t][a] = return_time + timedelta(minutes=turnaround_time)
                            flight_hours[t][a] += T_r[r] + H_t[t]

                            # Povećaj globalno vreme
                            global_time = departure_time + timedelta(minutes=5)

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


def greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time):
    
    # Pretvaranje base_time u datetime ako je prosleđeno kao string
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%H:%M")

    # Početna matrica rešenja
    code = np.zeros((R, T), dtype=int)

    # Dostupno vreme i radno vreme za svaki avion
    available_times = {t: [base_time] * A_t[t] for t in range(T)}
    flight_hours = {t: [0] * A_t[t] for t in range(T)}
    global_time = base_time
    turnaround_time = 45  # Vreme obrade aviona po povratku (u minutima)

    for r in range(R):  # Iteracija kroz rute
        best_t = -1
        best_cost = float('inf')
        assigned = False

        # Pronađi najranije dostupno vreme među svim avionima
        earliest_time = max(global_time, min([min(available_times[t]) for t in range(T)]))

        for t in range(T):  # Iteracija kroz tipove aviona
            
        #    # Provera da li avion tog tipa ima dovoljan domet za rutu
        #     if distance_r[r] > 8000 and range_t[t] < 12000:
        #         continue  # Dugodometni avioni su obavezni za duge rute
            
            for a in range(A_t[t]):  # Iteracija kroz avione tipa t
                if (
                    available_times[t][a] <= earliest_time and  # Provera dostupnosti
                    flight_hours[t][a] + T_r[r] + H_t[t] <= max_hours  # Provera radnog vremena
                ):
                    cost = F_t_r[r][t]  # Trošak za rutu
                    if cost < best_cost:  # Traženje najmanjeg troška
                        best_t = t
                        best_a = a  # Zabeleži koji avion je najbolji
                        assigned = True

        if assigned:  # Ako je ruta pokrivena
            code[r, best_t] = 1

            # Odredi vreme polaska i povratka
            departure_time = earliest_time
            return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])

            # Ažuriraj dostupnost aviona (dodaj vreme obrade)
            available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)

            # Ažuriraj radno vreme aviona
            flight_hours[best_t][best_a] += T_r[r] + H_t[best_t]

            # Povećaj globalno vreme za minimalni razmak između letova
            global_time = departure_time + timedelta(minutes=5)

    return code



def genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE):
    # Inicijalizacija populacije
    population = [Individual(R, T) for _ in range(POPULATION_SIZE - 10)]

     # Otvaranje fajla za zapisivanje
    log_file = open("log.txt", "w")

    
    for _ in range(10):  # Generiši heuristički jedinke
        individual = Individual(R, T)
        individual.code = greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time)
        individual.calculate_fitness()
        print(individual.fitness)
        population.append(individual)
    
    newPopulation = [Individual(R, T) for _ in range(POPULATION_SIZE)]
    
    for individual in population:
        individual.calculate_fitness()

    for generation in range(NUM_GENERATIONS):
        population.sort(key=lambda x: x.fitness)

        # Debugging: Prikaz najboljeg u generaciji
        print(f"Generacija {generation}: Najbolji fitnes = {population[0].fitness}")
        log_file.write(f"Generacija {generation}: Najbolji fitnes = {population[0].fitness}\n")

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
    turnaround_time = 45

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
        earliest_time = max(global_time, min([min(available_times[t]) for t in range(len(plane_types))]))
        assigned = False # Da li je ruta pokrivena

        for t, assigned in enumerate(row):
            if assigned == 1:  # Ako je ruta pokrivena tipom aviona
                for a in specific_planes[t]:
                    if (
                        available_times[t][a] <= earliest_time and  # Proveri dostupnost
                        flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours  # Proveri maksimalno radno vreme
                    ):
                        # added
                        departure_time = earliest_time  # Najraniji mogući polazak
                        return_time = departure_time + timedelta(hours=T_r[r] + H_t[t])  # Ažuriraj dostupnost

                        available_times[t][a] = return_time + timedelta(minutes=turnaround_time) # Ažuriranje vremena dostupnosti
                        flight_hours[t][a] += T_r[r] + H_t[t]  # Ažuriranje radnog vremena

                        flight_schedule.append({
                            "Flight Number": flight_count,
                            "Destination": destinations[r],
                            "Plane Type": plane_types[t],
                            "Plane ID": f"T{t+1}-{a+1}",
                            "Departure Time": departure_time.strftime("%H:%M"),
                            "Return Time": return_time.strftime("%H:%M"),
                        })
                        flight_count += 1
                        # added
                        global_time = departure_time + timedelta(minutes=5)  # Povećaj globalno vreme za 5 minuta
                        assigned = True
                        break  # Prvi slobodan avion unutar tipa

    # Kreiraj i prikaži tabelu
    df = pd.DataFrame(flight_schedule)
    print(df)
    return df

# Parametri za genetski algoritam
NUM_GENERATIONS = 500
POPULATION_SIZE = 100
ELITISIM_SIZE = POPULATION_SIZE // 10
if ELITISIM_SIZE % 2 == 1:
    ELITISIM_SIZE -= 1 


best_individual = genetic_algorithm(R, T, NUM_GENERATIONS, POPULATION_SIZE, ELITISIM_SIZE)
print("Najbolja jedinka:")
print(best_individual.code)

#izmena aleksa
uncovered_routes, fitness  = best_individual.evaluate_solution(best_individual.code, base_time)
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
