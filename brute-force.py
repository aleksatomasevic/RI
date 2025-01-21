import itertools
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

# Generisanje svih mogućih kombinacija (mislim da je ovo najlakse zaa generisanje kombinacija)
combinations = itertools.product([0, 1], repeat=R * T)

def evaluate_solution(x_rt):
    x_rt = np.array(x_rt).reshape(R, T)  # Matrica R x T
    total_cost = 0
    total_profit = 0
    available_times = {t: [0] * A_t[t] for t in all_types}
    flight_hours = {t: [0] * A_t[t] for t in all_types}
    current_time = 0 
    previous_time = 0 #Prethodno vreme za racunanje proteklog vremena izmedju njega i trenutnog

    print(f"Evaluacija kombinacije: \n{x_rt}") 

    for r in all_routes:
        route_cost = 0
        route_profit = 0
        valid_assignments = 0  # Broj validnih dodela aviona za rutu
        assigned = False

        # Pronalazi prvi slobodan avion unutar svih tipova
        earliest_time = min([min(available_times[t]) for t in all_types])


        # for t in all_types:
        #     for a in specific_planes[t]:
        #         if available_times[t][a] > 0:
        #             available_times[t][a] -= (current_time - previous_time)
        #             if available_times[t][a] < 0: #Avion je slobodan
        #                 available_times[t][a]= 0


        for t in all_types:
            if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?
                # valid_assignments += 1
                for a in specific_planes[t]:
                    if (
                        available_times[t][a] <= earliest_time and
                        flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                    ):
                        valid_assignments += 1
                        
                        cost = F_t_r[r][t]

                        assigned = True


                        print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                          f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                        route_cost += cost
                        route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]

                        # Ažuriraj dostupnost i radne sate za trenutno avion
                        available_times[t][a] = earliest_time + T_r[r] + H_t[t]
                        flight_hours[t][a] += T_r[r] + H_t[t]

                        break

        if valid_assignments > 1:
            route_cost += 10000

        if  not assigned:  #Ruta nije pokrivena
            print(f'Ruta {t} nije pokrivena')
            route_cost += 10000  # Penal

        # Dodaj ukupne troškove i profit za rutu
        total_cost += route_cost
        total_profit += route_profit
        
        previous_time = current_time
        current_time += 1 #Simulacija vremena kroz svaki korak

    # F-ja koju minimizujemo
    objective = alpha * total_cost - beta * total_profit
    print(f"Ukupan trošak: {total_cost}, Ukupan profit: {total_profit}, Ciljna funkcija: {objective}\n")
    return objective, total_cost, total_profit


# Traženje min rešenja
best_solution = None
best_objective = float('inf')
best_cost = 0
best_profit = 0

for combination in combinations:
    obj, cost, profit = evaluate_solution(combination)
    if obj < best_objective:
        best_objective = obj
        best_cost = cost
        best_profit = profit
        best_solution = combination

# Prikaz rezultata
if best_solution is not None:
    print("Najbolje rešenje:")
    print(np.array(best_solution).reshape(R, T))
    print(f"Trošak: {best_cost}, Profit: {best_profit}, Ciljna funkcija: {best_objective}")
else:
    print("Nije pronađeno validno rešenje.")