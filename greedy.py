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



def greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time):
    """
    Greedy algoritam za inicijalno rešenje sa prioritizacijom troška i najranije dostupnog aviona.
    Args:
        R: Broj ruta.
        T: Broj tipova aviona.
        A_t: Broj aviona po tipu.
        T_r: Vreme trajanja leta po ruti.
        H_t: Vreme provedeno na zemlji za svaki tip aviona.
        max_hours: Maksimalno radno vreme aviona.
        F_t_r: Matrica troškova za tipove aviona i rute.
        base_time: Početno vreme (datetime ili string).
    Returns:
        np.array: Kod rešenja (matrica R x T).
    """
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
            
            #   # Provera da li avion tog tipa ima dovoljan domet za rutu
            # if distance_r[r] > range_t[t]:
            #     continue

            for a in range(A_t[t]):  # Iteracija kroz avione tipa t
                if (
                    available_times[t][a] <= earliest_time and  # Provera dostupnosti
                    flight_hours[t][a] + T_r[r] + H_t[t] <= max_hours  # Provera radnog vremena
                ):
                    penalty = max(0, range_t[t] - distance_r[r])  # Penalizacija ako avion ima mnogo veći domet od potrebnog
                    cost = F_t_r[r][t] + penalty  # Dodaj penalizaciju na trošak

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


def evaluate_solution(x_rt, base_time):
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
                    
                    #   # Provera da li avion tog tipa ima dovoljan domet za rutu
                    # if distance_r[r] > range_t[t]:
                    #     continue
                            
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


code = greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time)

uncovered_routes, fitness = evaluate_solution(code, base_time)

print(f'Nepokrivene rute {uncovered_routes} , fitness:{fitness}')
