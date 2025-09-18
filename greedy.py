from copy import deepcopy
import random
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# ucitavanje iz json fajla
# kroisticemo apsolutne putanje do fajla
current_dir = Path(__file__).parent  # direktorijum u kojem se nalazi trenutni .py fajl
json_path = current_dir / "parameters.json"

with open(json_path, "r") as f:
    datasets = json.load(f)

# biramo skup podatak s kojm radimo
#selected_dataset = os.getenv("SELECTED_DATASET", "albatross_large")  
selected_dataset = "albatross_large"  
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

# Konstante za dnevne limite i vreme čekanja
short_range_limit = 16  # Maksimalni sati rada za male avione
long_range_limit = 20   # Maksimalni sati rada za velike avione
wait_time_hours = 6     # Vreme čekanja na pisti nakon prekoračenja limita

def greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time):

    # Pretvaranje base_time u datetime ako je prosleđeno kao string
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M")

    # Početna matrica rešenja
    code = np.zeros((R, T), dtype=int)

    # Dostupno vreme i radno vreme za svaki avion
    available_times = {t: [base_time] * A_t[t] for t in range(T)}
    flight_hours = {t: [0] * A_t[t] for t in range(T)}
    global_time = base_time
    turnaround_time = 45  # Vreme obrade aviona po povratku (u minutima)

    for r in range(R):  # Iteracija kroz rute
        best_t = None
        best_a = None
        best_cost = float('inf')
        assigned = False

        # Pronađi najranije dostupno vreme među svim avionima
        earliest_time = max(global_time, min([min(available_times[t]) for t in range(T)]))

        for t in range(T):  # Iteracija kroz tipove aviona
            
              # Provera da li avion tog tipa ima dovoljan domet za rutu
            if distance_r[r] > range_t[t]:
                continue

            for a in range(A_t[t]):  # Iteracija kroz avione tipa t
                if (
                    available_times[t][a] <= earliest_time # Provera dostupnosti
                ):
                    cost = F_t_r[r][t]
                    if cost < best_cost:  # Traženje najmanjeg troška
                        best_t = t
                        best_a = a  # Zabeleži koji avion je najbolji

        if best_t is not None:  # Ako je ruta pokrivena
            code[r, best_t] = 1

            # Odredi vreme polaska i povratka
            departure_time = earliest_time
            return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])

            daily_limit = short_range_limit if range_t[best_t] <= 5000 else long_range_limit
            if flight_hours[best_t][best_a] + (T_r[r] + H_t[best_t]) >= daily_limit:
                available_times[best_t][best_a] = return_time + timedelta(hours=wait_time_hours)
                flight_hours[best_t][best_a] = 0
            else:
                available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)
                flight_hours[best_t][best_a] += T_r[r] + H_t[best_t] + turnaround_time / 60

            # Povećaj globalno vreme za minimalni razmak između letova
            global_time = departure_time + timedelta(minutes=5)
            assigned = True

        #ako nijedan avion nije slobodan dodeli sa najranijim vremenom povratka
        if not assigned:
            best_departure_time = None
            best_t = None
            best_a = None
            best_cost = float('inf')

            for t in range(T):
                if distance_r[r] > range_t[t]:
                    continue
                for a in range(A_t[t]):
                    candidate_departure_time = available_times[t][a]
                    cost = F_t_r[r][t]
                    # biraj najranijeg; ako je isto vreme, biraj manji trošak
                    if (best_departure_time is None or candidate_departure_time < best_departure_time
                        or (candidate_departure_time == best_departure_time and cost < best_cost)):
                        best_departure_time = candidate_departure_time
                        best_t, best_a = t, a
                        best_cost = cost

            if best_t is not None:
                code[r, best_t] = 1
                departure_time = best_departure_time
                return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])

                daily_limit = short_range_limit if range_t[best_t] <= 5000 else long_range_limit
                if flight_hours[best_t][best_a] + (T_r[r] + H_t[best_t]) >= daily_limit:
                    available_times[best_t][best_a] = return_time + timedelta(hours=wait_time_hours)
                    flight_hours[best_t][best_a] = 0
                else:
                    available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)
                    flight_hours[best_t][best_a] += T_r[r] + H_t[best_t] + turnaround_time / 60


    return code


def evaluate_solution(x_rt, base_time):
        total_cost = 0
        total_profit = 0
        turnaround_time = 45

         # Pretvaranje base_time u datetime ako je prosleđeno kao string
        if isinstance(base_time, str):
            base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M")

        available_times = {t: [base_time] * A_t[t] for t in all_types}
        flight_hours = {t: [0] * A_t[t] for t in all_types}
        global_time = base_time
        uncovered_routes = []

        for r in all_routes:
            route_cost = 0
            route_profit = 0
            assigned = False

            # Pronalazi prvi slobodan avion unutar svih tipova
            earliest_time = max(global_time, min([min(available_times[t]) for t in range(len(all_types))]))


            for t in all_types:
                if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?

                    if distance_r[r] > range_t[t]:
                        continue  # Preskoči avione sa nedovoljnim dometom

                    for a in specific_planes[t]:
                        if (
                            available_times[t][a] <= earliest_time
                            # flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                        ):

                            # Provera dnevnog limita za male i velike avione
                            daily_limit = short_range_limit if range_t[t] <= 5000 else long_range_limit

                            # Ažuriraj vreme polaska i povratka
                            departure_time = earliest_time
                            return_time = departure_time + timedelta(hours=T_r[r] + H_t[t])
                            
                            if flight_hours[t][a] + (T_r[r] + H_t[t]) >= daily_limit:
                                # Ažuriraj dostupnost sa čekanjem na pisti
                                available_times[t][a] = return_time + timedelta(hours=wait_time_hours)
                                flight_hours[t][a] = 0  # Resetuj sate posle čekanja
                            else:
                                # Ažuriraj dostupnost i radne sate za trenutno avion
                                available_times[t][a] = return_time + timedelta(minutes=turnaround_time)
                                flight_hours[t][a] += T_r[r] + H_t[t] + turnaround_time / 60

                            cost = F_t_r[r][t]
                            assigned = True
                            # print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                            # f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                            route_cost += cost
                            route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]
                            
                            # Povećaj globalno vreme
                            global_time = departure_time + timedelta(minutes=5)

                            break

            if not assigned:
                best_departure_time = None
                best_t = None
                best_a = None
                best_cost = float('inf')
                best_wait_penalty = float('inf')  # Početna vrednost kazne čekanja

                for t in all_types:
                    if x_rt[r, t] == 1:
                        if distance_r[r] > range_t[t]:
                            continue  # Preskoči avione sa nedovoljnim dometom

                        for a in specific_planes[t]:
                            candidate_departure_time = available_times[t][a]
                            cost = F_t_r[r][t]  # Trošak aviona na toj ruti

                            # Kazna za čekanje aviona
                            wait_penalty = 0
                            if candidate_departure_time > earliest_time:
                                wait_penalty = (candidate_departure_time - earliest_time).total_seconds() / 3600  # Sati čekanja
                                wait_penalty *= 10  # Penalizacija po satu čekanja

                            # Pronađi avion sa najranijim mogućim polaskom
                            if best_departure_time is None or candidate_departure_time < best_departure_time:
                                best_departure_time = candidate_departure_time
                                best_t = t
                                best_a = a
                                best_cost = cost
                                best_wait_penalty = wait_penalty  # Pamti penalizaciju za najbolji avion

                # Dodeli avion sa najranijim mogućim polaskom
                if best_t is not None and best_a is not None:
                    departure_time = best_departure_time
                    return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])

                    # Provera dnevnog limita
                    daily_limit = short_range_limit if range_t[best_t] <= 5000 else long_range_limit
                    if flight_hours[best_t][best_a] + (T_r[r] + H_t[best_t]) >= daily_limit:
                        # Penalizacija zbog čekanja na pisti
                        available_times[best_t][best_a] = return_time + timedelta(hours=wait_time_hours)
                        flight_hours[best_t][best_a] = 0  # Reset radnog vremena
                    else:
                        # Ažuriraj dostupnost i radno vreme
                        available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)
                        flight_hours[best_t][best_a] += T_r[r] + H_t[best_t] + turnaround_time / 60

                    route_cost += best_cost + best_wait_penalty
                    route_profit += min(P_r[r], C_t[best_t]) * price_per_passenger[r]

                    assigned = True

            if  not assigned:  #Ruta nije pokrivena
                uncovered_routes.append(r)

            # Dodaj ukupne troškove i profit za rutu
            total_cost += route_cost
            total_profit += route_profit

        # F-ja koju minimizujemo
        objective = alpha * total_cost - beta * total_profit

        return (len(uncovered_routes), objective)

_t0 = time.perf_counter()
code = greedy_algorithm(R, T, A_t, T_r, H_t, max_hours, F_t_r, base_time)

uncovered_routes, fitness = evaluate_solution(code, base_time)
_t1 = time.perf_counter()
runtime_s = _t1 - _t0

print(f'Nepokrivene rute {uncovered_routes} , fitness:{fitness}')
#print(f"RESULT,GREEDY,{selected_dataset},{uncovered_routes},{fitness:.6f},{runtime_s:.3f}")
