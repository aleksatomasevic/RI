from copy import deepcopy
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd


current_dir = Path(__file__).parent 
json_path = current_dir / "parameters.json"

with open(json_path, "r") as f:
    datasets = json.load(f)

# ovde biramo skup podatak s kojm radimo
selected_dataset = "dataset10"  
params = datasets[selected_dataset]


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

all_routes = range(R) 
all_types = range(T) 
specific_planes = {t: range(A_t[t]) for t in all_types}  # Skup specifičnih aviona unutar svakog tipa


short_range_limit = 16  # Maksimalni sati rada za male avione
long_range_limit = 20   # Maksimalni sati rada za velike avione
wait_time_hours = 6     # Vreme čekanja na pisti nakon prekoračenja limita


# Inicijalizacija rešenja (svaka ruta dobije nasumičan tip)
def initialize_vns():
    solution = np.zeros((R, T), dtype=int)
    for r in range(R):
        t = random.randint(0, T - 1)
        solution[r, t] = 1
    return solution


# Vraća tuple: (broj_nepokrivenih, objective) 
def evaluate_solution_vns(x_rt, base_time):
    total_cost = 0
    total_profit = 0
    turnaround_time = 45  # u minutama

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
 
        earliest_time = max(global_time, min([min(available_times[t]) for t in range(len(all_types))]))
        
        for t in all_types:
            if x_rt[r, t] == 1: 
                if distance_r[r] > range_t[t]:
                    continue
                for a in specific_planes[t]:
                    if available_times[t][a] <= earliest_time:
                        daily_limit = short_range_limit if range_t[t] <= 5000 else long_range_limit
                        departure_time = earliest_time
                        return_time = departure_time + timedelta(hours=T_r[r] + H_t[t])
                        if flight_hours[t][a] + (T_r[r] + H_t[t]) >= daily_limit:
                            available_times[t][a] = return_time + timedelta(hours=wait_time_hours)
                            flight_hours[t][a] = 0
                        else:
                            available_times[t][a] = return_time + timedelta(minutes=turnaround_time)
                            flight_hours[t][a] += T_r[r] + H_t[t] + turnaround_time / 60
                        cost = F_t_r[r][t]
                        assigned = True
                        route_cost += cost
                        route_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]
                        global_time = departure_time + timedelta(minutes=5)
                        break
        if not assigned:
            best_departure_time = None
            best_t = None
            best_a = None
            best_cost = float('inf')
            best_wait_penalty = float('inf')
            for t in all_types:
                if x_rt[r, t] == 1:
                    if distance_r[r] > range_t[t]:
                        continue
                    for a in specific_planes[t]:
                        candidate_departure_time = available_times[t][a]
                        cost = F_t_r[r][t]
                        wait_penalty = 0
                        if candidate_departure_time > earliest_time:
                            wait_penalty = (candidate_departure_time - earliest_time).total_seconds() / 3600 * 10
                        if best_departure_time is None or candidate_departure_time < best_departure_time:
                            best_departure_time = candidate_departure_time
                            best_t = t
                            best_a = a
                            best_cost = cost
                            best_wait_penalty = wait_penalty
            if best_t is not None and best_a is not None:
                departure_time = available_times[best_t][best_a]
                return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])
                daily_limit = short_range_limit if range_t[best_t] <= 5000 else long_range_limit
                if flight_hours[best_t][best_a] + (T_r[r] + H_t[best_t]) >= daily_limit:
                    available_times[best_t][best_a] = return_time + timedelta(hours=wait_time_hours)
                    flight_hours[best_t][best_a] = 0
                else:
                    available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)
                    flight_hours[best_t][best_a] += T_r[r] + H_t[best_t] + turnaround_time / 60
                route_cost += best_cost + best_wait_penalty
                route_profit += min(P_r[r], C_t[best_t]) * price_per_passenger[r]
                assigned = True
        if not assigned:
            uncovered_routes.append(r)
        total_cost += route_cost
        total_profit += route_profit

    objective = alpha * total_cost - beta * total_profit
    return (len(uncovered_routes), objective)


# Shaking: bira se k nasumičnih ruta i menja se dodeljeni tip
def shaking_vns(solution, k):
    new_solution = solution.copy()
    indices = random.sample(range(R), k)  
    for r in indices:
        current_type = int(np.argmax(new_solution[r]))
        new_types = [t for t in range(T) if t != current_type]
        if new_types:
            new_type = random.choice(new_types)
            new_solution[r] = np.zeros(T, dtype=int)
            new_solution[r, new_type] = 1
    return new_solution


# Lokalna pretraga 
def local_search_vns(solution, fitness, base_time):
    improved = True
    while improved:
        improved = False
        best_fitness = fitness
        best_move = None  # (indeks rute, novi tip)
        for r in range(R):
            current_type = int(np.argmax(solution[r]))
            for t in range(T):
                if t != current_type:
                    temp_solution = solution.copy()
                    temp_solution[r] = np.zeros(T, dtype=int)
                    temp_solution[r, t] = 1
                    new_fitness = evaluate_solution_vns(temp_solution, base_time)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_move = (r, t)
        if best_move is not None:
            r, t = best_move
            solution[r] = np.zeros(T, dtype=int)
            solution[r, t] = 1
            fitness = best_fitness
            improved = True
    return fitness


def vns_flight_assignment(num_iters, neighborhoods, move_prob, base_time):
    solution = initialize_vns()
    fitness = evaluate_solution_vns(solution, base_time)
    fitnesses = []
    for it in range(num_iters):
        for k in neighborhoods:
            new_solution = shaking_vns(solution, k)
            new_fitness = evaluate_solution_vns(new_solution, base_time)
            new_fitness = local_search_vns(new_solution, new_fitness, base_time)
            print(f'Novi fintess :{new_fitness}, iteracija: {it}')
            if new_fitness < fitness or (new_fitness == fitness and random.random() < move_prob):
                fitness = new_fitness
                solution = new_solution.copy()
                break
        fitnesses.append(fitness)
    
    objectives = [f[1] for f in fitnesses]  # uzimamo drugu komponentu (objective)
    plt.plot(objectives)
    plt.xlabel("Iteracija")
    plt.ylabel("Objective")
    plt.title("VNS - Progresija fitnesa")
    plt.show()
    
    return solution, fitness


def generate_flight_schedule(best_code, destinations, plane_types, specific_planes, T_r, base_time):
    flight_schedule = []
    flight_count = 1  
    turnaround_time = 45  # u minutama

    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M")
    
    available_times = {t: [base_time] * A_t[t] for t in range(len(all_types))}
    flight_hours = {t: [0] * A_t[t] for t in range(len(all_types))}
    global_time = base_time

    for r, row in enumerate(best_code):
        earliest_time = max(global_time, min([min(available_times[t]) for t in range(len(plane_types))]))
        assigned = False

        for t, value in enumerate(row):
            if value == 1:
                if distance_r[r] > range_t[t]:
                    continue 
                for a in specific_planes[t]:
                    if available_times[t][a] <= earliest_time:
                        daily_limit = short_range_limit if range_t[t] <= 5000 else long_range_limit
                        departure_time = earliest_time
                        return_time = departure_time + timedelta(hours=T_r[r] + H_t[t])
                        if flight_hours[t][a] + (T_r[r] + H_t[t]) >= daily_limit:
                            available_times[t][a] = return_time + timedelta(hours=wait_time_hours)
                            flight_hours[t][a] = 0
                        else:
                            available_times[t][a] = return_time + timedelta(minutes=turnaround_time)
                            flight_hours[t][a] += T_r[r] + H_t[t] + turnaround_time / 60
                        flight_schedule.append({
                            "Flight Number": flight_count,
                            "Destination": destinations[r],
                            "Plane Type": plane_types[t],
                            "Plane ID": f"T{t+1}-{a+1}",
                            "Departure Time": departure_time.strftime("%Y-%m-%d %H:%M"),
                            "Return Time": return_time.strftime("%Y-%m-%d %H:%M"),
                            "Departure Timestamp": departure_time,
                            "Return Timestamp": return_time
                        })
                        flight_count += 1
                        global_time = departure_time + timedelta(minutes=5)
                        assigned = True
                        break
            if assigned:
                break

        if not assigned:
            earliest_available = None
            best_t = None
            best_a = None
            best_cost = float('inf')
            best_wait_penalty = float('inf')
            for t in range(len(plane_types)):
                if best_code[r][t] == 1:
                    if distance_r[r] > range_t[t]:
                        continue
                    for a in specific_planes[t]:
                        candidate_departure_time = available_times[t][a]
                        cost = F_t_r[r][t]
                        wait_penalty = 0
                        if candidate_departure_time > earliest_time:
                            wait_penalty = (candidate_departure_time - earliest_time).total_seconds() / 3600 * 10
                        if earliest_available is None or candidate_departure_time < earliest_available:
                            earliest_available = candidate_departure_time
                            best_t = t
                            best_a = a
                            best_cost = cost
                            best_wait_penalty = wait_penalty
            if best_t is not None and best_a is not None:
                departure_time = available_times[best_t][best_a]
                return_time = departure_time + timedelta(hours=T_r[r] + H_t[best_t])
                daily_limit = short_range_limit if range_t[best_t] <= 5000 else long_range_limit
                if flight_hours[best_t][best_a] + (T_r[r] + H_t[best_t]) >= daily_limit:
                    available_times[best_t][best_a] = return_time + timedelta(hours=wait_time_hours)
                    flight_hours[best_t][best_a] = 0
                else:
                    available_times[best_t][best_a] = return_time + timedelta(minutes=turnaround_time)
                    flight_hours[best_t][best_a] += T_r[r] + H_t[best_t] + turnaround_time / 60
                flight_schedule.append({
                    "Flight Number": flight_count,
                    "Destination": destinations[r],
                    "Plane Type": plane_types[best_t],
                    "Plane ID": f"T{best_t+1}-{best_a+1}",
                    "Departure Time": departure_time.strftime("%Y-%m-%d %H:%M"),
                    "Return Time": return_time.strftime("%Y-%m-%d %H:%M"),
                    "Departure Timestamp": departure_time,
                    "Return Timestamp": return_time
                })
                flight_count += 1
                assigned = True

    df = pd.DataFrame(flight_schedule)
    df = df.sort_values(by="Departure Timestamp").drop(columns=["Departure Timestamp"])
    print("Raspored letova (sortirano po polasku):")
    print(df)
    
    dg = pd.DataFrame(flight_schedule)
    dg = dg.sort_values(by="Return Timestamp").drop(columns=["Return Timestamp"])
    print("\nRaspored letova (sortirano po povratku):")
    print(dg)
    
    return dg


num_iters = 500    
neighborhoods = [1, 2, 3]  
move_prob = 0.1           

best_solution, best_fitness = vns_flight_assignment(num_iters, neighborhoods, move_prob, base_time)
print("Najbolje rešenje (matrica dodele):")
print(best_solution)
print("Fitnes (broj nepokrivenih ruta, objective):", best_fitness)


flight_schedule_df = generate_flight_schedule(best_solution, destinations, plane_types, specific_planes, T_r, base_time)