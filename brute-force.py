import itertools
import numpy as np

# R = 5  # Broj ruta
# T = 2  # Broj tipova aviona
# A_t = {0: 3, 1: 2}  # Broj aviona po tipu
# P_r = [100, 150, 200, 180, 120]  # Broj putnika za svaku rutu
# F_t_r = [ 
#     [500, 600],  # Ruta 1, trošak za tipove 0 i 1
#     [700, 800],  
#     [400, 500],  
#     [450, 550],  
#     [650, 750],  
# ]
# C_t = [200, 300]  # Kapacitet aviona po tipu
# T_r = [2, 3, 4, 2, 5]  # Vreme trajanja leta za svaku rutu
# H_t = [1, 1]  # Vreme provedeno na zemlji za svaki tip aviona
# alpha = 1 
# beta = 1  
# max_hours = 18  # Maksimalno vreme rada aviona

R = 5  # Broj ruta
T = 3  # Broj tipova aviona
A_t = {0: 2, 1: 1, 2:2}  # Broj aviona po tipu
P_r = [70, 120, 90, 150, 100]  # Broj putnika za svaku rutu
F_t_r = [ 
    [400, 550, 600],  # Ruta 1, trošak za tipove 0 i 1
    [500, 650, 750],  
    [450, 500, 650],  
    [600, 700, 800],
    [350, 500, 550]   
]
C_t = [100, 120, 150]  # Kapacitet aviona po tipu
T_r = [1.5, 2, 2.5, 3, 1.5]  # Vreme trajanja leta za svaku rutu
H_t = [0.5, 1, 0.5]  # Vreme provedeno na zemlji za svaki tip aviona
alpha = 1 
beta = 1  
max_hours = 10  # Maksimalno vreme rada aviona
price_per_passenger = [10, 15, 12, 20, 10]  # Cena karte po putniku za svaku rutu

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
        assigned = False  # Da li je ruta pokrivena


        for t in all_types:
            for a in specific_planes[t]:
                if available_times[t][a] > 0:
                    available_times[t][a] -= (current_time - previous_time)
                    if available_times[t][a] < 0: #Avion je slobodan
                        available_times[t][a]= 0


        for t in all_types:
            if x_rt[r, t] == 1:  # Ruta r pokrivena tipom t?
                best_a = None
                best_cost = float('inf')
                
                for a in specific_planes[t]:
                    if (
                        available_times[t][a] <= 0 and
                        flight_hours[t][a] + (T_r[r] + H_t[t]) <= max_hours
                    ):
                        cost = F_t_r[r][t]
                        print(f"Ruta {r}, Tip {t}, Avion {a} je validan. Trošak: {cost}, "
                          f"Trenutno radno vreme: {flight_hours[t][a]}, Dostupnost: {available_times[t][a]}")  # Debugging
                        if cost < best_cost:  # Avion sa najmanjim troskom
                            best_a = a
                            best_cost = cost

                # Avion je pronadjen?
                if best_a is not None:
                    print(f"Ruta {r} dodeljena tipu {t}, avionu {best_a}, trošak {best_cost}")  # Debugging linija
                    available_times[t][best_a] += T_r[r] + H_t[t]
                    flight_hours[t][best_a] += T_r[r] + H_t[t]
                    total_cost += best_cost
                    total_profit += min(P_r[r], C_t[t]) * price_per_passenger[r]
                    assigned = True
                    break

        if not assigned:  #Ruta nije pokrivena
            total_cost += 10000  # Penal

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