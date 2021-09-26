from GP_parameter_fit import GP_PP_finder
import json

with open("./classes/Countries/wave_regions.json","r") as file:
    waves = json.load(file)

combinations = [['Germany', 1],['Sweden', 1], ['UnitedStates', 1]]

for comb in combinations:
    
   # N = 1# waves[country]["N_waves"]

    print(f"Country: {comb[0]} | Wave: {comb[1]}")


    params_real = {
        "file":f"{comb[0]}.txt",
        "wave":comb[1],
        "full":False,
        "use_running_average":True,
        "dt_running_average":14
    }

    gp = GP_PP_finder(N_initial_PP_samples = 60,iterations = 240)
#    gp = GP_PP_finder(N_initial_PP_samples = 2,iterations = 3)
    gp(params_real, use_calibrated_values=True)