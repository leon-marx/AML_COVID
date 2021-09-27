from GP_parameter_fit import GP_PP_finder
import json

with open("./classes/Countries/wave_regions.json","r") as file:
    waves = json.load(file)

countries = ['Germany']
waves = {'Germany': {
    "N_waves": 1,
    "1": [30,150]
}}

for country in waves.keys():
    N = waves[country]["N_waves"]

    print(f"\n{country}")

    for i in range(N):
        print(f"\t{i+1} of {N}")

        params_real = {
            "file":f"{country}.txt",
            "wave":i+1,
            "full":False,
            "use_running_average":True,
            "dt_running_average":14
        }

        gp = GP_PP_finder(N_initial_PP_samples = 60,iterations = 240)
        gp(params_real,use_calibrated_values=True)