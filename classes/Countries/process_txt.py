import numpy as np

file = "Zwischen_Speicher.txt"
target_file = "Russia.txt"
pop = 144526636

data = np.loadtxt(file,delimiter="\t") / pop
print(data)

np.savetxt(target_file,data)