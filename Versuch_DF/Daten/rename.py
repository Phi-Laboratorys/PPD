import os

data = "Reflexion"

for d in os.listdir(data):
    for f in os.listdir(data + "/" + d):
        os.rename(data + "/" + d + "/" + f, data + "/" + d + "/" + f.replace("Relexion", "Reflexion"))