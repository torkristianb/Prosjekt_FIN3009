import numpy as np
import pandas as pd


def finn_daglig_gjennomsnitt_r(kolonne_fond):
    daglig_r=[]
    for i in range(len(kolonne_fond)-1):
        r=np.log(kolonne_fond[i+1]/kolonne_fond[i])
        daglig_r.append(r)
    rbar=sum(daglig_r)/(len(daglig_r))
    return rbar




def finn_varians(kolonne_fond,rbar):
    varteller=[]
    for i in range(len(kolonne_fond)-1):
        r=np.log(kolonne_fond[i+1]/kolonne_fond[i])
        var1=(r-rbar)**2
        varteller.append(var1)

    varteller1=sum(varteller)
    var=varteller1/(len(varteller)-1)
    return var






Nordea_stabil_avskastning = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\Nordea_stabil_avkastning.csv"
df = pd.read_csv(Nordea_stabil_avskastning)
kolonne_fond = df['Open'].dropna().tolist()
rbar=finn_daglig_gjennomsnitt_r(kolonne_fond)
var=finn_varians(kolonne_fond,rbar)

print("Daglig")
print("glidene logaritmisk gjennomsnitt:", rbar*100,"%")
print("Standardavviket:",np.sqrt(var)*100,"%")
print("")
print("Årlig")
print("Logaritmisk avkastning:", rbar*100*574,"%")
print("Aretmetisk avkastning:" ,(kolonne_fond[574]-kolonne_fond[0])/(kolonne_fond[0])*100,"%")
print("Standardavviket:",np.sqrt(var)*100*np.sqrt(574))


        

        

