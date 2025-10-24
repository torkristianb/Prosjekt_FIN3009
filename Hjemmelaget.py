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

def beta(kolonne_fond,kolonne_marked):
    liste_marked=[]
    liste_portefolje=[]



def jensens_alpha(rp,rm,beta):
    rf=0.03
    return rp-rf-beta*(rm-rf)


    


def sharp_ratio(rp,rm,varp,varm):
    rf=0.03
    sharp_p=(rp-rf)/np.sqrt(varp)
    sharp_m=(rm-rf)/np.sqrt(varp)
    return sharp_p,sharp_m


def Treynor_ratio(rp,beta):
    rf=0.03
    treynor_p=(rp-rf)/beta
    return treynor_p


def M_square(rp,varp,varm,rm):
    rf=0.03
    skalar=np.sqrt(varp)/np.sqrt(varm)
    rpm=skalar*rp+(1-skalar)*rf
    M_square=rpm-rm
    return M_square


Nordea_stabil_avskastning = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\Nordea_stabil_avkastning.csv"
Marked = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\Nordea_stabil_avkastning.csv"
df = pd.read_csv(Marked)
df = pd.read_csv(Nordea_stabil_avskastning)
kolonne_fond = df['Close'].dropna().tolist()
marked = df['Close'].dropna().tolist()
rbarp=finn_daglig_gjennomsnitt_r(kolonne_fond)
varp=finn_varians(kolonne_fond,rbarp)
rp=rbarp*100*574
rbarm=finn_daglig_gjennomsnitt_r(Marked)
varm=finn_varians(Marked,rbarm)
rm=rbarm*100*574



print("Daglig")
print("glidene logaritmisk gjennomsnitt:", rbar*100,"%")
print("Standardavviket:",np.sqrt(varp)*100,"%")
print("")
print("Årlig")
print("Logaritmisk avkastning:", rbar*100*574,"%")
print("Aretmetisk avkastning:" ,(kolonne_fond[574]-kolonne_fond[0])/(kolonne_fond[0])*100,"%")
print("Standardavviket:",np.sqrt(varp)*100*np.sqrt(574))


        

        

