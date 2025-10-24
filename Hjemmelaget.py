import numpy as np
import pandas as pd
import time as time


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

def beta_funk(rbarp,rbarm,kolonne_fond,Marked,var):
    teller_liste=[]
    for i in range(len(Marked)-1):
        teller=(np.log(kolonne_fond[i+1]/kolonne_fond[i])-rbarp)*(np.log(Marked[i+1]/Marked[i])-rbarm)
        teller_liste.append(teller)
    teller=sum(teller_liste)
    cov=teller/(len(Marked)-1)
    beta=cov/var
    return beta

def jensens_alpha(rp,rm,beta):
    rf=0.03
    return rp-rf-beta*(rm-rf)


    


def sharp_ratio(rp,rm,varp,varm):
    rf=0.03
    sharp_p=(rp-rf)/np.sqrt(varp)
    sharp_m=(rm-rf)/np.sqrt(varm)
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
Marked = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\gspc_download.csv"
dfm = pd.read_csv(Marked)
dfp = pd.read_csv(Nordea_stabil_avskastning)
dfp['Close'] = dfp['Close'].fillna(method='ffill')
kolonne_fond = dfp['Close'].tolist()
Marked = dfm['Close'].dropna().tolist()


Marked = Marked[1:]
Marked = [float(x) for x in Marked]
rbarp=finn_daglig_gjennomsnitt_r(kolonne_fond)
varp=finn_varians(kolonne_fond,rbarp)
rp=np.log(kolonne_fond[-1]/kolonne_fond[0])
rbarm=finn_daglig_gjennomsnitt_r(Marked)
varm=finn_varians(Marked,rbarm)
rm=np.log(Marked[-1]/Marked[0])
beta=beta_funk(rbarp,rbarm,kolonne_fond,Marked,varm)

print("Daglig")
print("glidene logaritmisk gjennomsnitt:", rbarp*100,"%")
print("Standardavviket:",np.sqrt(varp)*100,"%")
print("")
print("Årlig Nordea")
print("Logaritmisk avkastning:", rp*100,"%")
print("Aretmetisk avkastning:" ,(kolonne_fond[-1]-kolonne_fond[0])/(kolonne_fond[0])*100,"%")
print("Standardavviket:",np.sqrt(varp)*100*np.sqrt(len(kolonne_fond)))
print("Årlig marked")
print("Logaritmisk avkastning:", rm*100,"%")
print("Aretmetisk avkastning:" ,(Marked[-1]-Marked[0])/(Marked[0])*100,"%")
print("Standardavviket:",np.sqrt(varm)*100*np.sqrt(len(Marked)))
print("beta:",beta)     
print("jensen's alpha:", jensens_alpha(rp,rm,beta))
print("Sharp ratio Norde, marked:", sharp_ratio(rp,rm,varp,varm))
print("treynor ratio nordea:", Treynor_ratio(rp,beta))
print("M_square:", M_square(rp,varp,varm,rm))
        


