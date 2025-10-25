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

def beta_funk(rbarp_årlig,rbarm,kolonne_fond,Marked,var):
    teller_liste=[]
    for i in range(len(Marked)-1):
        teller=(np.log(kolonne_fond[i+1]/kolonne_fond[i])-rbarp_årlig)*(np.log(Marked[i+1]/Marked[i])-rbarm)
        teller_liste.append(teller)
    teller=sum(teller_liste)
    cov=teller/(len(Marked)-1)
    beta=cov/var
    return beta

def jensens_alpha(rp_årlig,rm_årlig,beta):
    rf=0.03
    return rp_årlig-rf-beta*(rm_årlig-rf)


def sharp_årlig_ratio(rp_årlig,rm_årlig,varp_årlig,varm_årlig):
    rf=0.03
    sharp_årlig_p=(rp_årlig-rf)/(np.sqrt(varp_årlig)*100)
    sharp_årlig_m=(rm_årlig-rf)/(np.sqrt(varm_årlig)*100)
    return sharp_årlig_p,sharp_årlig_m


def Treynor_ratio(rp_årlig,beta):
    rf=0.03
    treynor_p=(rp_årlig-rf)/(beta)
    return treynor_p

def M_square(rp_årlig,varp_årlig,varm_årlig,rm_årlig):
    rf=0.03
    skalar=np.sqrt(varp_årlig)/np.sqrt(varm_årlig)
    rp_årligm=skalar*rp_årlig+(1-skalar)*rf
    M_square=rp_årligm-rm_årlig
    return M_square


def finne_og_sortere_tabell():

    Nordea_stabil_avskastning = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\Nordea_stabil_avkastning.csv"
    Marked = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\gspc_download.csv"
   
    dfm = pd.read_csv(Marked)
    dfp = pd.read_csv(Nordea_stabil_avskastning)

    # Konverter dato til datetime
    dfp['Date'] = pd.to_datetime(dfp['Date'], errors='coerce', format='%m/%d/%Y')
    dfm['Date'] = pd.to_datetime(dfm['Date'], errors='coerce', format='%Y-%m-%d')

    # Slå sammen på felles datoer, utrolig kjekkt bibliotek
    merged = pd.merge(dfm[['Date', 'Close']], dfp[['Date', 'Close']], on='Date', suffixes=('_m', '_p'))
    merged = merged.dropna()
    # Hent ut hver kolonne som en liste

    datoer = merged['Date'].tolist()
    kolonne_fond = merged['Close_p'].tolist()
    Marked = merged['Close_m'].dropna().tolist()


    kolonne_fond = kolonne_fond[1:]
    kolonne_fond = [float(x) for x in kolonne_fond]

    Marked = Marked[1:]
    Marked = [float(x) for x in Marked]
    return kolonne_fond, Marked

kolonne_fond,Marked=finne_og_sortere_tabell()
rbarp_årlig=finn_daglig_gjennomsnitt_r(kolonne_fond)
varp_årlig=finn_varians(kolonne_fond,rbarp_årlig)
rp_årlig=rbarp_årlig*len(kolonne_fond)
rbarm=finn_daglig_gjennomsnitt_r(Marked)
varm_årlig=finn_varians(Marked,rbarm)
rm_årlig=rbarm*len(Marked)
beta=beta_funk(rbarp_årlig,rbarm,kolonne_fond,Marked,varm_årlig)
print("DAGLIG:")
print("glidene logaritmisk gjennomsnitt:", rbarp_årlig*100,"%")
print("Standardavviket:",np.sqrt(varp_årlig)*100,"%")
print("")
print("ÅRLIG NORDEA:")
print("Logaritmisk avkastning:", rp_årlig*100,"%")
print("Aretmetisk avkastning:" ,(kolonne_fond[-1]-kolonne_fond[0])/(kolonne_fond[0])*100,"%")
print("Standardavviket:",np.sqrt(varp_årlig)*100*np.sqrt(len(kolonne_fond)))
print("")
print("ÅRLIG MARKEDSINDEKSEN:")
print("Logaritmisk avkastning:", rm_årlig*100,"%")
print("Aretmetisk avkastning:" ,(Marked[-1]-Marked[0])/(Marked[0])*100,"%")
print("Standardavviket:",np.sqrt(varm_årlig)*100*np.sqrt(len(Marked)))
print("")
print("VURDERING PÅ FONDET MOT MARKEDET:")
print("beta:",beta)     
print("jensen's alpha:", jensens_alpha(rp_årlig,rm_årlig,beta))
print("Sharp_årlig ratio Norde, marked:", sharp_årlig_ratio(rp_årlig,rm_årlig,varp_årlig,varm_årlig))
print("treynor ratio nordea:", Treynor_ratio(rp_årlig,beta))
print("M_square:", M_square(rp_årlig,varp_årlig,varm_årlig,rm_årlig))
print("Sigma_p/Sigma_m:",np.sqrt(varp_årlig)/np.sqrt(varm_årlig))


