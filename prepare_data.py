# Pipeline de données depuis CEPII,UNData, vers un unique .csv



# 1. Charger les flux (notre Y) - Fichier .csv
# 2. Charger les données CEPII Dyadiques (=couples de pays) (X statiques) - Fichier stata .dta
# 3. Charger les données CEPII Pays (X statiques) - Fichier .dta
# 4. Charger les données ONU (X dynamiques) - Fichiers .csv
# 5. Tout fusionner et sauvegarder.

import pandas as pd
import sys
import warnings

# --- 1. PARAMÈTRES GLOBAUX ---
# (Normalement complets, mais à re-vérifier)


PATHS = {
    'flows': 'data/azoseRaftery2019flows.csv',     # Fichier .csv des flux (Azose & Raftery)
    'cepii_dyadic': 'data/dist_cepii.dta',       # Fichier .dta de Geodist (Dyadic file)
    'cepii_country': 'data/geo_cepii.dta',      # Fichier .dta de Geodist (Country-specific file)
    'un_data_pop': 'data/population_un.csv', # Fichier .csv de UN Data (Pop, 0-14, 60+)
    'un_data_imr': 'data/mortality_un.csv',  # Fichier .csv de UN Data (Mortalité)
    'un_data_urban': 'data/urban_un.csv'       # Fichier .csv de UN Data (Urbanisation)
}


# dictionnaires qui va contenir les colonnes
COLS = {
    # Fichier Flows (.csv)
    'origin': 'origin',              # Le CODE NUMÉRIQUE M49 (ex: 533)
    'dest': 'destination',           # Le CODE NUMÉRIQUE M49 (ex: 4)
    'iso_origin': 'origIso',         # Code ISO (ex: 'ABW')
    'iso_dest': 'destIso',           # Code ISO (ex: 'AFG')
    'period': 'year',                # Année (ex: 1990)
    'flow': 'migrantCount',          # Le flux m_ijt (ex: 26)

    # Fichier CEPII Dyadic (.dta)
    'cepii_d_origin': 'iso_o',       # Clé de jointure ISO (ex: 'ABW')
    'cepii_d_dest': 'iso_d',         # Clé de jointure ISO (ex: 'AFG')
    'distance': 'distcap',           # Distance entre capitales (Utiliser 'distcap')
    'contiguous': 'contig',          # Frontière commune (1 ou 0)
    'comlang': 'comlang_off',        # Langue officielle commune (1 ou 0)
    'colony': 'colony',              # Lien colonial (1 ou 0)

    # Fichier CEPII Country (.dta)
    'cepii_c_iso': 'iso3',           # Clé de jointure ISO (ex: 'ABW')
    'area': 'area',                  # Superficie
    'landlocked': 'landlocked',      # Enclavé (1 ou 0)

    # Fichier UN Data (.csv) - Format LONG
    'un_country_code': 'Region/Country/Area', # Clé de jointure M49 (ex: 533)
    'un_period': 'Year',                # Clé de jointure (ex: 1990)
    'un_series': 'Series',              # La colonne qui contient les noms des variables
    'un_value': 'Value',                # La colonne qui contient les valeurs

    # --- NOMS EXACTS TROUVÉS DANS LES CSV DE L'ONU ---
    'un_series_pop': 'Population mid-year estimates (millions)',
    'un_series_age_0_14': 'Population aged 0 to 14 years old (percentage)', # Trouvé dans population_un.csv
    'un_series_age_60_plus': 'Population aged 60+ years old (percentage)', # Trouvé dans population_un.csv
    'un_series_imr': 'Under five mortality rate for both sexes (per 1,000 live births)', # Trouvé dans mortality_un.csv (PROXY pour IMR)
    'un_series_urban': 'Urban population (percent)'  # Trouvé dans urban_un.csv
}

# --- 2. FONCTIONS DE CHARGEMENT ---

def safe_read_csv(path):
    """Tente de lire un CSV avec plusieurs encodages et séparateurs."""
    for sep in [',', ';']:
        for encoding in ['utf-8', 'latin1']:
            try:
                # Ajout de low_memory=False et header=1 (pour sauter les titres FR)
                return pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, low_memory=False, header=1)
            except Exception:
                # Essayer header=0 si header=1 échoue
                try:
                    return pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, low_memory=False, header=0)
                except Exception:
                    continue
    print(f"ERREUR FATALE: Impossible de lire le fichier CSV '{path}'. Vérifiez le séparateur et l'encodage.")
    sys.exit()

def safe_read_stata(path):
    """Tente de lire un fichier Stata .dta."""
    try:
        return pd.read_stata(path, convert_categoricals=False, convert_dates=False)
    except FileNotFoundError:
        print(f"ERREUR FATALE: Le fichier '{path}' n'a pas été trouvé.")
        sys.exit()
    except Exception as e:
        print(f"ERREUR FATALE: Impossible de lire le fichier Stata '{path}'. Assurez-vous d'avoir installé 'pyreadstat'. Erreur: {e}")
        sys.exit()

def load_flows(path):
    """Charge la table de base (les flux) et la filtre."""
    print(f"Chargement des flux depuis {path}...")
    # Le fichier flux a un header simple
    df = pd.read_csv(path, sep=',', dtype=str, low_memory=False, header=0) 
    
    # Garder toutes les clés (M49 et ISO) + période et flux
    cols_to_keep = [COLS['origin'], COLS['dest'], COLS['iso_origin'], COLS['iso_dest'], COLS['period'], COLS['flow']]
    df = df[cols_to_keep]
    
    # Conversion en numérique
    for col in [COLS['origin'], COLS['dest'], COLS['period'], COLS['flow']]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=[COLS['origin'], COLS['dest'], COLS['period'], COLS['flow']])
    
    # --- FILTRE CLÉ DU MODÈLE DE GRAVITÉ ---
    df = df[df[COLS['flow']] > 0].copy()
    
    print(f"-> Flux chargés et filtrés (m_ijt > 0). {len(df)} observations.")
    return df

def load_cepii_dyadic(path):
    """Charge les données statiques DYADIQUES (distance, langue...) du CEPII."""
    print(f"Chargement des données CEPII dyadiques depuis {path}...")
    df = safe_read_stata(path)
    
    # Liste des colonnes statiques dont nous avons besoin
    static_cols = [
        COLS['cepii_d_origin'], COLS['cepii_d_dest'], COLS['distance'], 
        COLS['contiguous'], COLS['comlang'], COLS['colony']
    ]
    cols_to_use = [col for col in static_cols if col in df.columns]
    df = df[cols_to_use]
    
    # S'assurer que les clés de jointure sont des strings
    df[COLS['cepii_d_origin']] = df[COLS['cepii_d_origin']].astype(str)
    df[COLS['cepii_d_dest']] = df[COLS['cepii_d_dest']].astype(str)
    
    print(f"-> Données CEPII dyadiques chargées. {len(df)} paires de pays.")
    return df

def load_cepii_country(path):
    """Charge les données statiques PAYS (superficie, enclavé...) du CEPII."""
    print(f"Chargement des données CEPII pays depuis {path}...")
    df = safe_read_stata(path)
    
    # Liste des colonnes statiques dont nous avons besoin
    country_cols = [COLS['cepii_c_iso'], COLS['area'], COLS['landlocked']]
    cols_to_use = [col for col in country_cols if col in df.columns]
    df = df[cols_to_use]
    
    # S'assurer que la clé de jointure est un string
    df[COLS['cepii_c_iso']] = df[COLS['cepii_c_iso']].astype(str)
    
    print(f"-> Données CEPII pays chargées. {len(df)} pays.")
    return df

def load_un_dynamic(path_list):
    """
    Charge UN ou PLUSIEURS fichiers ONU au format LONG
    et les PIVOTE au format WIDE.
    """
    print(f"Chargement des données dynamiques ONU (long format) depuis: {path_list}")
    
    df_list = []
    for path in path_list:
        try:
            df_list.append(safe_read_csv(path))
        except Exception as e:
            print(f"Attention: Impossible de charger le fichier {path}. Il sera ignoré. Erreur: {e}")
            
    df_long = pd.concat(df_list, ignore_index=True)

    # 1. Garder uniquement les colonnes et séries dont on a besoin
    series_needed = [
        COLS['un_series_pop'],
        COLS['un_series_age_0_14'],    # Pour le calcul du PSR
        COLS['un_series_age_60_plus'], # Pour le calcul du PSR
        COLS['un_series_imr'],
        COLS['un_series_urban']
    ]
    
    cols_needed = [COLS['un_country_code'], COLS['un_period'], COLS['un_series'], COLS['un_value']]
    
    # S'assurer que les colonnes nécessaires existent
    cols_to_use_long = [col for col in cols_needed if col in df_long.columns]
    if not cols_to_use_long:
        print("ERREUR: Aucune des colonnes ONU attendues n'a été trouvée. Vérifiez COLS.")
        sys.exit()
        
    df_long = df_long[cols_to_use_long]
    
    # Filtrer les séries
    df_long = df_long[df_long[COLS['un_series']].isin(series_needed)]

    # 2. Nettoyer les valeurs
    df_long[COLS['un_value']] = df_long[COLS['un_value']].astype(str).str.replace(',', '', regex=False)
    df_long[COLS['un_value']] = pd.to_numeric(df_long[COLS['un_value']], errors='coerce')
    
    # Nettoyer les clés
    df_long[COLS['un_country_code']] = pd.to_numeric(df_long[COLS['un_country_code']], errors='coerce')
    df_long[COLS['un_period']] = pd.to_numeric(df_long[COLS['un_period']], errors='coerce')
    df_long = df_long.dropna(subset=[COLS['un_country_code'], COLS['un_period'], COLS['un_value']])
    
    # 3. Pivoter le DataFrame
    print("Pivotage du/des fichier(s) ONU...")
    df_wide = df_long.pivot_table(
        index=[COLS['un_country_code'], COLS['un_period']],
        columns=COLS['un_series'],
        values=COLS['un_value']
    ).reset_index()
    
    # Renommer les colonnes avec les noms de variables plus simples
    # (Les noms exacts des séries de COLS sont maintenant les noms des colonnes)
    df_wide = df_wide.rename(columns={
        COLS['un_series_pop']: 'Population',
        COLS['un_series_age_0_14']: 'age_0_14_pct',
        COLS['un_series_age_60_plus']: 'age_60_plus_pct',
        COLS['un_series_imr']: 'IMR', # C'est en fait U5MR, un proxy
        COLS['un_series_urban']: 'Urban'
    })
    
    # 4. Ajuster la Population (le fichier dit "millions")
    if 'Population' in df_wide.columns:
        df_wide['Population'] = df_wide['Population'] * 1_000_000
    
    print(f"-> Données ONU pivotées. {len(df_wide)} observations pays-année.")
    return df_wide

# --- 3. FONCTION PRINCIPALE (Le Pipeline) ---

def main():
    """Orchestre le chargement, la fusion et la sauvegarde."""
    print("--- Démarrage du pipeline 'Modèle de Gravité' ---")

    # --- ÉTAPE 1 : CHARGEMENT ---
    df_base = load_flows(PATHS['flows'])
    df_cepii_dyadic = load_cepii_dyadic(PATHS['cepii_dyadic'])
    df_cepii_country = load_cepii_country(PATHS['cepii_country'])
    
    # On charge TOUS les fichiers ONU que vous avez listés
    un_files_to_load = [PATHS['un_data_pop'], PATHS['un_data_imr'], PATHS['un_data_urban']] 
    
    df_un = load_un_dynamic(un_files_to_load)

    # --- ÉTAPE 2 : FUSIONS ---
    print("Démarrage des fusions...")
    
    # A. Fusion Statique (CEPII Dyadique) - sur ISO
    master_df = pd.merge(
        df_base,
        df_cepii_dyadic,
        left_on=[COLS['iso_origin'], COLS['iso_dest']],
        right_on=[COLS['cepii_d_origin'], COLS['cepii_d_dest']],
        how='inner'
    )
    print(f"Après fusion CEPII Dyadique, shape: {master_df.shape}")

    # B. Fusion Statique (CEPII Pays Origine) - sur ISO
    master_df = pd.merge(
        master_df,
        df_cepii_country.add_suffix('_i'), # Ajoute '_i' à 'area', 'landlocked'
        left_on=COLS['iso_origin'],
        right_on=COLS['cepii_c_iso'] + '_i',
        how='inner'
    )
    print(f"Après fusion CEPII Pays (Origine), shape: {master_df.shape}")
    
    # C. Fusion Statique (CEPII Pays Destination) - sur ISO
    master_df = pd.merge(
        master_df,
        df_cepii_country.add_suffix('_j'), # Ajoute '_j' à 'area', 'landlocked'
        left_on=COLS['iso_dest'],
        right_on=COLS['cepii_c_iso'] + '_j',
        how='inner'
    )
    print(f"Après fusion CEPII Pays (Destination), shape: {master_df.shape}")

    # D. Fusion Dynamique n°1 (Données de l'Origine 'i') - sur M49 + Période
    master_df = pd.merge(
        master_df,
        df_un.add_suffix('_i'), # Ajoute '_i' à 'Population', 'IMR', etc.
        left_on=[COLS['origin'], COLS['period']],
        right_on=[COLS['un_country_code'] + '_i', COLS['un_period'] + '_i'],
        how='inner'
    )
    print(f"Après fusion ONU (Origine), shape: {master_df.shape}")

    # E. Fusion Dynamique n°2 (Données de la Destination 'j') - sur M49 + Période
    master_df = pd.merge(
        master_df,
        df_un.add_suffix('_j'), # Ajoute '_j' à 'Population', 'IMR', etc.
        left_on=[COLS['dest'], COLS['period']],
        right_on=[COLS['un_country_code'] + '_j', COLS['un_period'] + '_j'],
        how='inner'
    )
    print(f"Après fusion ONU (Destination), shape: {master_df.shape}")

    # --- ÉTAPE 3 : CALCUL DES VARIABLES COMPOSITES (PSR) ---
    print("Calcul des variables composites (PSR)...")
    
    # Calcul de la part des 15-59 ans (proxy pour 15-64)
    # C'est la soustraction que vous avez suggérée
    master_df['age_15_59_pct_i'] = 100 - master_df['age_0_14_pct_i'] - master_df['age_60_plus_pct_i']
    master_df['age_15_59_pct_j'] = 100 - master_df['age_0_14_pct_j'] - master_df['age_60_plus_pct_j']
    
    # Calcul du PSR (proxy 15-59 / 60+)
    # On gère la division par zéro (si age_60_plus_pct == 0)
    master_df['PSR_i_t'] = master_df['age_15_59_pct_i'] / master_df['age_60_plus_pct_i'].replace(0, pd.NA)
    master_df['PSR_j_t'] = master_df['age_15_59_pct_j'] / master_df['age_60_plus_pct_j'].replace(0, pd.NA)
    
    
    # --- ÉTAPE 4 : NETTOYAGE FINAL ET SAUVEGARDE ---
    
    # Renommer les colonnes finales pour qu'elles soient simples
    master_df = master_df.rename(columns={
        COLS['flow']: 'm_ijt', COLS['distance']: 'D_ij',
        COLS['contiguous']: 'LB_ij', COLS['comlang']: 'OL_ij',
        COLS['colony']: 'COL_ij',
        COLS['area'] + '_i': 'LA_i',
        COLS['area'] + '_j': 'LA_j',
        COLS['landlocked'] + '_i': 'LL_i',
        COLS['landlocked'] + '_j': 'LL_j',
        'Population_i': 'P_i_t', 'Population_j': 'P_j_t',
        'IMR_i': 'IMR_i_t', 'IMR_j': 'IMR_j_t', # C'est le proxy U5MR
        'Urban_i': 'urban_i_t', 'Urban_j': 'urban_j_t',
        COLS['period']: 'period',
        COLS['dest']: 'dest'
    })
    
    final_cols = [
        'origin', 'dest', 'period', 'm_ijt', 
        'P_i_t', 'P_j_t', 'D_ij', 
        'PSR_i_t', 'PSR_j_t', 'IMR_i_t', 'IMR_j_t',
        'urban_i_t', 'urban_j_t', 'LA_i', 'LA_j',
        'LL_i', 'LL_j', 'LB_ij', 'OL_ij', 'COL_ij'
    ]
    
    final_cols_exist = [col for col in final_cols if col in master_df.columns]
    master_df = master_df[final_cols_exist]
    
    print(f"Données manquantes avant nettoyage final:\n {master_df.isnull().sum()}")
    master_df = master_df.dropna()
    print(f"Shape final après suppression des NaN: {master_df.shape}")

    output_path = 'gravity_data.csv'
    master_df.to_csv(output_path, index=False)
    
    print("--- Pipeline terminé ! ---")
    print(f"Le fichier final '{output_path}' est prêt pour l'analyse.")
    print(master_df.head())

if __name__ == "__main__":
    main()