library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(car)
library(dplyr)
library(estimatr)

# Cmd+Entrée avec curseur sur une ligne pour exécuter la ligne

file_path <- "/Users/romain/Desktop/Projets DS/ProjetStat/data/reg_gravity_CEPII.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)


# le modèle de gravité c'est la loi de la gravitation ! donc multiplicatif, prendre le log

# probleme math de base : ln(0) = -infty. 
# Solution A: exclure les zéros
# Solution B: migrants+=1 pour tous. Pb: pourquoi +1 et pas +0.001 ? 
# Solution A gardée. Faire un PPML plus tard 

df_pos <- subset(df, migrantCount > 0) #enlève 100 000 lignes



df_pos$l_migrants <- log(df_pos$migrantCount)
df_pos$l_dist     <- log(df_pos$distw_harmonic) # distance harmonique pondérée de toutes les grandes villes
df_pos$l_pop_o    <- log(df_pos$pop_o)
df_pos$l_pop_d    <- log(df_pos$pop_d)
df_pos$l_gdpcap_o <- log(df_pos$gdp_o) - df_pos$l_pop_o
df_pos$l_gdpcap_d <- log(df_pos$gdp_d) - df_pos$l_pop_d #PIB par tete, sinon PIB et pop , pop va capturer l'effet de PIB. Plus economique de regarder PIB par tete
# attention à la colinéarité, ne pas rajouter gdp_o ou gdp_d
df_pos$l_LA_o <- log(df_pos$LA_o)
df_pos$l_LA_d <- log(df_pos$LA_d)
# intégrer les variables t-2000 linéaire et quadratique, pour capter l'effet de la date. 
# cela résout le probleme de 'year' mélangées. 
# hypothèse: la physique de la migration est constante dans le temps (%), seul le VOLUME change.
df_pos$time_trend <- df_pos$year - 2000
df_pos$time_trend_sq <- df_pos$time_trend^2

model_ols_robust <- lm_robust(l_migrants ~ l_dist + l_pop_o + l_pop_d + l_gdpcap_o + l_gdpcap_d + l_LA_o + l_LA_d + 
                  contig + comlang_off + col_dep_ever + LL_o + LL_d + time_trend + time_trend_sq, 
                data = df_pos)

# 5. Résultats
summary(model_ols_robust)

# on a supposé l'homoscédasticité. On a les bons coeff de beta, mais les p-value sont fausses! 












