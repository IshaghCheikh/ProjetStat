#Chargement des packages
library(tidyverse)
library(sf)  # Permet de lire les données géométriques
library(raster)
library(dplyr)
library(spdep)
library(ggplot2)
library(readxl)
library(viridis)
library(tmap)

# Cloner le dépôt
system("git clone https://github.com/ngwelch/bayesFlow.git")
setwd("bayesFlow")

# Explorer le dépôt
list.files(pattern = "world")

#Charger les données
regions <- read.csv("local/data/200isoRegionCodes.csv")
f1 <- read.csv("local/data/abelCohen2019flowsv6.csv")
f2 <- read.csv("local/data/abelCohen2019flowsv6_flowdt.csv")
f3 <- read.csv("local/data/azoseRaftery2019flows.csv")
head(f2)
view(f3)

#-------Importation du Shapefile-------------------
wrld <- st_read(dsn ="world-administrative-boundaries.shp" , stringsAsFactors = F)
View(wrld)
par(mar = c(0, 0, 0, 0))
plot(st_geometry(wrld), col = "gray")


#---Constitution de la base
df_origin <- wrld %>%
  rename(orig = iso3, continent_origin = continent, geometry_origin = geometry)

df_dest <- wrld %>%
  rename(dest = iso3, continent_dest = continent, geometry_dest = geometry)

df_joined <- f2 %>%
  left_join(df_origin, by = "orig") %>%
  left_join(df_dest, by = "dest")

df <- df_joined[,c("year0","orig","dest","name.x","name.y","flow", "continent_origin",
                   "continent_dest","geometry_origin","geometry_dest")]
view(df)


#----Structure agrégée---
df_outflows <- df %>%
  group_by(year0, orig, continent_origin, geometry_origin) %>%           # Regrouper par pays d'origine et année
  summarise(
    total_outflow = sum(flow, na.rm = TRUE)  # Somme de tous les flux sortants
  ) %>%
  ungroup() %>%
  # Transformer en objet sf avec la géométrie correspondante
  st_as_sf(sf_column_name = "geometry_origin")


df_inflows <- df %>%
  group_by(year0, dest, continent_dest, geometry_dest) %>%           # Regrouper par pays d'origine et année
  summarise(
    total_inflow = sum(flow, na.rm = TRUE)  # Somme de tous les flux entrants
  ) %>%
  ungroup() %>%
  # Transformer en objet sf avec la géométrie correspondante
  st_as_sf(sf_column_name = "geometry_dest")

view(df_inflows)
view(df_outflows)
#------------------------------ANALYSE DESCRIPTIVE------------------------------

# Trouver le pays top émetteur par année
top_outflow_countries <- df_outflows %>%
  group_by(year0) %>%
  slice_max(total_outflow, n = 3) %>%
  ungroup()


# Trouver le pays top récepteur par année
top_inflow_countries <- df_inflows %>%
  group_by(year0) %>%
  slice_max(total_inflow, n = 3) %>%
  ungroup()


# Total global de flux par année

df_global_trends <- df_outflows %>%
  group_by(year0) %>%
  summarise(
    total_inflow = sum(total_outflow, na.rm = TRUE)  
  )

# Graphique d’évolution
ggplot(df_global_trends, aes(x = year0)) +
  geom_line(aes(y = total_inflow, color = "Entrants"), size = 1.2) +
  scale_color_manual(values = c("Entrants" = "darkgreen")) +
  labs(
    title = "Évolution mondiale des flux migratoires",
    x = "Année", y = "Nombre total de migrants",
    color = "Type de flux"
  ) +
  theme_minimal()


# Total cumulé par paire de pays
df_routes <- df %>%
  filter(year0 == 2000) %>%
  group_by(orig, dest) %>%
  summarise(total_flow = sum(flow, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(desc(total_flow))

# Top 10 routes migratoires
top_routes <- df_routes %>%
  slice_max(total_flow, n = 10) %>%
  ungroup()%>%
  arrange(desc(total_flow))

ggplot(top_routes, aes(
  x = reorder(paste(orig, "→", dest), total_flow),
  y = total_flow
)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 routes migratoires",
    x = "Origine → Destination",
    y = "Nombre de migrants"
  ) +
  theme_minimal(base_size = 13)

#--------------------Analyse continentale---------------------#

##----Analyse inter-------##
df_continent_flows <- df %>%
  filter(continent_origin != continent_dest) %>%
  group_by(continent_origin, continent_dest, year0) %>%
  summarise(total_flow = sum(flow, na.rm = TRUE)) %>%
  ungroup()

df_continent_flows = drop_na(df_continent_flows)
#------------Top flux entre continents Barplot-----------------
df_continent_flows %>%
  #filter(continent_origin != continent_dest) %>%
  group_by(continent_origin, continent_dest) %>%
  summarise(total_flow = mean(total_flow, na.rm = TRUE)) %>%
  arrange(desc(total_flow)) %>%
  #slice_head(n = 10) %>%
  ggplot(aes(
    x = reorder(paste(continent_origin, "→", continent_dest), total_flow),
    y = total_flow,
    fill = continent_origin,
  )) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Flux migratoires intercontinentaux (en moyenne)",
    x = "Origine → Destination",
    y = "Nombre total de migrants",
    fill = "Continent d’origine"
  ) +
  theme_minimal(base_size = 14)


#---Flux de sorties par continent au fil du temps------
df_out_by_continent <- df_continent_flows %>%
  group_by(year0, continent_origin) %>%
  summarise(total_outflow = sum(total_flow, na.rm = TRUE)) %>%
  ungroup()

ggplot(df_out_by_continent, aes(x = year0, y = total_outflow, color = continent_origin)) +
  geom_line(linewidth = 1.2) +
  geom_point() +
  theme_minimal(base_size = 13) +
  labs(
    title = "Évolution des flux sortants par continent",
    x = "Année", y = "Nombre total de migrants sortants",
    color = "Continent d’origine"
  )


#---Flux d'entrées par continent au fil du temps
df_in_by_continent <- df_continent_flows %>%
  group_by(year0, continent_dest) %>%
  summarise(total_inflow = sum(total_flow, na.rm = TRUE)) %>%
  ungroup()

ggplot(df_in_by_continent, aes(x = year0, y = total_inflow, color = continent_dest)) +
  geom_line(linewidth = 1.2) +
  geom_point() +
  theme_minimal(base_size = 13) +
  labs(
    title = "Évolution des flux entrants par continent",
    x = "Année", y = "Nombre total de migrants entrants",
    color = "Continent de destination"
  )

# Matrice des flux continentaux

df_continent_flows %>%
  filter(year0 == 2015) %>%  # ou une autre année
  ggplot(aes(x = continent_dest, y = continent_origin, fill = total_flow)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Flux migratoires intercontinentaux en 1990",
    x = "Continent de destination",
    y = "Continent d’origine",
    fill = "Migrants"
  ) +
  theme_minimal(base_size = 13)

# Diagramme des cordes pour montrer les liens
library(circlize)

df_links <- df_continent_flows %>%
  filter(continent_origin != continent_dest) %>%
  group_by(continent_origin, continent_dest) %>%
  summarise(total_flow = mean(total_flow, na.rm = TRUE)) %>%
  ungroup()

chordDiagram(df_links[, c("continent_origin", "continent_dest", "total_flow")])



#----Evolution relative depuis 1990 (flux sortants)--
df_indexed <- df_out_by_continent %>%
  group_by(continent_origin) %>%
  mutate(
    index_1990 = total_outflow[year0 == 1990],
    migration_index = (total_outflow / index_1990) * 100
  ) %>%
  ungroup()

ggplot(df_indexed, aes(x = year0, y = migration_index, color = continent_origin)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  labs(
    title = "Évolution relative des flux migratoires sortants par continent (base 1990 = 100)",
    x = "Année",
    y = "Indice (1990 = 100)",
    color = "Continent d’origine"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


#----Evolution relative depuis 1990 (flux entrants)--
df_indexed2 <- df_in_by_continent %>%
  group_by(continent_dest) %>%
  mutate(
    index_1990 = total_inflow[year0 == 1990],
    migration_index = (total_inflow / index_1990) * 100
  ) %>%
  ungroup()

ggplot(df_indexed2, aes(x = year0, y = migration_index, color = continent_dest)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  labs(
    title = "Évolution relative des flux migratoires entrants par continent (base 1990 = 100)",
    x = "Année",
    y = "Indice (1990 = 100)",
    color = "Continent d’origine"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")



##--------Analyse intra---------#

df_intra <- df %>%
  filter(continent_origin == continent_dest) %>%  
  filter(continent_origin %in% c("Asia", "Europe", "Americas")) %>%
  group_by(continent_origin, orig) %>%
  summarise(mean_flow = mean(flow, na.rm = TRUE)) %>%
  ungroup()

top5_intra_countries <- df_intra %>%
  group_by(continent_origin) %>%
  slice_max(mean_flow, n = 5) %>%
  ungroup()

ggplot(top5_intra_countries, aes(
  x = reorder(orig, mean_flow),
  y = mean_flow,
  fill = continent_origin
)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~continent_origin, scales = "free_y") +
  labs(
    title = "Top 5 pays avec les flux intra-continentaux les plus élevés (moyenne sur la période)",
    x = "Pays",
    y = "Flux moyen intra-continentaux",
    fill = "Continent"
  ) +
  theme_minimal(base_size = 13)


###--Boxplot de dispersion----###
df_intra %>%
  filter(orig %in% top5_intra_countries$orig) %>%
  ggplot(aes(x = orig, y = mean_flow, fill = continent_origin)) +
  #geom_violin(trim = FALSE, alpha = 0.8) +
  geom_boxplot(width = 0.15, color = "black", outlier.shape = NA) +
  facet_wrap(~continent_origin, scales = "free_y") +
  labs(
    title = "Dispersion des flux intra-continentaux pour les 5 pays les plus actifs",
    x = "Pays",
    y = "Flux intra-continentaux (distribution)",
    fill = "Continent"
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#-------Cartes chloropèthes--------

#-------Carte des inflows----
ggplot(df_inflows %>% 
         filter(year0 ==2015)) +
  geom_sf(aes(fill = total_inflow)) +
  theme_minimal() +
  scale_fill_gradient(name = "Total flux entrants",
                      high = "#A20000",
                      low = "#9CE09C")+
  theme(axis.title.x = element_blank(), # Supprimer l'étiquette de l'axe des X
        axis.title.y = element_blank(), # Supprimer l'étiquette de l'axe des Y
        axis.text = element_blank(),    # Supprimer les axes des X et Y
        legend.position = "left")+
  theme(plot.background = element_rect(fill = "lightgrey"))

#-------Carte des outflows----
ggplot(df_outflows %>% 
         filter(year0 ==1990)) +
  geom_sf(aes(fill = total_outflow)) +
  theme_minimal() +
  scale_fill_gradient(name = "Total flux sortants",
                      high = "#A20000",
                      low = "#9CE09C")+
  theme(axis.title.x = element_blank(), # Supprimer l'étiquette de l'axe des X
        axis.title.y = element_blank(), # Supprimer l'étiquette de l'axe des Y
        axis.text = element_blank(),    # Supprimer les axes des X et Y
        legend.position = "left")+
  theme(plot.background = element_rect(fill = "lightgrey"))
