For our project, we mostly want to demonstrate SHAP explanations used to back up predictions made by a random forest. In our case, we wanted to focus on the tactical level of warfare. 
Inspired by the near boarding of a Chinese vessel by U.S. operatives due to misleading A.I. inputs, we wanted to show how our agent can provide reasoning for commanders for the decisions made. 

https://www.kaggle.com/datasets/residentmario/database-of-battles?resource=download

Battle.csv

**Dataset Column Names and Definitions**
isqno — Unique battle identification number
war — Name of the war
name — Name of the battle
locn — Geographic location of the battle
campgn — Military campaign
postype — Type of military position or posture
post1 — Primary battlefield position or defensive posture
post2 — Secondary battlefield position or defensive posture
front — Battlefield frontage (tentative)
depth — Defensive depth (tentative)
time — Time-related battlefield factor (tentative)
aeroa — Air superiority advantage
surpa — Surprise advantage
cea — Combat effectiveness advantage (tentative)
leada — Leadership advantage
trnga — Training advantage
morala — Morale advantage
logsa — Logistics advantage
momnta — Momentum advantage (tentative)
intela — Intelligence advantage
techa — Technological advantage
inita — Initiative advantage
wina — Battle outcome indicator (exact coding unverified)
kmda — Distance advanced in kilometers (tentative)
crit — Criterion for determining battle outcome (tentative)
quala — Force quality
resa — Reserve-force advantage (tentative)
mobila — Mobility advantage
aira — Airpower-related factor
fprepa — Preparatory fire advantage (tentative)
wxa — Weather-related factor
terra — Terrain-related factor
leadaa — Leadership-related assessment
plana — Planning-related assessment
surpaa — Surprise-related assessment
mana — Maneuver-related assessment
logsaa — Logistics-related assessment
fortsa — Fortification-related assessment
deepa — Depth-related assessment
is_hero — Binary indicator (definition unverified)
war2 — Alternative war classification or name
war3 — Additional war classification or name
war4 — Standardized historical war name
war4_theater — Theater of operations
dbpedia — DBpedia reference for the battle
cow_warno — Correlates of War identification number
cow_warname — Correlates of War war name
war_initiator — War initiation indicator
parent — Related or parent record identifier (definition unverified)
