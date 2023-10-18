# Assembleur basé sur les graphes de Debruijn

Vous trouverez la description complète du TP [ici]( 
https://docs.google.com/document/d/1P4v3bHbSurD7RXA-ldVwmtNKGvWsnBae51RMGye_KLs/edit?usp=sharing).

## Introduction

L’objectif de ce TP sera d’assembler le génome de l’entérovirus A71. Ce génome présente l’intérêt d’être très court: 7408 nucléotides, linéaire et non segmenté.
Le fichier fastq dont vous disposez a été généré à l’aide du programme ART [Huang 2011] via la commande:
art_illumina -i eva71.fna -ef -l 100 -f 20 -o eva71 -ir 0 -dr 0 -ir2 0 -dr2 0 -na -qL 41 -rs 1539952693 
Les lectures ont une qualité maximale (41) et ne présentent pas d’insertion. Seuls les lectures correspondant aux brins 5’ -> 3’ vous sont ici fournies. 

Dans le dossier debruijn-tp/data/, vous trouverez:
eva71.fna : génome du virus d’intérêt
eva71_plus_perfect.fq: lectures 


## Installation des dépendances

Vous utiliserez les librairies networkx, pytest et pylint de Python:

```
pip3 install --user networkx pytest pylint pytest-cov
```

## Utilisation

Vous créerez un programme Python3 nommé debruijn.py dans le dossier debruijn/.  Il prendra en argument :
 -i fichier fastq single end
 -k taille des kmer (optionnel - default 21)
 -o fichier output avec les contigs

## Tests

Vous testerez vos fonctions à l’aide de la commande pytest --cov=debruijn à exécuter dans le dossier debruijn-tp/. En raison de cette contrainte, les noms des fonctions ne seront pas libre. Il sera donc impératif de respecter le nom des fonctions “imposées”, de même que leur caractéristique et paramètres. 
Vous vérifierez également la qualité syntaxique de votre programme en exécutant la commande: pylint debruijn.py

## Contact

En cas de questions, vous pouvez me contacter par email: amine.ghozlane[at]pasteur.fr
