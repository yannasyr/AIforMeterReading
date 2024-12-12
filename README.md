# Challenge Data ENS 2020 | AI for Meter Reading

## But personnel 

Découvrir des outils de DL/OCR/Computer Vision. Projet solo + temps limité. 

# État du projet

- Les modèles d'extraction des cadrans et des digits sont **OK**, mais pas ultra-performant. (*La labelisation des données étant à faire soi-même, j'ai réduit le nombre de data d'entrainement.*)*
- L'étape de lecture des chiffres n'est pas satisfaisante.
- L'étape de labelisation de la consommation d'eau n'est pas encore abordée.


## Description

Ce projet vise à résoudre un défi proposé par l'entreprise Suez, qui a fourni des images de compteurs d'eau prises par leurs utilisateurs. L'objectif est d'extraire avec précision la consommation en m³ d'eau affichée sur les cadrans. Le problème est ici abordé sous l'angle du Deep Learning.

## Pistes explorées 

- **Extraction des cadrans**  
  - Labelisation des cadrans avec [Label Studio](https://labelstud.io/).
  - Finetuning de [YOLO V11](https://docs.ultralytics.com/fr/models/yolo11/) sur ces données --> ```models/bbox_yolov11_200train.pt```

- **Extraction des digits**  
  - Extraction ciblée des chiffres en positions 1, 2, 3, 4 et 5 sur le cadran, représentant la consommation en m³.
  - Labelisation des cadrans avec Label Studio.
  - Finetuning de YOLO V11 sur ces données --> ```models/digits_yolov11```

- **CNN pour lire les digits**  
  - Entrainement d'un ConvNet sur MNIST.
  - Application du modèle sur les imagettes extraites à l'étape précédente.
