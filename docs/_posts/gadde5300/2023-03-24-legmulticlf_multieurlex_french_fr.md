---
layout: model
title: Legal Multilabel Classification (MultiEURLEX, French)
author: John Snow Labs
name: legmulticlf_multieurlex_french
date: 2023-03-24
tags: [legal, classification, fr, licensed, multieurlex, open_source, tensorflow]
task: Text Classification
language: fr
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Text Classification model that can help you classify 16 types of German legal documents.

## Predicted Entities

`commission parlementaire`, `commerce des armes`, `Commission des droits de l'homme`, `commission ONU`, `commission PE`, `commerce extérieur`, `Ouest`, `adhésif`, `abus de confiance`, `commerce d'État`, `adjudication de marché`, `commerce intérieur`, `commerce international`, `commerce de gros`, `commerce de détail`, `Århus (comté)`, `commission d'enquête`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_french_fr_1.0.0_3.0_1679670521086.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_french_fr_1.0.0_3.0_1679670521086.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

embeddings = nlp.UniversalSentenceEncoder.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")

docClassifier = nlp.MultiClassifierDLModel().pretrained('legmulticlf_multieurlex_french', 'fr', 'legal/models')\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        embeddings,
        docClassifier
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_model = nlp.LightPipeline(model)

result = light_model.annotate("""Règlement (CE) no 925/2003 de la Commission
du 27 mai 2003
portant modalités d'application de la décision 2003/298/CE du Conseil en ce qui concerne les concessions sous forme de contingents tarifaires communautaires pour certains produits céréaliers en provenance de la République tchèque et modifiant le règlement (CE) n° 2809/2000
LA COMMISSION DES COMMUNAUTÉS EUROPÉENNES,
vu le traité instituant la Communauté européenne,
vu la décision 2003/298/CE du Conseil du 14 avril 2003 relative à la conclusion d'un protocole d'adaptation des aspects commerciaux de l'accord européen établissant une association entre les Communautés européennes et leurs États membres, d'une part, et la République tchèque, d'autre part, pour tenir compte des résultats des négociations entre les parties concernant l'établissement de nouvelles concessions agricoles réciproques(1), et notamment son article 3, paragraphe 2,
considérant ce qui suit:
(1) Conformément à la décision 2003/298/CE, la Communauté s'est engagée à établir pour chaque campagne de commercialisation des contingents tarifaires d'importation à droit nul pour le blé, le méteil, la farine de blé et de méteil, le malt et le maïs en provenance de la République tchèque.
(2) Afin de permettre l'importation réglementaire et non spéculative du blé et du maïs visés par ces contingents tarifaires, il y a lieu de subordonner ces importations à la délivrance d'un certificat d'importation. Les certificats sont délivrés à la demande des intéressés dans les limites des quantités fixées, moyennant, le cas échéant, la fixation d'un coefficient de réduction des quantités demandées.
(3) Pour assurer la bonne gestion de ces contingents, il convient de fixer la date limite de dépôt des demandes de certificat et de préciser les informations devant figurer dans les demandes et les certificats.
(4) Afin de veiller à ce que les produits importés de la République tchèque ne bénéficient d'aucune subvention à l'exportation et ne proviennent pas de stocks publics d'intervention, la demande de certificat d'importation et le certificat d'importation doivent être accompagnés d'un certificat spécial.
(5) Pour tenir compte des conditions de livraison, les certificats d'importation sont valables à compter du jour de leur délivrance jusqu'à la fin du mois suivant.
(6) Afin d'assurer une gestion efficace des contingents, il convient de prévoir des dérogations au règlement (CE) n° 1291/2000 de la Commission du 9 juin 2000 portant modalités communes d'application du régime des certificats d'importation, d'exportation et de préfixation pour les produits agricoles(2), modifié en dernier lieu par le règlement (CE) n° 325/2003(3), en ce qui concerne la transmissibilité des certificats et la tolérance relative aux quantités mises en libre pratique.
(7) Pour permettre une bonne gestion des contingents, il est nécessaire que la garantie relative aux certificats d'importation soit fixée à un niveau relativement élevé, par dérogation à l'article 10 du règlement (CE) n° 1162/95 de la Commission du 23 mai 1995 portant modalités d'application du régime des certificats d'importation et d'exportation dans le secteur des céréales et du riz(4), modifié en dernier lieu par le règlement (CE) n° 498/2003(5).
(8) Il importe d'assurer la transmission rapide et réciproque entre la Commission et les États membres des informations concernant les quantités demandées et importées.
(9) Le règlement (CE) n° 2433/2000 du Conseil du 17 octobre 2000 établissant certaines concessions sous forme de contingents tarifaires communautaires pour certains produits agricoles et prévoyant l'adaptation autonome et transitoire de certaines concessions agricoles prévues dans l'accord européen avec la République tchèque(6) ayant été abrogé par la décision 2003/298/CE, il convient de modifier en conséquence le règlement (CE) n° 2809/2000 de la Commission portant modalités d'application, pour les produits du secteur céréalier, des règlements (CE) n° 2290/2000, (CE) n° 2433/2000 et (CE) n° 2851/2000 établissant certaines concessions sous forme de contingents tarifaires communautaires pour certains produits agricoles en provenance respectivement de la République de Bulgarie, de la République tchèque et de la République de Pologne, et modifiant le règlement (CE) n° 1218/96(7), modifié en dernier lieu par le règlement (CE) n 788/2003(8).
(10) Les mesures prévues par le présent règlement sont conformes à l'avis du comité de gestion des céréales,
A ARRÊTÉ LE PRÉSENT RÈGLEMENT:
Article premier
1. Les importations de blé et méteil (code NC 1001 ) visées à l'annexe I en provenance de la République tchèque et bénéficiant d'un droit nul à l'importation, dans le cadre du contingent tarifaire portant le numéro d'ordre 09.4638, en vertu de la décision 2003/298/CE sont soumises à un certificat d'importation délivré conformément aux dispositions du présent règlement.
2. Les importations de maïs (code NC 1005 10 90 et 1005 90 00 ) visées à l'annexe I en provenance de la République tchèque et bénéficiant d'un droit nul à l'importation, dans le cadre du contingent tarifaire portant le numéro d'ordre 09.4639, en vertu de la décision 2003/298/CE sont soumises à un certificat d'importation délivré conformément aux dispositions du présent règlement.
3. Les produits visés aux paragraphes 1 et 2 sont mis en libre pratique sur présentation de l'un des documents suivants:
a) le certificat de circulation des marchandises EUR.1, délivré par les autorités compétentes du pays d'exportation conformément aux dispositions du protocole n° 4 de l'accord européen établissant une association entre la Communauté et ledit pays;
b) une déclaration sur la facture établie par l'exportateur, conformément aux dispositions dudit protocole.
Article 2
1. Les demandes de certificats d'importation sont déposées auprès des autorités compétentes des États membres le deuxième lundi de chaque mois, au plus tard à 13 heures, heure de Bruxelles.
La quantité indiquée dans la demande de certificat ne peut dépasser la quantité fixée pour l'importation du produit faisant l'objet de la campagne de commercialisation concernée.
2. Le jour même du dépôt des demandes de certificats, avant 18 heures, heure de Bruxelles, les autorités compétentes des États membres communiquent à la Commission par télécopieur [(32-2) 295 25 15], conformément au modèle figurant à l'annexe II, la somme totale de toutes les quantités indiquées dans les demandes de certificats d'importation.
Cette information est communiquée séparément des informations concernant les autres demandes de certificats d'importation de céréales.
3. Si le total des quantités octroyées pour chaque produit concerné depuis le début de la campagne visé au paragraphe 2 dépasse le contingent prévu pour la campagne concernée, la Commission fixe, au plus tard le troisième jour ouvrable suivant le dépôt des demandes, un coefficient unique de réduction à appliquer aux quantités demandées.
4. Sans préjudice de l'application du paragraphe 3, les certificats sont délivrés le cinquième jour ouvrable suivant celui du dépôt de la demande. Le jour de la délivrance des certificats, avant 18 heures, heure de Bruxelles, les autorités compétentes transmettent par télécopieur à la Commission la quantité totale obtenue en additionnant les quantités pour lesquelles les certificats d'importation ont été délivrés ce même jour.
Article 3
Conformément à l'article 23, paragraphe 2, du règlement (CE) n° 1291/2000, la durée de validité du certificat est calculée à partir de la date effective de sa délivrance.
Les certificats d'importation sont valables jusqu'à la fin du mois suivant celui de leur délivrance.
Article 4
Les droits découlant du certificat d'importation ne sont pas transmissibles.
Article 5
La quantité mise en libre pratique ne peut être supérieure à celle indiquée dans les cases 17 et 18 du certificat d'importation. Le chiffre "0" est inscrit à cet effet dans la case 19 dudit certificat.
Article 6
1. La demande de certificat d'importation et le certificat d'importation comportent les informations suivantes:
a) dans la case 8, le nom du pays d'origine;
b) dans la case 20, l'une des indications suivantes:
- Reglamento (CE) n° 925/2003
- Forordning (EF) nr. 925/2003
- Verordnung (EG) Nr. 925/2003
- Kανονισμός (EK) αριθ. 925/2003
- Regulation (EC) No 925/2003
- Règlement (CE) n° 925/2003
- Regolamento (CE) n. 925/2003
- Verordening (EG) nr. 925/2003
- Regulamento (CE) n.o 925/2003
- Asetus (EY) N:o 925/2003
- Förordning (EG) nr 925/2003
c) dans la case 24, la mention "droit nul".
2. Pour les produits importés dans le cadre des contingents visés à l'article premier, paragraphe 3, la demande de certificat d'importation et le certificat d'importation doivent être accompagnés d'un certificat indiquant que le produit exporté ne bénéficie d'aucune subvention à l'exportation et ne provient pas d'un stock public d'intervention. Les certificats délivrés par le Fonds national d'intervention agricole de la République tchèque (SIAF) sont reconnus officiellement par la Commission dans le cadre de la coopération administrative visée aux articles 63 à 65 du règlement (CEE) n° 2454/93 de la Commission(9).
Le modèle de ce certificat, ainsi que le cachet et les signatures autorisés par les autorités tchèques, sont reproduits à l'annexe III, points A et B.
Article 7
La garantie relative aux certificats d'importation prévus par le présent règlement est de 30 euros par tonne.
Article 8
Le règlement (CE) n° 2809/2000 est modifié comme suit:
1) le titre du règlement est remplacé par le titre suivant:
"Règlement (CE) n° 2809/2000 de la Commission du 20 décembre 2000 portant modalités d'application, pour les produits du secteur céréalier, des règlements (CE) n° 2290/2000 et (CE) n° 2851/2000, établissant certaines concessions sous forme de contingents tarifaires communautaires pour certains produits agricoles en provenance respectivement de la République de Bulgarie et de la République de Pologne, et abrogeant le règlement (CE) n° 1218/96";
2) l'article 2 est remplacé par le texte suivant:
"Article 2
L'importation des produits énumérés à l'annexe I du présent règlement en provenance de la République de Pologne et bénéficiant de l'exonération totale du droit à l'importation dans les limites des quantités et des taux de réduction ou du montant indiqués à l'annexe I est soumise à la présentation d'un certificat d'importation délivré conformément aux dispositions du présent règlement.";
3) à l'annexe I, les lignes concernant la République tchèque sont supprimées.
Article 9
Le présent règlement entre en vigueur le jour suivant celui de sa publication au Journal officiel de l'Union européenne.
Il est applicable à compter du 1er juin 2003.
Le présent règlement est obligatoire dans tous ses éléments et directement applicable dans tout État membre.
Fait à Bruxelles, le 27 mai 2003.""")

```

</div>

## Results

```bash
commission parlementaire,commission ONU,adhésif
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_multieurlex_french|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|12.9 MB|

## References

https://huggingface.co/datasets/nlpaueb/multi_eurlex

## Benchmarking

```bash
 
labels precision    recall  f1-score   support
0       0.58      0.64      0.61        11
1       0.98      0.83      0.90       281
2       0.88      0.94      0.91      1213
3       0.88      0.57      0.69        49
4       0.92      0.90      0.91        61
5       1.00      0.76      0.86        21
6       0.75      0.68      0.71       177
7       0.78      0.63      0.70       735
8       0.00      0.00      0.00        17
9       0.87      0.65      0.74        20
10      0.93      0.74      0.82        50
11      0.87      1.00      0.93        13
12      0.88      0.88      0.88      1100
13      0.72      0.34      0.46        67
14      0.81      0.57      0.67       240
15      0.80      0.81      0.80       745
16      0.71      0.55      0.62       140
   micro avg       0.85      0.80      0.82      4940
   macro avg       0.78      0.68      0.72      4940
weighted avg       0.84      0.80      0.81      4940
 samples avg       0.84      0.80      0.80      4940

```
