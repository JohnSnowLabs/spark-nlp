---
layout: model
title: Legal Multilabel Classification (MultiEURLEX, Greek)
author: John Snow Labs
name: legmulticlf_multieurlex_greek
date: 2023-03-24
tags: [legal, classification, el, licensed, multieurlex, open_source, tensorflow]
task: Text Classification
language: el
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

This is a Multilabel Text Classification model that can help you classify 16 types of Greek legal documents.

## Predicted Entities

`κρατικό εμπόριο`, `εξεταστική επιτροπή`, `συγκολλητικό`, `Ώρχους (κομητεία)`, `εσωτερικό εμπόριο`, `Δύσης`, `εξωτερικό εμπόριο`, `κοινοβουλευτική επιτροπή`, `χονδρικό εμπόριο`, `εμπόριο όπλων`, `διεθνές εμπόριο`, `επιτροπή του ΟΗΕ`, `επιτροπή του Ευρωπαϊκού Κοινοβουλίου`, `ανάθεση σύμβασης με δημοπρασία`, `απιστία`, `λιανικό εμπόριο`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_greek_el_1.0.0_3.0_1679670943184.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_greek_el_1.0.0_3.0_1679670943184.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

docClassifier = nlp.MultiClassifierDLModel().pretrained('legmulticlf_multieurlex_greek', 'el', 'legal/models')\
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

result = light_model.annotate("""ΚΑΝΟΝΙΣΜΌΣ (ΕΚ) αριθ. 1038/2004 ΤΗΣ ΕΠΙΤΡΟΠΉΣ
της 27ης Μαΐου 2004
για καθορισμό του μέγιστου ποσού της επιστροφής κατά την εξαγωγή της λευκής ζάχαρης προς ορισμένες τρίτες χώρες για την 28η τμηματική δημοπρασία που πραγματοποιείται στο πλαίσιο της διαρκούς δημοπρασίας του κανονισμού (ΕΚ) αριθ. 1290/2003
Η ΕΠΙΤΡΟΠΗ ΤΩΝ ΕΥΡΩΠΑΪΚΩΝ ΚΟΙΝΟΤΗΤΩΝ,
Έχοντας υπόψη:
τη συνθήκη για την ίδρυση της Ευρωπαϊκής Κοινότητας,
τον κανονισμό (ΕΟΚ) αριθ. 1260/2001 του Συμβουλίου, της 19ης Ιουνίου 2001 περί κοινής οργανώσεως αγοράς στον τομέα της ζάχαρης (1), και ιδίως το άρθρο 27 παράγραφος 5, δεύτερο εδάφιο,
Εκτιμώντας τα ακόλουθα:
(1)
Δυνάμει του κανονισμού (ΕΚ) αριθ. 1290/2003 της Επιτροπής, της 18ης Ιουλίου 2003, σχετικά με μόνιμη δημοπρασία στο πλαίσιο της περιόδου εμπορίας 2003/04 για τον καθορισμό των εισφορών ή/και των επιστροφών κατά την εξαγωγή λευκής ζάχαρης (2), πραγματοποιούνται τμηματικές δημοπρασίες για την εξαγωγή της ζάχαρης αυτής προς ορισμένες τρίτες χώρες.
(2)
Σύμφωνα με το άρθρο 9 παράγραφος 1 του κανονισμού (ΕΚ) αριθ. 1290/2003, καθορίζεται ένα μέγιστο ποσό επιστροφής κατά την εξαγωγή, κατά περίπτωση, για την εν λόγω τμηματική δημοπρασία, αφού ληφθούν υπόψη, ιδίως, η κατάσταση και η προβλεπόμενη εξέλιξη της αγοράς της ζάχαρης στην Κοινότητα και στη διεθνή αγορά.
(3)
Τα μέτρα που προβλέπονται στον παρόντα κανονισμό είναι σύμφωνα με τη γνώμη της επιτροπής διαχείρισης ζάχαρης,
ΕΞΕΔΩΣΕ ΤΟΝ ΠΑΡΟΝΤΑ ΚΑΝΟΝΙΣΜΟ:
Άρθρο 1
Για την 28η τμηματική δημοπρασία λευκής ζάχαρης, που πραγματοποιείται σύμφωνα με τον κανονισμό (ΕΚ) αριθ. 1290/2003, το ανώτατο ποσό της επιστροφής κατά την εξαγωγή καθορίζεται σε 49,950 EUR/100 kg.
Άρθρο 2
Ο παρών κανονισμός αρχίζει να ισχύει στις 28 Μαΐου 2004.
Ο παρών κανονισμός είναι δεσμευτικός ως προς όλα τα μέρη του και ισχύει άμεσα σε κάθε κράτος μέλος.
Βρυξέλλες, 27 Μαΐου 2004.""")

```

</div>

## Results

```bash
συγκολλητικό,επιτροπή του ΟΗΕ
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_multieurlex_greek|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|el|
|Size:|13.0 MB|

## References

https://huggingface.co/datasets/nlpaueb/multi_eurlex

## Benchmarking

```bash
 
labels               precision    recall  f1-score   support
0       0.60      0.42      0.49       132
1       0.94      0.83      0.88        75
2       0.97      0.87      0.92       343
3       0.84      0.79      0.81        89
4       0.93      0.86      0.89        98
5       0.00      0.00      0.00        20
6       0.90      0.61      0.72       272
7       0.88      0.47      0.61        32
8       0.71      0.35      0.47        34
9       0.89      0.88      0.89      1277
10      1.00      0.72      0.84        25
11      0.79      0.86      0.83       900
12      0.73      0.47      0.57        17
13      0.88      0.61      0.72       271
14      0.90      0.96      0.92      1443
15      0.77      0.79      0.78       943
   micro-avg       0.85      0.83      0.84      5971
   macro-avg       0.80      0.65      0.71      5971
weighted-avg       0.85      0.83      0.84      5971
 samples-avg       0.85      0.83      0.82      5971
F1-micro-averaging: 0.8439355385920272
ROC:  0.9010070958863019

```