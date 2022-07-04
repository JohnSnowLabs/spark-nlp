---
layout: model
title: Sentence Detection in Multi-lingual Texts
author: John Snow Labs
name: sentence_detector_dl
date: 2021-01-02
task: Sentence Detection
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, xx, sentence_detection]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

In this model, we treated the sentence boundary detection task as a classification problem based on a paper {Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed) using CNN architecture. We also modified the original implemenation a little bit to cover broken sentences and some impossible end of line chars.

We are releasing two pretrained SDDL models: english and multilanguage that are trained on SETimes corpus (Tyers and Alperen, 2010) and Europarl. Wong et al. (2014) datasets.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")
sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""Όπως ίσως θα γνωρίζει, όταν εγκαθιστάς μια νέα εφαρμογή, θα έχεις διαπιστώσει 
λίγο μετά, ότι το PC αρχίζει να επιβραδύνεται. Στη συνέχεια, όταν επισκέπτεσαι την οθόνη ή από την διαχείριση εργασιών, θα διαπιστώσεις ότι η εν λόγω εφαρμογή έχει προστεθεί στη 
λίστα των προγραμμάτων που εκκινούν αυτόματα, όταν ξεκινάς το PC.
Προφανώς, κάτι τέτοιο δεν αποτελεί μια ιδανική κατάσταση, ιδίως για τους λιγότερο γνώστες, οι 
οποίοι ίσως δεν θα συνειδητοποιήσουν ότι κάτι τέτοιο συνέβη. Όσο περισσότερες εφαρμογές στη λίστα αυτή, τόσο πιο αργή γίνεται η 
εκκίνηση, ιδίως αν πρόκειται για απαιτητικές εφαρμογές. Τα ευχάριστα νέα είναι ότι η τελευταία και πιο πρόσφατη preview build της έκδοσης των Windows 10 που θα καταφθάσει στο πρώτο μισό του 2021, οι εφαρμογές θα 
ενημερώνουν το χρήστη ότι έχουν προστεθεί στη λίστα των εφαρμογών που εκκινούν μόλις ανοίγεις το PC.""")
```
```scala
val documenter = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")
val pipeline = new Pipeline().setStages(Array(documenter, model))
val result = pipeline.fit(Seq.empty["Όπως ίσως θα γνωρίζει, όταν εγκαθιστάς μια νέα εφαρμογή, θα έχεις διαπιστώσει 
λίγο μετά, ότι το PC αρχίζει να επιβραδύνεται. Στη συνέχεια, όταν επισκέπτεσαι την οθόνη ή από την διαχείριση εργασιών, θα διαπιστώσεις ότι η εν λόγω εφαρμογή έχει προστεθεί στη 
λίστα των προγραμμάτων που εκκινούν αυτόματα, όταν ξεκινάς το PC.
Προφανώς, κάτι τέτοιο δεν αποτελεί μια ιδανική κατάσταση, ιδίως για τους λιγότερο γνώστες, οι 
οποίοι ίσως δεν θα συνειδητοποιήσουν ότι κάτι τέτοιο συνέβη. Όσο περισσότερες εφαρμογές στη λίστα αυτή, τόσο πιο αργή γίνεται η 
εκκίνηση, ιδίως αν πρόκειται για απαιτητικές εφαρμογές. Τα ευχάριστα νέα είναι ότι η τελευταία και πιο πρόσφατη preview build της έκδοσης των Windows 10 που θα καταφθάσει στο πρώτο μισό του 2021, οι εφαρμογές θα 
ενημερώνουν το χρήστη ότι έχουν προστεθεί στη λίστα των εφαρμογών που εκκινούν μόλις ανοίγεις το PC."].toDS.toDF("text")).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.sentence_detector").predict("""Όπως ίσως θα γνωρίζει, όταν εγκαθιστάς μια νέα εφαρμογή, θα έχεις διαπιστώσει 
λίγο μετά, ότι το PC αρχίζει να επιβραδύνεται. Στη συνέχεια, όταν επισκέπτεσαι την οθόνη ή από την διαχείριση εργασιών, θα διαπιστώσεις ότι η εν λόγω εφαρμογή έχει προστεθεί στη 
λίστα των προγραμμάτων που εκκινούν αυτόματα, όταν ξεκινάς το PC.
Προφανώς, κάτι τέτοιο δεν αποτελεί μια ιδανική κατάσταση, ιδίως για τους λιγότερο γνώστες, οι 
οποίοι ίσως δεν θα συνειδητοποιήσουν ότι κάτι τέτοιο συνέβη. Όσο περισσότερες εφαρμογές στη λίστα αυτή, τόσο πιο αργή γίνεται η 
εκκίνηση, ιδίως αν πρόκειται για απαιτητικές εφαρμογές. Τα ευχάριστα νέα είναι ότι η τελευταία και πιο πρόσφατη preview build της έκδοσης των Windows 10 που θα καταφθάσει στο πρώτο μισό του 2021, οι εφαρμογές θα 
ενημερώνουν το χρήστη ότι έχουν προστεθεί στη λίστα των εφαρμογών που εκκινούν μόλις ανοίγεις το PC.""")
```

</div>

## Results

```bash
+---+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 0 | Όπως ίσως θα γνωρίζει, όταν εγκαθιστάς μια νέα εφαρμογή, θα έχεις διαπιστώσει λίγο μετά, ότι το PC αρχίζει να επιβραδύνεται.                                                                                                                                     |
| 1 | Στη συνέχεια, όταν επισκέπτεσαι την οθόνη ή από την διαχείριση εργασιών, θα διαπιστώσεις ότι η εν λόγω εφαρμογή έχει προστεθεί στη λίστα των προγραμμάτων που εκκινούν αυτόματα, όταν ξεκινάς το PC.                                                             |
| 2 | Προφανώς, κάτι τέτοιο δεν αποτελεί μια ιδανική κατάσταση, ιδίως για τους λιγότερο γνώστες, οι οποίοι ίσως δεν θα συνειδητοποιήσουν ότι κάτι τέτοιο συνέβη.                                                                                                       |
| 3 | Όσο περισσότερες εφαρμογές στη λίστα αυτή, τόσο πιο αργή γίνεται η εκκίνηση, ιδίως αν πρόκειται για απαιτητικές εφαρμογές.                                                                                                                                       |
| 4 | Τα ευχάριστα νέα είναι ότι η τελευταία και πιο πρόσφατη preview build της έκδοσης των Windows 10 που θα καταφθάσει στο πρώτο μισό του 2021, οι εφαρμογές θα ενημερώνουν το χρήστη ότι έχουν προστεθεί στη λίστα των εφαρμογών που εκκινούν μόλις ανοίγεις το PC. |
+---+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_detector_dl|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[sentences]|
|Language:|xx|

## Data Source

Please visit the repo for more information https://github.com/dbmdz/deep-eos