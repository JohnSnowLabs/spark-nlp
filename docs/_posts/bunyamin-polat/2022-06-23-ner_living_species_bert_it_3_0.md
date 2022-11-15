---
layout: model
title: Detect Living Species (bert_embeddings_bert_base_italian_xxl_cased)
author: John Snow Labs
name: ner_living_species_bert
date: 2022-06-23
tags: [it, ner, clinical, licensed, bert]
task: Named Entity Recognition
language: it
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Italian which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `bert_embeddings_bert_base_italian_xxl_cased` embeddings.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

**NOTE :**
1.	The text files were translated from Spanish with a neural machine translation system.
2.	The annotations were translated with the same neural machine translation system.
3.	The translated annotations were transferred to the translated text files using an annotation transfer technology.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_it_3.5.3_3.0_1655972219820.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_italian_xxl_cased", "it")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "it", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter   
])

data = spark.createDataFrame([["""Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_italian_xxl_cased", "it")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species_bert", "it", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
sentence_detector,
tokenizer,
embeddings,
ner_model,
ner_converter))

val data = Seq("""Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("it.med_ner.living_species.bert").predict("""Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato.""")
```

</div>

## Results

```bash
+----------------+-------+
|ner_chunk       |label  |
+----------------+-------+
|donna           |HUMAN  |
|personale       |HUMAN  |
|madre           |HUMAN  |
|fratello        |HUMAN  |
|sorella         |HUMAN  |
|virus epatotropi|SPECIES|
|HBV             |SPECIES|
|HCV             |SPECIES|
|HIV             |SPECIES|
|paziente        |HUMAN  |
+----------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_bert|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|it|
|Size:|16.4 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.88       0.95    0.91      2772    
B-SPECIES     0.76       0.89    0.82      2860    
I-HUMAN       0.70       0.59    0.64      101     
I-SPECIES     0.70       0.81    0.75      1036    
micro-avg     0.80       0.90    0.85      6769    
macro-avg     0.76       0.81    0.78      6769    
weighted-avg  0.80       0.90    0.85      6769  
```
