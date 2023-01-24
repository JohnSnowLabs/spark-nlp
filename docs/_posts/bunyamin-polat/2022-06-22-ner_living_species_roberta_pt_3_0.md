---
layout: model
title: Detect Living Species(roberta_embeddings_BR_BERTo)
author: John Snow Labs
name: ner_living_species_roberta
date: 2022-06-22
tags: [pt, ner, clinical, licensed, roberta]
task: Named Entity Recognition
language: pt
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts in Portuguese which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained using `roberta_embeddings_BR_BERTo` embeddings.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_roberta_pt_3.5.3_3.0_1655923058986.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_roberta_pt_3.5.3_3.0_1655923058986.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_BR_BERTo","pt")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species_roberta", "pt", "clinical/models")\
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

data = spark.createDataFrame([["""Mulher de 23 anos, de Capinota, Cochabamba, Bolívia. Ela está no nosso país há quatro anos. Frequentou o departamento de emergência obstétrica onde foi encontrada grávida de 37 semanas, com um colo dilatado de 5 cm e membranas rompidas. O obstetra de emergência realizou um teste de estreptococos negativo e solicitou um hemograma, glucose, bioquímica básica, HBV, HCV e serologia da sífilis."""]]).toDF("text")

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

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_BR_BERTo","pt")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species_roberta", "pt", "clinical/models")
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

val data = Seq("""Mulher de 23 anos, de Capinota, Cochabamba, Bolívia. Ela está no nosso país há quatro anos. Frequentou o departamento de emergência obstétrica onde foi encontrada grávida de 37 semanas, com um colo dilatado de 5 cm e membranas rompidas. O obstetra de emergência realizou um teste de estreptococos negativo e solicitou um hemograma, glucose, bioquímica básica, HBV, HCV e serologia da sífilis.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.med_ner.living_species.roberta").predict("""Mulher de 23 anos, de Capinota, Cochabamba, Bolívia. Ela está no nosso país há quatro anos. Frequentou o departamento de emergência obstétrica onde foi encontrada grávida de 37 semanas, com um colo dilatado de 5 cm e membranas rompidas. O obstetra de emergência realizou um teste de estreptococos negativo e solicitou um hemograma, glucose, bioquímica básica, HBV, HCV e serologia da sífilis.""")
```

</div>

## Results

```bash
+-------------+-------+
|ner_chunk    |label  |
+-------------+-------+
|Mulher       |HUMAN  |
|grávida      |HUMAN  |
|estreptococos|SPECIES|
|HBV          |SPECIES|
|HCV          |SPECIES|
|sífilis      |SPECIES|
+-------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_roberta|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|16.4 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B-HUMAN       0.86       0.91    0.88      2827    
B-SPECIES     0.52       0.86    0.65      2796    
I-HUMAN       0.79       0.43    0.55      180     
I-SPECIES     0.62       0.81    0.70      1099    
micro-avg     0.65       0.86    0.74      6902    
macro-avg     0.69       0.75    0.70      6902    
weighted-avg  0.68       0.86    0.75      6902  
```
