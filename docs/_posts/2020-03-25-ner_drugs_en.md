---
layout: model
title: Detect Drugs 
author: John Snow Labs
name: ner_drugs_en
date: 2020-03-25
tags: [ner, en, licensed, clinical]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---
{:.h2_title}
## Description

Pretrained named entity recognition deep learning model for Drugs. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

{:.h2_title}
## Predicted Entities 
``DrugChem``.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_en_2.4.4_2.4_1584452534235.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...

clinical_ner = NerDLModel.pretrained("ner_drugs", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

...

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(data)

```

```scala
...

val ner = NerDLModel.pretrained("ner_drugs", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings") 
  .setOutputCol("ner")

...

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata.

```bash
+-----------------+---------+
|chunk            |ner_label|
+-----------------+---------+
|Bactrim          |DrugChem |
|Fragmin 5000     |DrugChem |
|Xenaderm         |DrugChem |
|OxyContin        |DrugChem |
|folic acid       |DrugChem |
|levothyroxine    |DrugChem |
|Prevacid         |DrugChem |
|Norvasc          |DrugChem |
|Lexapro          |DrugChem |
|aspirin          |DrugChem |
|Neurontin        |DrugChem |
|Percocet         |DrugChem |
|magnesium citrate|DrugChem |
|Wellbutrin       |DrugChem |
|Bactrim          |DrugChem |
+-----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_en_2.4.4_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.4+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on i2b2_med7 + FDA with 'embeddings_clinical'.
https://www.i2b2.org/NLP/Medication

{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-DrugChem    | 32745 |  1738 |   979 | 0.949598 | 0.97097  | 0.960165 |
|  1 | I-DrugChem    | 35522 |  1551 |   764 | 0.958164 | 0.978945 | 0.968443 |
|  2 | Macro-average | 68267 |  3289 |  1743 | 0.953881 | 0.974958 | 0.964304 |
|  3 | Micro-average | 68267 |  3289 |  1743 | 0.954036 | 0.975104 | 0.964455 |
```