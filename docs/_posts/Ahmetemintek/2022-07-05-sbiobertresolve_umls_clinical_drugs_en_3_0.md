---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes (Clinical Drug)
author: John Snow Labs
name: sbiobertresolve_umls_clinical_drugs
date: 2022-07-05
tags: [entity_resolution, licensed, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities to UMLS CUI codes. It is trained on 2022AA UMLS dataset. The complete dataset has 127 different categories, and this model is trained on the Clinical Drug category using sbiobert_base_cased_mli embeddings.

## Predicted Entities

`Predicts UMLS codes for Clinical Drug medical concepts`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_clinical_drugs_en_4.0.0_3.0_1657039242193.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("clinical_ner")

ner_model_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "clinical_ner"])\
    .setOutputCol("ner_chunk")


chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
     .pretrained("sbiobertresolve_umls_clinical_drugs","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_model_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg."""]]).toDF("text")

results = pipeline.fit(data).transform(data)
```
```scala
...
val document_assembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentence_detector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel
      .pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")

val ner_model = MedicalNerModel
      .pretrained("ner_clinical", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("clinical_ner")

val ner_model_converter = new NerConverterInternal()
      .setInputCols(Array("sentence", "token", "clinical_ner"))
      .setOutputCol("ner_chunk")

val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")
      .setInputCols(Array("ner_chunk_doc"))
      .setOutputCol("sbert_embeddings")
      .setCaseSensitive(False)
    
val resolver = SentenceEntityResolverModel
      .pretrained("sbiobertresolve_umls_clinical_drugs", "en", "clinical/models") 
      .setInputCols(Array("ner_chunk_doc", "sbert_embeddings")) 
      .setOutputCol("resolution")
      .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_model_converter, chunk2doc, sbert_embedder, resolver))
    
val data = Seq("She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg.").toDF("text") 

val res = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls_clinical_drugs").predict("""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg.""")
```
</div>

## Results

```bash
|    | chunk                         | code     | code_description           | all_k_code_desc                                              | all_k_codes                                                                                                                                                                             |
|---:|:------------------------------|:---------|:---------------------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 mg/ml | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 mg/ml', 'hydrogen peroxide solution 30%', 'hydrogen peroxide 30 mg/ml [proxacol]', 'hydrogen peroxide 30 mg/ml cutaneous solution', 'benzoyl peroxide 30 mg/ml'] |
|  1 | Neosporin Cream               | C0132149 | neosporin cream            | ['C0132149', 'C0358174', 'C0357999', 'C0307085', 'C0698810'] | ['neosporin cream', 'nystan cream', 'nystadermal cream', 'nupercainal cream', 'nystaform cream']                                                                                        |
|  2 | magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 mg | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 mg', 'magnesium hydroxide 100 mg/ml', 'magnesium sulphate 100mg/ml injection', 'magnesium sulfate 100 mg', 'magnesium sulfate 100 mg/ml']                     |
|  3 | metformin 1000 mg             | C0987664 | metformin 1000 mg          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 mg', 'metformin hydrochloride 1000 mg', 'metformin hcl 1000mg tab', 'metformin hydrochloride 1000 mg [fortamet]', 'metformin hcl 1000mg sa tab']                       |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_clinical_drugs|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[name]|
|Language:|en|
|Size:|2.5 GB|
|Case sensitive:|false|

## References

Trained on 2022AA UMLS dataset’s Clinical Drug category. https://www.nlm.nih.gov/research/umls/index.html