---
layout: model
title: English re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3 RoBertaForTokenClassification from ajtamayoh
author: John Snow Labs
name: re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3
date: 2023-12-13
tags: [roberta, en, open_source, token_classification, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.2.1
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3` is a English model originally trained by ajtamayoh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3_en_5.2.1_3.0_1702510814814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3_en_5.2.1_3.0_1702510814814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
        
    
tokenClassifier = RoBertaForTokenClassification.pretrained("re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3","en") \
            .setInputCols(["document","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() \
        .setInputCols(Array("document")) \
        .setOutputCol("token")

val tokenClassifier = RoBertaForTokenClassification  
    .pretrained("re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3", "en")
    .setInputCols(Array("document","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_negref_nsd_nubes_training_test_dataset_roberta_base_biomedical_clinical_spanish_fine_tuned_v3|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|447.0 MB|

## References

https://huggingface.co/ajtamayoh/RE_NegREF_NSD_Nubes_Training_Test_dataset_roberta-base-biomedical-clinical-es_fine_tuned_v3