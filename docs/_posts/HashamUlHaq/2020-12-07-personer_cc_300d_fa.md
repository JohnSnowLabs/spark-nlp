---
layout: model
title: Detect Person, Organization, Location, Facility, Product and Event entities in Persian (persian_w2v_cc_300d)
author: John Snow Labs
name: personer_cc_300d
date: 2020-12-07
task: Named Entity Recognition
language: fa
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [ner, fa, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses Persian word embeddings to find 6 different types of entities in Persian text. It is trained using `persian_w2v_cc_300d` word embeddings, so please use the same embeddings in the pipeline.


## Predicted Entities

Persons-`PER`, Facilities-`FAC`, Products-`PRO`, Locations-`LOC`, Organizations-`ORG`, Events-`EVENT`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/personer_cc_300d_fa_2.7.0_2.4_1607339059321.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("persian_w2v_cc_300d", "fa") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")
ner = NerDLModel.pretrained("personer_cc_300d", "fa") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
ner_converter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند")
```

```scala
...
val embeddings = WordEmbeddingsModel.pretrained("persian_w2v_cc_300d", "fa") 
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val ner_model = NerDLModel.pretrained("personer_cc_300d", "fa")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")
val ner_converter = NerConverter().setInputCols(Array("sentence", "token", "ner")).setOutputCol("ner_chunk")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter))
val data = Seq("به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.ner").predict("""به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند""")
```

</div>

## Results

```bash
|    | ner_chunk                 | entity       |
|---:|--------------------------:|-------------:|
|  0 | خبرنگار ایرنا            | ORG          |
|  1 | محمد قمی                  | PER          |
|  2 | پاکدشت                    | LOC          |
|  3 | علی‌اکبر موسوی خوئینی     | PER          |
|  4 | شمس‌الدین وهابی            | PER          |
|  5 | تهران                      | LOC          |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|personer_cc_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|fa|
|Dependencies:|persian_w2v_cc_300d|

## Data Source

This model is trained on data provided by [https://www.aclweb.org/anthology/C16-1319/](https://www.aclweb.org/anthology/C16-1319/).

## Benchmarking

```bash
|    | label         |    tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:--------------|------:|------:|-----:|---------:|---------:|---------:|
|  0 | B-Per         |  1035 |    99 |   75 | 0.912698 | 0.932432 | 0.92246  |
|  1 | I-Fac         |   239 |    42 |   64 | 0.850534 | 0.788779 | 0.818493 |
|  2 | I-Pro         |   173 |    52 |  158 | 0.768889 | 0.522659 | 0.622302 |
|  3 | I-Loc         |   221 |    68 |   66 | 0.764706 | 0.770035 | 0.767361 |
|  4 | I-Per         |   652 |    38 |   55 | 0.944928 | 0.922207 | 0.933429 |
|  5 | B-Org         |  1118 |   289 |  348 | 0.794598 | 0.762619 | 0.778281 |
|  6 | I-Org         |  1543 |   237 |  240 | 0.866854 | 0.865395 | 0.866124 |
|  7 | I-Event       |   486 |   130 |  108 | 0.788961 | 0.818182 | 0.803306 |
|  8 | B-Loc         |   974 |   252 |  168 | 0.794454 | 0.85289  | 0.822635 |
|  9 | B-Fac         |   123 |    31 |   44 | 0.798701 | 0.736527 | 0.766355 |
| 10 | B-Pro         |   168 |    81 |   97 | 0.674699 | 0.633962 | 0.653697 |
| 11 | B-Event       |   126 |    52 |   51 | 0.707865 | 0.711864 | 0.709859 |
| 12 | Macro-average | 6858  | 1371  | 1474 | 0.805657 | 0.776463 | 0.790791 |
| 13 | Micro-average | 6858  | 1371  | 1474 | 0.833394 | 0.823092 | 0.828211 |
```