---
layout: model
title: Multi-lingual DeBERTa base model
author: John Snow Labs
name: mdeberta_v3_base
date: 2022-03-10
tags: [xx, multilingual, deberta, mdeberta, v3, base, open_source]
task: Embeddings
language: xx
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DeBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks with 80GB training data. 

In [DeBERTa V3](https://arxiv.org/abs/2111.09543), we further improved the efficiency of DeBERTa using ELECTRA-Style pre-training with Gradient Disentangled Embedding Sharing. Compared to DeBERTa,  our V3 version significantly improves the model performance on downstream tasks.  You can find more technique details about the new model from our [paper](https://arxiv.org/abs/2111.09543).

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more implementation details and updates.

mDeBERTa is multilingual version of DeBERTa which use the same structure as DeBERTa and was trained with CC100 multilingual data.
The mDeBERTa V3 base model comes with 12 layers and a hidden size of 768. It has 86M backbone parameters  with a vocabulary containing 250K tokens which introduces 190M parameters in the Embedding layer.  This model was trained using the 2.5T CC100 data as XLM-R.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_xx_3.4.2_3.0_1646908683611.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

```
```scala
val embeddings = DeBertaEmbeddings.pretrained("mdeberta_v3_base", "xx")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.mdeberta_v3_base").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_v3_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|661.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[https://huggingface.co/microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)

## Benchmarking

```bash
#### Fine-tuning on NLU tasks

The dev results on XNLI with zero-shot cross-lingual transfer setting, i.e. training with English data only, test on other languages.

| Model        |avg | en |  fr| es  | de  | el  | bg  | ru  |tr   |ar   |vi   | th  | zh | hi  | sw  | ur  | 
|--------------| ----|----|----|---- |--   |--   |--   | --  |--   |--   |--   | --  | -- | --  | --  | --  |
| XLM-R-base   |76.2 |85.8|79.7|80.7 |78.7 |77.5 |79.6 |78.1 |74.2 |73.8 |76.5 |74.6 |76.7| 72.4| 66.5| 68.3|
| mDeBERTa-base|**79.8**+/-0.2|**88.2**|**82.6**|**84.4** |**82.7** |**82.3** |**82.4** |**80.8** |**79.5** |**78.5** |**78.1** |**76.4** |**79.5**| **75.9**| **73.9**| **72.4**|


```