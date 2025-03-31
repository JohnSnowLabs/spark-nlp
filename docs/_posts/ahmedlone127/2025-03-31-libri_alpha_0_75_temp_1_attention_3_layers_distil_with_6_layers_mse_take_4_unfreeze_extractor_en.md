---
layout: model
title: English libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor Wav2Vec2ForCTC from rohitp1
author: John Snow Labs
name: libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor
date: 2025-03-31
tags: [en, open_source, onnx, asr, wav2vec2]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor` is a English model originally trained by rohitp1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor_en_5.5.1_3.0_1743449790827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor_en_5.5.1_3.0_1743449790827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = Wav2Vec2ForCTC.pretrained("libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor","en") \
     .setInputCols(["audio_assembler"]) \
     .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val audioAssembler = new DocumentAssembler()
    .setInputCols("audio_content")
    .setOutputCols("audio_assembler")

val speechToText = Wav2Vec2ForCTC.pretrained("libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor", "en")
    .setInputCols(Array("audio_assembler")) 
    .setOutputCol("text") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, speechToText))
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_mse_take_4_unfreeze_extractor|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|115.0 MB|

## References

https://huggingface.co/rohitp1/libri-alpha-0.75-Temp-1-attention-3-layers-distil-with-6-layers-mse-take-4-unfreeze-extractor