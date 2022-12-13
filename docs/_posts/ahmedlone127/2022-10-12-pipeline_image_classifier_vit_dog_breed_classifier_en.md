---
layout: model
title: English pipeline_image_classifier_vit_dog_breed_classifier ViTForImageClassification from skyau
author: John Snow Labs
name: pipeline_image_classifier_vit_dog_breed_classifier
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_dog_breed_classifier` is a English model originally trained by skyau.


## Predicted Entities

`whippet`, `cairn`, `Old_English_sheepdog`, `Rottweiler`, `American_Staffordshire_terrier`, `Blenheim_spaniel`, `Leonberg`, `bluetick`, `Yorkshire_terrier`, `African_hunting_dog`, `Doberman`, `Appenzeller`, `Boston_bull`, `German_shepherd`, `kuvasz`, `standard_poodle`, `Chesapeake_Bay_retriever`, `toy_terrier`, `Australian_terrier`, `Dandie_Dinmont`, `Brittany_spaniel`, `basenji`, `Newfoundland`, `Airedale`, `giant_schnauzer`, `Bouvier_des_Flandres`, `golden_retriever`, `Welsh_springer_spaniel`, `Pekinese`, `West_Highland_white_terrier`, `briard`, `Gordon_setter`, `Border_collie`, `Pomeranian`, `Scotch_terrier`, `malamute`, `EntleBucher`, `toy_poodle`, `Mexican_hairless`, `clumber`, `Scottish_deerhound`, `curly-coated_retriever`, `Bedlington_terrier`, `soft-coated_wheaten_terrier`, `Irish_setter`, `Lhasa`, `bloodhound`, `French_bulldog`, `standard_schnauzer`, `Chihuahua`, `borzoi`, `Sealyham_terrier`, `malinois`, `Norwegian_elkhound`, `Staffordshire_bullterrier`, `bull_mastiff`, `Ibizan_hound`, `komondor`, `Kerry_blue_terrier`, `Saint_Bernard`, `basset`, `Eskimo_dog`, `Sussex_spaniel`, `English_springer`, `flat-coated_retriever`, `cocker_spaniel`, `Tibetan_terrier`, `Shih-Tzu`, `beagle`, `silky_terrier`, `Saluki`, `vizsla`, `pug`, `Shetland_sheepdog`, `Maltese_dog`, `Norwich_terrier`, `kelpie`, `Italian_greyhound`, `Walker_hound`, `Greater_Swiss_Mountain_dog`, `miniature_schnauzer`, `Great_Pyrenees`, `Tibetan_mastiff`, `collie`, `Siberian_husky`, `Bernese_mountain_dog`, `Irish_wolfhound`, `chow`, `boxer`, `Great_Dane`, `dingo`, `Japanese_spaniel`, `Rhodesian_ridgeback`, `Border_terrier`, `Afghan_hound`, `Irish_water_spaniel`, `black-and-tan_coonhound`, `redbone`, `Norfolk_terrier`, `affenpinscher`, `Brabancon_griffon`, `miniature_pinscher`, `Labrador_retriever`, `Lakeland_terrier`, `groenendael`, `schipperke`, `papillon`, `wire-haired_fox_terrier`, `Cardigan`, `English_foxhound`, `Pembroke`, `dhole`, `German_short-haired_pointer`, `miniature_poodle`, `Irish_terrier`, `Weimaraner`, `otterhound`, `English_setter`, `Samoyed`, `keeshond`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_dog_breed_classifier_en_4.2.1_3.0_1665535557269.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_dog_breed_classifier_en_4.2.1_3.0_1665535557269.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_dog_breed_classifier', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_dog_breed_classifier", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_dog_breed_classifier|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.3 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification