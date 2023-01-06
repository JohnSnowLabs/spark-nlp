---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 3.5.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_3_5_0
key: docs-licensed-release-notes
modify_date: 2022-08-25
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.5.0

Release date: **25-08-2022**

Annotation Lab 3.5.0 add support for out-of-the-box usage of Multilingual Models as well as support for some of the European Language Models: Romanian, Portuguese, Danish and Italian. It also provides support for split dataset using Test/Train tags in classification project and allows NER pretrained models evaluation with floating license. The release also includes fixes for known security vulnerabilities and for some bug reported by our user community.

Here are the highlights of this release:

### Highlights
- Support for Multilingual Models. Previously, only multilingual embeddings were available in Models Hub page. A new language filter has been added to the Models hub page to make searching for all available multilingual models and embeddings more efficient. User can select the target language and then explore the set of relevant multilingual models and embeddings. 
- Expended Support for European Language Models. Annotation Lab now offers support for four new European languages Romanian, Portuguese, Italian, and Danish, on top of English, Spanish, and German, already supported in previous versions. Many pretrained models in those languages are now available to download from the NLP Models Hub and easily use to preannotate documents on the Annotation Lab.
- Use Test/Train Tags for Classification Training Experiments. The Test/Train split of annotated tasks can be used when training classification models. When this option is checked on the Training Settings, all tasks that have the Test tag are used as test datasets. All tasks tagged as Train together with all other non Test tasks will be used as a training dataset.  
- NER Model Evaluation available for Floating License. Project Owner and/or Manager can evaluate pretrained NER models against a set of annotated tasks in the presence of floating licenses. Earlier, this feature was only available in the presence of airgap licenses. 
- Chunks preannotation in VisualNER. Annotation Lab 3.4.0 which first published the visual NER preannotation and visual NER model training could only create token level preannotations. With version 3.5.0, individual tokens are combined into one chunk entity and shown as merged to the user.
- Benchmarking Information for Models Trained with Annotation Lab. With version 3.5.0 benchmarking information is available for models trained within Annotation Lab. User can go to the Available Models Tab of the Models Hub page and view the benchmarking data by clicking the small graph icon next to the model.
- Configuration for Annotation Lab Deployment. The resources allocated to Annotation Lab deployment can be configured via the resource values in the annotationlab-updater.sh. The instruction to change the parameters are available in the instruction.md file.


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}