---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: NLP Server release notes 0.7.0
permalink: /docs/en/nlp_server/nlp_server_versions/release_notes_0_7_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

## NLP Server 0.7.0

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.7.0`    |
| Type         | Minor      |
| Release Date | 2022-06-07 |

<br>

### Overview

We are excited to release NLP Server v0.7.0! This new release comes with an exciting new feature of table extraction from various file formats.

Table extraction feature enables extracting tabular content from the document. This extracted content is available as JSON and hence can again be processed with different spells for further predictions. The various supported files formats are documents (pdf, doc, docx), slides (ppt, pptx), and zipped content containing the mentioned formats.

The improvements are mentioned in their respective sections below.

<br>

### Key Information

1. For smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications.
2. NLP Server is available on both [AWS](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm) and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) marketplace.

<br>

### Major Features and Improvements

#### Support for Table extraction

NLP Server now supports extracting tabular content from various file types. The currently supported file types are documents (pdf, doc, docx), slides (ppt, pptx), and zipped content containing any of the mentioned formats. These extracted contents are available as JSON output from both UI and API that can easily be converted to suitable Data Frames (e.g., pandas DF) for further processing. The output of the table extraction process can also be viewed in the NLP Server UI as a flat table. Currently, if multiple tables are extracted from the document, then only one of the tables selected randomly will be shown as a preview in the UI. However, upon downloading all the extracted tables are exported in separate JSON dumps combined in a single zipped file. For this version, the table extraction on PDF files is successful only if the PDF contains necessary metadata about the table content.

<br>

### Other Improvements

1. Support for over 600 new models, and over 75 new languages including ancient, dead, and extinct languages.
2. Transformer-based embeddings and token classifiers are powered by state-of-the-art [CamemBertEmbeddings](https://camembert-model.fr/) and [DeBertaForTokenClassification](https://arxiv.org/abs/2006.03654) based architectures.
3. Added Portuguese De-identification models, NER models for Gene detection, and RxNorm Sentence resolution model for mapping and extracting pharmaceutical actions as well as treatments.
4. JSON payload is now supported in the request body when using create result API.


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_0_6_1">Version 0.6.1</a>
    </li>
    <li>
        <strong>Version 0.7.0</strong>
    </li>
    <li>
        <a href="release_notes_0_7_1">Version 0.7.1</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_0_7_1">0.7.1</a></li>
  <li class="active"><a href="release_notes_0_7_0">0.7.0</a></li>
  <li><a href="release_notes_0_6_1">0.6.1</a></li>
  <li><a href="release_notes_0_6_0">0.6.0</a></li>
  <li><a href="release_notes_0_5_0">0.5.0</a></li>
  <li><a href="release_notes_0_4_0">0.4.0</a></li>
</ul>