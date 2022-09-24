---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: NLP Server release notes 0.6.0
permalink: /docs/en/nlp_server/nlp_server_versions/release_notes_0_6_0
key: docs-release-notes
modify_date: "2022-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

## NLP Server 0.6.0

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.6.0`    |
| Type         | Minor      |
| Release Date | 2022-04-06 |

<br>

### Overview

We are excited to release NLP Server v0.6.0! This new release comes with exciting new features and improvements that extend and enhance the capabilities of the NLP Server.

This release comes with the ability to share the models with the Annotation Lab. This will enable easy access to custom models uploaded to or trained with the Annotation Lab or to pre-trained models downloaded to Annotation Lab from the NLP Models Hub.
As such the NLP Server becomes an easy and quick tool for testing our trained models locally on your own infrastructure with zero data sharing.

Another important feature we have introduced is the support for Spark OCR spells. Now we can upload images, PDFs, or other documents to the NLP Server and run OCR spells on top of it. The results of the processed documents are also available for export.

The release also includes a few improvements to the existing features and some bug fixes.

<br>

### Key Information

1. For a smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications
2. NLP Server is now available on [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) as well as on [AWS marketplace](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm).

<br>

### Major Features and Improvements

#### Support for custom models trained with the Annotation Lab

Models trained with the Annotation Lab are now available as “custom” spells in the NLP Server. Similarly, models manually uploaded to the Annotation Lab, or downloaded from the NLP Models Hub are also made available for use in the NLP Server. This is only supported in a docker setup at present when both tools are deployed in the same machine.

#### Support for Spark OCR spells

OCR spells are now supported by NLP Server in the presence of a valid OCR license. Users can upload an image, PDF, or other supported document format and run the OCR spells on it. The processed results are also available for download as a text document. It is also possible to upload multiple files at once for OCR operation. These files can be images, PDFs, word documents, or a zipped file.

<br>

### Other Improvements

1. Now users can chain multiple spells together to analyze the input data. The order of operation on the input data will be in the sequence of the spell chain from left to right.
2. NLP Server now supports more than 5000+ models in 250+ languages powered by NLU.

<br>

### Bug Fixes

1. Not found error seen when running predictions using certain spells.
2. The prediction job runs in an infinite loop when using certain spells.
3. For input data having new line characters JSON exception was seen when processing the output from NLU.
4. Incorrect license information was seen in the license popup.
5. Spell field cleared abruptly when typing the spells.


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_0_6_1">Version 0.6.1</a>
    </li>
    <li>
        <strong>Version 0.6.0</strong>
    </li>
    <li>
        <a href="release_notes_0_5_0">Version 0.5.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
  <li><a href="release_notes_0_7_1">0.7.1</a></li>
  <li><a href="release_notes_0_7_0">0.7.0</a></li>
  <li><a href="release_notes_0_6_1">0.6.1</a></li>
  <li class="active"><a href="release_notes_0_6_0">0.6.0</a></li>
  <li><a href="release_notes_0_5_0">0.5.0</a></li>
  <li><a href="release_notes_0_4_0">0.4.0</a></li>
</ul>