---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: Release Notes
permalink: /docs/en/nlp_server/release_notes
key: docs-nlp-server
modify_date: "2022-04-05"
show_nav: true
sidebar:
    nav: nlp-server
---

## NLP Server 0.6.0

### Overview

We are excited to release NLP Server v0.6.0! This new release comes with exciting new features and improvements that extend and enhance the capabilities of the NLP Server. 

This release comes with the ability to share the models with our very own Annotation Lab. This will enable us to use all the custom trained, pre-trained, or models downloaded to the Annotation Lab directly as spells on the NLP Server making it easy to quickly test and play around with our trained models. 

Another important feature we have introduced is the support for Spark OCR spells. Now we can upload images, PDFs, or other documents to the NLP Server and run OCR spells on top of it. The results of the processed documents are also available for export. 

The release also includes a few improvements to the existing features and some bug fixes.

<br>

### Release Details

Fields | Details
--- | ---
Name | NLP Server
Version | 0.6.0
Type | Minor
Dependency | 
Release Date | 2022-04-05

<br>

### Installation Prerequisites

1. For a smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications

<br>

### Key Information

1. Is now available in the Azure marketplace from v0.6.0
2. Uses NLU v3.4.3rc1 as an underlying dependency

<br>

### Major Features and Improvements

#### Share models with Annotation Lab

Custom-trained models from the Annotation Lab are now available as “custom” spells in the NLP Server. Similarly, models uploaded to the Annotation Lab, pre-trained models, and the models downloaded from the Models Hub in Annotation Lab are also made available for use in the NLP Server. This is only supported in a docker setup at present when both the tools are deployed in the same machine.

#### Support for Spark OCR spells

Now we can use OCR spells directly from the NLP Server if we have an OCR license. We can upload an image, PDF, or other supported document format and run the OCR spells on it. The processed results are also available for download as a text document. We can also upload multiple files at once for OCR operation. These files can be images, PDFs, word documents, or a zipped file.

<br>

### Other Improvements

1. Now we can chain multiple spells together to predict the input data. The order of operation on the input data will be in the sequence of the spell chain from left to right.
2. Now supports more than 5000+ models in 250+ languages powered by NLU.

<br>

### Bug Fixes

1. Not found error is seen when running predictions using certain spells.
2. The prediction job runs in an infinite loop when using certain spells.
3. For input data having new line characters JSON exception was seen when processing the output from NLU.

<br>

### Known Issues

1. License generated using v3.5.0 secret will not work in the v0.6.0 of the NLP Server.

## NLP Server 0.5.0

### Highlights

- Support for easy license import from my.johnsnowlabs.com.
- Visualize annotation results with Spark NLP Display.
- Examples of results obtained using popular spells on sample texts have been added to the UI.
- Performance improvement when previewing the annotations.
- Support for 22 new models for 23 languages including various African and Indian languages as well as Medical Spanish models powered by NLU 3.4.1
- Various bug fixes


## NLP Server 0.4.0

### Highlights

- This version of NLP Server offers support for licensed models and annotators. Users can now upload a Spark NLP for Healthcare license file and get access to a wide range of additional [annotators and transformers](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators). A valid license key also gives access to more than [400 state-of-the-art healthcare models](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare). Those can be used via easy to learn NLU spells or via API calls.
- NLP Server now supports better handling of large amounts of data to quickly analyze via UI by offering support for uploading CSV files.
- Support for floating licenses. Users can now take advantage of the floating license flexibility and use those inside of the NLP Server.
