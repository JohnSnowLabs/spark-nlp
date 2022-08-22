---
layout: docs
header: true
seotitle: NLP Server | John Snow Labs
title: Release Notes
permalink: /docs/en/nlp_server/release_notes
key: docs-nlp-server
modify_date: "2022-06-17"
show_nav: true
sidebar:
  nav: nlp-server
---

## NLP Server 0.7.1

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.7.1`    |
| Type         | Patch      |
| Release Date | 2022-06-17 |

<br>

### Overview

We are excited to release NLP Server v0.7.1! We are committed to continuously improve the experience for our users and make our product reliable and easy to use.

This release focuses on solving a few bugs and improving the stability of the NLP Server.

<br>

### Key Information

1. For smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications.
2. NLP Server is available on both [AWS](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm) and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) marketplaces.

<br>

### Bug Fixes

1. Issue when running NER ONTO spell.
2. Issue when running `dep` spell. Since the spell was broken it is temporarily blacklisted.
3. Document normalizer included the HTML, XML tags to the output even after normalization.
4. Issue when running language translation spells `<from_lang>.translate_to.<to_lang>`.
5. Upon cancelation of custom model uploading job exception was seen in the logs.
6. Some few UI related issues and abnormalities during operation.

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

## NLP Server 0.6.1

| Fields       | Details    |
| ------------ | ---------- |
| Name         | NLP Server |
| Version      | `0.6.1`    |
| Type         | Patch      |
| Release Date | 2022-05-06 |

<br>

### Overview

We are excited to release NLP Server v0.6.1! We are continually committed towards improving the experience for our users and making our product reliable and easy to use.

This release focuses on improving the stability of the NLP Server and cleaning up some annoying bugs. To enhance the user experience, the product now provides interactive and informative responses to the users.

The improvements and bug fixes are mentioned in their respective sections below.

<br>

### Key Information

1. For smooth and optimal performance, it is recommended to use an instance with 8 core CPU, and 32GB RAM specifications.
2. NLP Server is available on both [AWS](https://aws.amazon.com/marketplace/pp/prodview-4ohxjejvg7vwm) and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.nlp_server) marketplace.

<br>

### Improvements

1. Support for new models for Lemmatizers, Parts of Speech Taggers, and Word2Vec Embeddings for over 66 languages, with 20 languages being covered for the first time by NLP Server, including ancient and exotic languages like Ancient Greek, Old Russian, Old French and much more.

<br>

### Bug Fixes

1. The prediction job runs in an infinite loop when using certain spells. Now after 3 retries it aborts the process and informs users appropriately.
2. Issue when running lang spell for language classification.
3. The prediction job runs in an infinite loop when incorrect data format is selected for a given input data.
4. The API request for processing spell didn’t work when format parameter was not provided. Now it uses a default value in such case.
5. Users were unable to login to their MYJSL account from NLP Server.
6. Proper response when there is issue in internet connectivity when running spell.

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
