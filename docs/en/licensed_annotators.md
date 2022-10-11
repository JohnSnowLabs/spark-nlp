---
layout: docs
header: true
seotitle: Spark NLP | John Snow Labs
title: Enterprise NLP Annotators
permalink: /docs/en/licensed_annotators
key: docs-licensed-annotators
modify_date: "2020-08-10"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

A Spark NLP Enterprise license includes access to unique annotators.
At the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/e06141715bc14e5ed43388585ce002b8b37e3f18/tutorials/Certification_Trainings) you can see different types of annotators in action.

By clicking on any annotator, you will see different sections:
- The `Approach`, or class to train models.
- The `Model`, to infer using pretrained models.

Also, for most of the annotators, you will find examples for the different enterprise libraries:
- Healthcare NLP
- Finance NLP
- Legal NLP
- 
Check out the [Spark NLP Annotators page](https://nlp.johnsnowlabs.com/docs/en/annotators) for more information on how to read this page.

</div>

## Available Annotators

{:.table-model-big}
|Annotators|Description|
|---|---|
{% include templates/licensed_table_entry.md  name="AssertionDL" summary="AssertionDL is a deep Learning based approach used to extract Assertion Status from extracted entities and text."%}
{% include templates/licensed_table_entry.md  name="AssertionFilterer" summary="Filters entities coming from ASSERTION type annotations and returns the CHUNKS."%}
{% include templates/licensed_table_entry.md  name="AssertionLogReg" summary="Logistic Regression is used to extract Assertion Status from extracted entities and text."%}
{% include templates/licensed_table_entry.md  name="Chunk2Token" summary="A feature transformer that converts the input array of strings (annotatorType CHUNK) into an array of chunk-based tokens (annotatorType TOKEN)."%}
{% include templates/licensed_table_entry.md  name="ChunkEntityResolver" summary="Returns a normalized entity for a particular trained ontology / curated dataset (e.g. clinical ICD-10, RxNorm, SNOMED; financial SEC's EDGAR database,  etc)."%}
{% include templates/licensed_table_entry.md  name="ChunkFilterer" summary="Filters entities coming from CHUNK annotations."%}
{% include templates/licensed_table_entry.md  name="ChunkKeyPhraseExtraction" summary="Uses Bert Sentence Embeddings to determine the most relevant key phrases describing a text."%}
{% include templates/licensed_table_entry.md  name="ChunkMerge" summary="Merges entities coming from different CHUNK annotations."%}
{% include templates/licensed_table_entry.md  name="ContextualParser" summary="Extracts entity from a document based on user defined rules."%}
{% include templates/licensed_table_entry.md  name="DeIdentification" summary="Deidentifies Input Annotations of types DOCUMENT, TOKEN and CHUNK, by either masking or obfuscating the given CHUNKS."%}
{% include templates/licensed_table_entry.md  name="DocumentLogRegClassifier" summary="Classifies documents with a Logarithmic Regression algorithm."%}
{% include templates/licensed_table_entry.md  name="DrugNormalizer" summary="Annotator which normalizes raw text from documents, e.g. scraped web pages or xml documents"%}
{% include templates/licensed_table_entry.md  name="FeaturesAssembler" summary="Collects features from different columns."%}
{% include templates/licensed_table_entry.md  name="GenericClassifier" summary="Creates a generic single-label classifier which uses pre-generated Tensorflow graphs."%}
{% include templates/licensed_table_entry.md  name="IOBTagger" summary="Merges token tags and NER labels from chunks in the specified format."%}
{% include templates/licensed_table_entry.md  name="NerChunker" summary="Extracts phrases that fits into a known pattern using the NER tags."%}
{% include templates/licensed_table_entry.md  name="NerConverterInternal" summary="Converts a IOB or IOB2 representation of NER to a user-friendly one, by associating the tokens of recognized entities and their label."%}
{% include templates/licensed_table_entry.md  name="NerDisambiguator" summary="Links words of interest, such as names of persons, locations and companies, from an input text document to a corresponding unique entity in a target Knowledge Base (KB)."%}
{% include templates/licensed_table_entry.md  name="MedicalNer" summary="This Named Entity recognition annotator is a generic NER model based on Neural Networks.."%}
{% include templates/licensed_table_entry.md  name="RENerChunksFilter" summary="Filters and outputs combinations of relations between extracted entities, for further processing."%}
{% include templates/licensed_table_entry.md  name="ReIdentification" summary="Reidentifies obfuscated entities by DeIdentification."%}
{% include templates/licensed_table_entry.md  name="RelationExtraction" summary="Extracts and classifies instances of relations between named entities."%}
{% include templates/licensed_table_entry.md  name="RelationExtractionDL" summary="Extracts and classifies instances of relations between named entities."%}
{% include templates/licensed_table_entry.md  name="SentenceEntityResolver" summary="Returns the normalized entity for a particular trained ontology / curated dataset (e.g. clinical ICD-10, RxNorm, SNOMED; financial SEC's EDGAR database,  etc) based on sentence embeddings."%}

<script> {% include scripts/approachModelSwitcher.js %} </script>

{% assign parent_path = "en/licensed_annotator_entries" %}

{% for file in site.static_files %}
  {% if file.path contains parent_path %}
    {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "licensed_annotator_entries/" %}
    {% include_relative {{ file_name }} %}
  {% endif %}
{% endfor %}
