---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.4
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_4
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

### 2.7.4

We are glad to announce that Spark NLP for Healthcare 2.7.4 has been released!  

#### Highlights:

- Introducing a new annotator to extract chunks with NER tags using regex-like patterns: **NerChunker**.
- Introducing two new annotators to filter chunks: **ChunkFilterer** and **AssertionFilterer**.
- Ability to change the entity type in **NerConverterInternal** without using ChunkMerger (`setReplaceDict`).
- In **DeIdentification** model, ability to use `faker` and static look-up lists at the same time randomly in `Obfuscation` mode.
- New **De-Identification NER** model, augmented with synthetic datasets to detect uppercased name entities.
- Bug fixes & general improvements.

#### 1. NerChunker:

Similar to what we used to do in **POSChunker** with POS tags, now we can also extract phrases that fits into a known pattern using the NER tags. **NerChunker** would be quite handy to extract entity groups with neighboring tokens when there is no pretrained NER model to address certain issues. Lets say we want to extract clinical findings and body parts together as a single chunk even if there are some unwanted tokens between.

**How to use:**

    ner_model = NerDLModel.pretrained("ner_radiology", "en", "clinical/models")\
        .setInputCols("sentence","token","embeddings")\
        .setOutputCol("ner")

    ner_chunker = NerChunker().\
        .setInputCols(["sentence","ner"])\
        .setOutputCol("ner_chunk")\
        .setRegexParsers(["<IMAGINGFINDINGS>*<BODYPART>"])

    text = 'She has cystic cyst on her kidney.'

    >> ner tags: [(cystic, B-IMAGINGFINDINGS), (cyst,I-IMAGINGFINDINGS), (kidney, B-BODYPART)
    >> ner_chunk: ['cystic cyst on her kidney']


#### 2. ChunkFilterer:

**ChunkFilterer** will allow you to filter out named entities by some conditions or predefined look-up lists, so that you can feed these entities to other annotators like Assertion Status or Entity Resolvers.  It can be used with two criteria: **isin** and **regex**.

**How to use:**

    ner_model = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")\
          .setInputCols("sentence","token","embeddings")\
          .setOutputCol("ner")

    ner_converter = NerConverter() \
          .setInputCols(["sentence", "token", "ner"]) \
          .setOutputCol("ner_chunk")

    chunk_filterer = ChunkFilterer()\
          .setInputCols("sentence","ner_chunk")\
          .setOutputCol("chunk_filtered")\
          .setCriteria("isin") \
          .setWhiteList(['severe fever','sore throat'])

    text = 'Patient with severe fever, sore throat, stomach pain, and a headache.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> chunk_filtered: ['severe fever','sore throat']


#### 3. AssertionFilterer:

**AssertionFilterer** will allow you to filter out the named entities by the list of acceptable assertion statuses. This annotator would be quite handy if you want to set a white list for the acceptable assertion statuses like `present` or `conditional`; and do not want `absent` conditions get out of your pipeline.

**How to use:**

    clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
      .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
      .setOutputCol("assertion")

    assertion_filterer = AssertionFilterer()\
      .setInputCols("sentence","ner_chunk","assertion")\
      .setOutputCol("assertion_filtered")\
      .setWhiteList(["present"])


    text = 'Patient with severe fever and sore throat, but no stomach pain.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> assertion_filtered: ['severe fever','sore throat']


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_7_3">Version 2.7.3</a>
    </li>
    <li>
        <strong>Version 2.7.4</strong>
    </li>
    <li>
        <a href="release_notes_2_7_5">Version 2.7.5</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li class="active"><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>