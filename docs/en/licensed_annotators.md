---
layout: article
title: Licensed Annotators
permalink: /docs/en/licensed_annotators
key: docs-licensed-annotators
modify_date: "2020-02-13"
---

## Spark-NLP Licensed

The following annotators are available by buying a John Snow Labs Spark NLP license.
They are mostly meant for healthcare applications but other applications have been made with these NLP features.
Check out www.johnsnowlabs.com for more information.

### AssertionLogReg

It will classify each clinically relevant named entity into its assertion:

type: "present", "absent", "hypothetical", "conditional",
"associated_with_other_person", etc.

**Input types:** "sentence", "ner_chunk", "embeddings"

**Output type:** "assertion"

**Functions:**

- setLabelCol(label)
- setMaxIter(maxiter)
- setReg(lamda)
- setEnet(enet)
- setBefore(before)
- setAfter(after)
- setStartCol(s)
- setEndCol(e)
- setNerCol(n):
- setTargetNerLabels(v)

### AssertionDL

It will classify each clinically relevant named entity into its assertion
type: "present", "absent", "hypothetical", "conditional", "associated_with_other_person", etc.

**Input types:** "sentence", "ner_chunk", "embeddings"

**Output type:** "assertion"

**Functions:**

- setGraphFolder(p)
- setConfigProtoBytes(b)
- setLabelCol(label)
- setStartCol(s)
- setEndCol(e)
- setBatchSize(size)
- setEpochs(number)
- setLearningRate(lamda)
- setDropout(rate)
- setMaxSentLen(length):

### EntityResolver

Assigns a standard code (ICD10 CM, PCS; ICDO; CPT) to chunk tokens identified from TextMatchers or the NER Clinical Models.

**Input types:** "ner_chunk_tokenized", "embeddings"

**Output type:** "entity"

**Functions:**

- setLabelCol(k)
- setNeighbours(k)
- setThreshold(dist)
- setMergeChunks(merge)
- setMissAsEmpty(value)

### Chunk2Token

Transforms a complete chunk Annotation into a token Annotation without further tokenization, as opposed to ChunkTokenizer.

**Input types:** "chunk",

**Output type:** "token"

### ChunkEntityResolver

Assigns a standard code (ICD10 CM, PCS, ICDO; CPT) to chunk tokens identified from TextMatchers or the NER Clinical Models and embeddings pooled by ChunkEmbeddings

**Input types:** "chunk_token", "embeddings"

**Output type:** "resolution"

**Functions:**

- setSearchTree(s)
- setNeighbours(k)
- setThreshold(dist)
- setMissAsEmpty(value)

### DocumentLogRegClassifier

A convenient TFIDF-LogReg classifier that accepts "token" input type and outputs "selector"; an input type mainly used in RecursivePipelineModels

**Input types:** "token"

**Output type:** "selector"

**Functions:**

- setVectorizationModelPath(path_to_tfidfer)
- setClassificationModelPath(path_to_ovrlrc)
- setLabelCol(label_col)
- setMaxIter(int_val)
- setTol(float_val)
- setFitIntercept(bool_val)
- setLabels(label_list)

### ChunkEntityResolverSelector

An annotator to be used as the final stage of a RecursivePipelineModel in order to "select" a model from the previous stages based on a "selector" column 

**Input types:** "selector", "token", "embeddings"

**Output type:** "resolution*"

**Functions:**

- setSelectorArray(label_list)

### DeIdentificator

Identifies potential pieces of content with personal information about
patients and remove them by replacing with semantic tags.

**Input types:** "sentence", "token", "ner_chunk"

**Output type:** "deidentified"

**Functions:**

- setRegexPatternsDictionary(path, read_as, options)

### ContextSpellChecker

This spell checker utilizes tensorflow to do context based spell checking. At this moment, this annotator cannot be trained from Spark NLP. We are providing pretrained models only, for now.  
**Output type:** Token  
**Input types:** Tokenizer  

### References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018
