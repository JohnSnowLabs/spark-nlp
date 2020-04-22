---
layout: article
title: Licensed Annotators
permalink: /docs/en/licensed_annotators
key: docs-licensed-annotators
modify_date: "2020-04-21"
---

## Spark-NLP Licensed

The following annotators are available by buying a John Snow Labs Spark NLP license.
They are mostly meant for healthcare applications but other applications have been made with these NLP features.
Check out www.johnsnowlabs.com for more information.

### AssertionLogReg 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel">Transformer scaladocs</a>

It will classify each clinically relevant named entity into its assertion:

type: "present", "absent", "hypothetical", "conditional",
"associated_with_other_person", etc.

**Input types:** `"sentence", "ner_chunk", "embeddings"`

**Output type:** `"assertion"`

**Parameter Setters:**
```
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
```

### AssertionDL 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel">Transformer scaladocs</a>

It will classify each clinically relevant named entity into its assertion
type: "present", "absent", "hypothetical", "conditional", "associated_with_other_person", etc.

**Input types:** "sentence", "ner_chunk", "embeddings"

**Output type:** "assertion"

**Parameter Setters:**
```
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
```

### Chunk2Token
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.Chunk2Token">Transformer scaladocs</a>

Transforms a complete chunk Annotation into a token Annotation without further tokenization, as opposed to ChunkTokenizer.

**Input types:** "chunk",

**Output type:** "token"

### ChunkEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.ChunkEntityResolverApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.ChunkEntityResolverModel">Transformer scaladocs</a>

Assigns a standard code (ICD10 CM, PCS, ICDO; CPT) to chunk tokens identified from TextMatchers or the NER Clinical Models and embeddings pooled by ChunkEmbeddings

**Input types:** "chunk_token", "embeddings"

**Output type:** "resolution"

**Parameter Setters:**
```
- setNeighbours($(neighbours))
- setAlternatives($(alternatives))
- setThreshold($(threshold))
- setExtramassPenalty($(extramassPenalty))
- setEnableWmd($(enableWmd))
- vsetEnableTfidf($(enableTfidf))
- setEnableJaccard($(enableJaccard))
- setEnableSorensenDice($(enableSorensenDice))
- setEnableJaroWinkler($(enableJaroWinkler))
- setEnableLevenshtein($(enableLevenshtein))
- setDistanceWeights($(distanceWeights))
- setPoolingStrategy($(poolingStrategy))
- setMissAsEmpty($(missAsEmpty))
```

### EnsembleEntityResolver
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.EnsembleEntityResolverApproach">Estimator scaladocs</a> | 
<a href="https://nlp.johnsnowlabs.com/licensed/api/index.html#com.johnsnowlabs.nlp.annotators.resolution.EnsembleEntityResolverModel">Transformer scaladocs</a>

Assigns a standard code (RxNorm, SNOMED, UMLS) to chunk tokens identified from TextMatchers or the NER Clinical Models and embeddings pooled by ChunkEmbeddings.
Designed to scale on a sub-log rate compared to the cardinality of the dataset

**Input types:** "chunk_token", "embeddings"

**Output type:** "resolution"

**Parameter Setters:**
```
- setClassifierLabelCol
- setMaxIter
- setTol
- setFitIntercept
- setIdfModelPath
- setOvrModelPath
- setClassifierLabels
- setResolverLabelCol
- setNormalizedCol
- setNeighbours
- setAlternatives
- setThreshold
- setExtramassPenalty
- setEnableWmd
- vsetEnableTfidf
- setEnableJaccard
- setEnableSorensenDice
- setEnableJaroWinkler
- setEnableLevenshtein
- setDistanceWeights
- setPoolingStrategy
- setMissAsEmpty
```
### DocumentLogRegClassifier

A convenient TFIDF-LogReg classifier that accepts "token" input type and outputs "selector"; an input type mainly used in RecursivePipelineModels

**Input types:** "token"

**Output type:** "category"

**Parameter Setters:**
```
- setVectorizationModelPath(path_to_tfidfer)
- setClassificationModelPath(path_to_ovrlrc)
- setLabelCol(label_col)
- setMaxIter(int_val)
- setTol(float_val)
- setFitIntercept(bool_val)
- setLabels(label_list)
```


### DeIdentificator

Identifies potential pieces of content with personal information about
patients and remove them by replacing with semantic tags.

**Input types:** "sentence", "token", "ner_chunk"

**Output type:** "deidentified"

**Functions:**

- setRegexPatternsDictionary(path, read_as, options)

### ContextSpellChecker

This spell checker utilizes TensorFlow to do context based spell checking. At this moment, this annotator cannot be trained from Spark NLP. We are providing pretrained models only, for now.  
**Output type:** Token  
**Input types:** Tokenizer  

### References

[1] Speech and Language Processing. Daniel Jurafsky & James H. Martin. 2018
