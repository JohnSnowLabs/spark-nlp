---
layout: docs
header: true
title: Wiki
permalink: /docs/en/wiki
key: docs-concepts
modify_date: "2022-09-27"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

This page is created for sharing some tips and tricks for the Spark NLP library. You can find valuable information under the related highlights.

</div>
<div class="h3-box" markdown="1">

### Miscellaneous

#### Loading the Same Model Into the Same Annotator More Than One Time

There is 1 instance of a model when we use `.pretrained` or `.load`. So when we try to use the same model with the same annotator more than one time with different parameters, it fails. We cannot have more than 1 model per annotator in the memory. **You can load 10 different `NerModel`, but not the same model twice with different parameters, it will just load once and reuse it the other times. It's not possible to duplicate the annotator/model unless the model is different. (each model creates a unique id).**

You can only load 1 model per annotator, once that happens that model with all its parameters stays in the memory. So if you want to load the very same model on the very same annotator in another pipeline, whether you use `.transform`, or LightPipeline, it will take the already loaded model from the memory. So if the first one has different inputCol/outputCol then the second pipeline just can't find the input/output or if the parameters are different in the second pipeline you may not see the desired outcome.

**So the lesson here is, if you want to use the same model in different places, you must make sure they all have the same parameters.** This behavior is the same for LP and `.transform`.

</div>
<div class="h3-box" markdown="1">

### LightPipeline

+ `LightPipeline` does not check the `storageRef` of resolver models. This feature will make LP so complicated and also slower. So, the resolver models can work with an embeddings model that is not trained with in `LightPipeline`, but they return irrelevant results.

</div>
<div class="h3-box" markdown="1">

### ChunkMergeApproach

#### Chunk Prioritization in ChunkMergeApproach
`ChunkMergeApproach()` has some prioritizing rules while merging chunks that come from entity extractors (NER models, `ContextualParser`, `TextMatcher`, `RegexMatcher`, etc.):

+ In case of the extracted chunks are same in the all given entity extractors, `ChunkMergeApproach` prioritizes the **leftmost** chunk output.

 *Example:* When we use `ner_posology` and `ner_clinical` models together, and if there is `insulin` in the clinical text, merger will behave like this:
 ```python
 chunk_merger = ChunkMergeApproach()\
      .setInputCols(["ner_posology_chunk", "ner_clinical_chunk"])\
      .setOutputCol("merger_output")
 ...

 >> ner_posology_chunk: insulin -> DRUG
 >> ner_clinical_chunk: insulin -> TREATMENT
 >> merger_output: insulin -> DRUG
 ```

+ In the event of chunk names being different but some of them are overlapped, `ChunkMergeApproach` prioritizes the **longest** chunk even though it is not in the leftmost.

 *Example:* If we use `ner_posology` and `ner_posology_greedy` models together in the same pipeline and merge their results on a clinical text that has "*... bactrim for 14 days ...*", merger result will be as shown below:

 ```python
 chunk_merger = ChunkMergeApproach()\
      .setInputCols(["ner_posology_chunk", "ner_posology_greedy_chunk"])\
      .setOutputCol("merger_output")
 ...

 >> ner_posology_chunk: bactrim -> DRUG
 >> ner_posology_greedy_chunk: bactrim for 14 days -> DRUG
 >> merger_output: bactrim for 14 days -> DRUG
 ```

 + **Confidence scores don't have any effect on prioritization.**

</div>
<div class="h3-box" markdown="1">

### Sentence Entity Resolver

#### Confidence vs Cosine Distance Calculation of Resolvers

Let's assume we have the 10 closest candidates (close meaning lower cosine distance) in our results. The confidence score is calculated with Softmax (vector to vector function). The vector is the full input and the output is also a full vector, it is not a function that is calculated item by item. Each item in the output depends on all the distances. So, what you are expecting is not “expected”.

If you get two distances 0.1 and 0.1, Softmax would return 0.5 and 0.5 for each. But if you have 0.1 and 10 distances, Softmax would be 1 and 0.

You can have a low distance (chunks are very similar semantically) but low confidence if there are many other chunks also very similar. And sometimes you can have high confidence but high distance, meaning there is only one chunk "close" to your target but not so close.

In general, *we can see less distance and less confidence but not perfect linear relationships*. We can say that **using the distance is a better parameter to judge the "goodness" of the resolution than the confidence**. So, We recommend that you consider the cosine distance.

</div>