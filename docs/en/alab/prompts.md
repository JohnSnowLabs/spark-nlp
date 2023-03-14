---
layout: docs
comment: no
header: true
seotitle: NLP Lab | John Snow Labs
title: Prompts
permalink: /docs/en/alab/prompts
key: docs-training
modify_date: "2023-02-10"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
es {
    font-weight:400;
    font-style: italic;
}
</style>

NLP Lab offers support for prompt engineering. On the `Prompts` page, from the resources `HUB`, users can easily discover and explore the existing prompts or create new prompts for identifying entities or relations. Currently, NLP Lab supports prompts for Healthcare, Finance, and Legal domains applied using pre-trained question-answering language models published on the NLP Models Hub and available to download in one click. The main advantage behind the use of prompts in entity or relation recognition is the ease of definition. Non-technical domain experts can easily create prompts, test and edit them on the `Playground` on custom text snippets and, when ready, deploy them for pre-annotation as part of larger NLP projects. Together with rules, prompts are very handy in situations where no pre-trained models exist, for the target entities and domains. With rules and prompts the annotators never start their projects from scratch but can capitalize on the power of zero-shot models and rules to help them pre-annotate the simple entities and relations and speed up the annotation process. As such, the NLP Lab ensures fewer manual annotations are required from any given task.

## Creating NER Prompts

NER prompts, can be used to identify entities in natural language text documents. Those can be created based on healthcare, finance, and legal zero-shot models selectable from the "Domain" dropdown. For one prompt, the user adds one or more questions for which the answer represents the target entity to annotate.

   ![entity_prompt](https://user-images.githubusercontent.com/26042994/211890279-2ea02cd5-36fa-4b56-86fd-38b0c20ba880.gif)

## Creating Relation Prompts

Prompts can also be used to identify relations between entities for healthcare, finance, and legal domains. The domain-specific zero-shot model to use for detecting relation can be selected from the "Domain" dropdown. The relation prompts are defined by a pair of entities related by a predicate. The entities can be selected from the available dropdowns listing all entities available in the current NLP Lab (included in available NER models, prompts or rules) for the specified domain. 
   
   ![relation_prompt](https://user-images.githubusercontent.com/26042994/211890317-362f193c-b80b-4caa-b242-69df6fa8a257.gif)

## Mix and Match models, rules, and prompts

The project configuration page was simplified by grouping into one page all available resources that can be reused for pre-annotation: models, rules, and prompts. Users can easily mix and match the relevant resources and add them to their configuration. 

![updated_configuration_page](https://user-images.githubusercontent.com/26042994/211890361-14c5b17c-762d-4d0a-a6a6-0ac235565aa0.gif)

**Note:** One project configuration can only reuse the prompts defined by one single zero-shot model. Prompts created based on multiple zero-shot models (e.g. finance or legal or healthcare) cannot be mixed into the same project because of high resource consumption. Furthermore, all prompts require a license with a scope that matches the domain of the prompt.

## Zero-Shot Models available in the NLP Models Hub
NLP Models Hub now lists the newly released zero-shot models that are used to define prompts. These models need to be downloaded to NLP Lab instance before prompts can be created. A valid license must be available for the models to be downloaded to NLP Lab.

![Zero-shot-models](https://user-images.githubusercontent.com/26042994/211890478-3aa90dfc-f474-42c8-a73f-ce6c3efecbbe.png)