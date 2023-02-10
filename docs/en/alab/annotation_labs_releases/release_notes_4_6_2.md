---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 4.6.2
permalink: /docs/en/alab/annotation_labs_releases/release_notes_4_6_2
key: docs-licensed-release-notes
modify_date: 2023-01-21
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.6.2

Release date: **21-01-2023**

NLP Lab 4.6.2 comes with support for zero-shot learning via prompts. Prompt engineering is a very recent but rapidly growing discipline that aims to guide language models such as GPT-3 to generate specific and desired outputs, such as answering a question or writing a coherent story. This version of the NLP Lab, adds support for the creation and use of prompts for entities and relations identification within text documents. 
The goal of prompt engineering in this context is designing and crafting some questions, which are fed into a question-answering model together with some input text. The purpose is to guide the language model to generate specific and desired outputs, such as identifying entities or relations within the input text. 
This release offers features such as creation and editing of prompts, a dedicated section for prompts management and sharing inside the resources Hub, an optimized configuration page allowing mixing models, prompts, and rules into the same project, and support for quick prompts deployments and testing to the Playground.


## Prompts on the Hub
The resources Hub has a new page dedicated to prompts. It allows users to easily discover and explore the existing prompts or create new prompts for identifying entities or relations. Currently, NLP Lab supports prompts for Healthcare, Finance, and Legal domains applied using pre-trained question-answering language models published on the NLP Models Hub and available to download in one click. The main advantage behind the use of prompts in entity or relation recognition is the ease of definition. Non-technical domain experts can easily create prompts, test and edit them on the playground on custom text snippets and, when ready, deploy them for pre-annotation as part of larger NLP projects. 
Together with rules, prompts are very handy in situations where no pre-trained models exist, for the target entities and domains. With rules and prompts the annotators never start their projects from scratch but can capitalize on the power of zero-shot models and rules to help them pre-annotate the simple entities and relations and speed up the annotation process. As such the NLP Lab ensures fewer manual annotations are required from any given task.


  - **Creating NER Prompts**

NER prompts, can be used to identify entities in natural language text documents. Those can be created based on healthcare, finance, and legal zero-shot models selectable from the "Domain" dropdown. For one prompt, the user adds one or more questions for which the answer represents the target entity to annotate.

   ![entity_prompt](https://user-images.githubusercontent.com/26042994/211890279-2ea02cd5-36fa-4b56-86fd-38b0c20ba880.gif)

  - **Creating Relation Prompts**

Prompts can also be used to identify relations between entities for healthcare, finance, and legal domains. The domain-specific zero-shot model to use for detecting relation can be selected from the "Domain" dropdown. The relation prompts are defined by a pair of entities related by a predicate. The entities can be selected from the available dropdowns listing all entities available in the current NLP Lab (included in available NER models or rules) for the specified domain. 
   
   ![relation_prompt](https://user-images.githubusercontent.com/26042994/211890317-362f193c-b80b-4caa-b242-69df6fa8a257.gif)

## A simplified configuration wizard allows the reuse of models, rules, and prompts
The project configuration page was simplified by grouping into one page all available resources that can be reused for pre-annotation: models, rules, and prompts. Users can easily mix and match the relevant resources and add them to their configuration. 

![updated_configuration_page](https://user-images.githubusercontent.com/26042994/211890361-14c5b17c-762d-4d0a-a6a6-0ac235565aa0.gif)

**Note:** One project configuration can only reuse the prompts defined by one single zero-shot model. Prompts created based on multiple zero-shot models (e.g. finance or legal or healthcare) cannot be mixed into the same project because of high resource consumption. Furthermore, all prompts require a license with a scope that matches the domain of the prompt.

## Experiment with prompts in Playground
NLP Lab's Playground supports the deployment and testing of prompts. Users can quickly test the results of applying a prompt on custom text, can easily edit the prompt, save it, and deploy it right away to see the change in the pre-annotation results.

![demo3](https://user-images.githubusercontent.com/33893292/213699722-543d13f6-c410-4398-83a1-26a832a032ca.gif)

## Zero-Shot Models available in the NLP Models Hub
NLP Models Hub now lists the newly released zero-shot models that are used to define prompts. These models need to be downloaded to NLP Lab instance before prompts can be created. A valid license must be available for the models to be downloaded to NLP Lab.

![Zero-shot-models](https://user-images.githubusercontent.com/26042994/211890478-3aa90dfc-f474-42c8-a73f-ce6c3efecbbe.png)

## Bug Fixes

- **Error while deploying classification model to the playground**

  Previously, deploying the classification model to the playground had some issues which have been fixed in this version.

- **Information on the model's details not visible completely on the playground **

  In this version, we have fixed an issue related to the visibility of the information for Edition, Uploaded by, and Source inside the Models Detail accordion. Now, the UI can handle long model names on the playground page.

- **Undo and Reset buttons are not working**

  With release 4.6.2, issues regarding undo/redo buttons in the labeling page for annotated tokens have been fixed. Now, the Undo and Redo button works as expected.

- **Finance and Legal models cannot be downloaded to NLP Lab with a floating license from Models Hub**

  Earlier, users were not able to download the Finance and Legal model from the NLP Models HUB page using floating licenses. This issue has been fixed. Now, legal and finance models are downloadable in the NLP lab using a floating license.

- **Pre-annotation server cannot be deployed for Visual NER**

  This version also fixes the issue of failing to deploy the pre-annotation server for Visual NER models.

- **Draft saved is seen for submitted completion**

  Previously, in the NER task when the user clicked on regions of a previously submitted completion and viewed the versions submitted by the users, a draft was saved. A draft should not be created and saved for submitted completions. This issue was fixed in 4.6.2.

- **Training fails for NER when embedding_clinical is used and the license type is open-source**

  Earlier it was not possible to train a NER model with the open-source library using embeddings_clinical. This issue has been fixed. Hence users can now train open-sourced models with embeddings_clinical.

- **UI goes blank for the Visual NER project when an annotation is saved and the next button is clicked**

  In the previous version, annotators were not served the next task after clicking the Next button. A blank page with a console error was seen. Now the next task is served in the Visual NER project without any error.

- **Pre-annotation server cannot be deployed for RE model**

  There was an issue with the deployment of trained NER models with a relation extraction model. This issue has been fixed in this version.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}