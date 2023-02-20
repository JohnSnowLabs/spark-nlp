---
layout: docs
comment: no
header: true
seotitle: NLP Lab | John Snow Labs
title: Playground
permalink: /docs/en/alab/playground
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

The Playground feature of the NLP Lab allows users to deploy and test models, rules, and/or prompts without going through the project setup wizard. This simplifies the initial resources exploration, and facilitates experiments on custom data.  Any model, rule, or prompt can now be selected and deployed for testing by clicking on the "Open in Playground" button.


## Experiment with Rules

Rules can be  deployed to the Playground from the rules page. When a particular rule is deployed in the playground, the user can also change the parameters of the rules via the rule definition form from the right side of the page. After saving the changes users need to click on the "Deploy" button to refresh the results of the pre-annotation on the provided text.

![ruledeploy](https://user-images.githubusercontent.com/33893292/209978724-355fd3a2-aab4-4c38-869d-4f3432c657d3.gif)

## Experiment with Prompts 

NLP Lab's Playground also supports the deployment and testing of prompts. Users can quickly test the results of applying a prompt on custom text, can easily edit the prompt, save it, and deploy it right away to see the change in the pre-annotation results.

![demo3](https://user-images.githubusercontent.com/33893292/213699722-543d13f6-c410-4398-83a1-26a832a032ca.gif)


## Experiment with Models
Any Classification, NER or Assertion Status model available on the NLP Lab can also be deployed to Playground for testing on custom text. 

![Playground deployment](https://user-images.githubusercontent.com/33893292/209965776-1c0a6b07-5526-496e-97f8-4b7a2cf2a6d1.gif)

Deployment of models and rules is supported by floating and air-gapped licenses. Healthcare, Legal, and Finance models require a license with their respective scopes to be deployed in Playground. Unlike pre-annotation servers, only one playground can be deployed at any given time.
