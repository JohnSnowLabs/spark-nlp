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


## Direct Navigation to Active Playground Sessions

Navigating between multiple projects to and from the playground experiments can be necessary, especially when you want to revisit a previously edited prompt or rule. This is why NLP Lab Playground now allow users to navigate to any active Playground session without having to redeploy the server. 
This feature enables users to check how their resources (models, rules and prompts) behave at project level, compare the preannotation results with ground truth, and quickly get back to experiments for modifying prompts or rules without losing progress or spending time on new deployments. This feature makes experimenting with NLP prompts and rules in a playground more efficient, streamlined, and productive.

![reopen_playground](https://user-images.githubusercontent.com/26042994/219060474-0c6fc8ab-f946-4ea5-886c-659f357b7f7d.gif)

## Automatic Deployment of Updated Rules/Prompts 

Another benefit of experimenting with NLP prompts and rules in the playground is the immediate feedback that you receive. When you make changes to the parameters of your rules or to the questions in your prompts, the updates are deployed instantly. Manually deploying the server is not necessary any more for changes made to Rules/Prompts to be reflected in the preannotation results. Once the changes are saved, by simply clicking on the Test button, updated results are presented. 
This allows you to experiment with a range of variables and see how each one affects the correctness and completeness of the results. The real-time feedback and immediate deployment of changes in the playground make it a powerful tool for pushing the boundaries of what is possible with language processing.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.7.1/automatic_deployment.gif" style="width:100%;"/>

## Playground Server Destroyed after 5 Minutes of Inactivity

When active, the NLP playground consumes resources from your server. For this reason, NLP Lab defines an idle time limit of 5 minutes after which the playground is automatically destroyed. This is done to ensure that the server resources are not being wasted on idle sessions. When the server is destroyed, a message is displayed, so users are aware that the session has ended. Users can view information regarding the reason for the Playground's termination, and have the option to restart by pressing the Restart button.

![Screenshot 2023-02-15 at 9 02 36 PM](https://user-images.githubusercontent.com/26042994/219069508-fc241f70-1544-4c68-ba3e-9aa7158a065a.png)

## Playground Servers use Light Pipelines
The replacement of regular preannotation pipelines with Light Pipelines has a significant impact on the performance of the NLP playground. Light pipelines allow for faster initial deployment, quicker pipeline update and fast processing of text data, resulting in overall quicker results in the UI.


## Direct Access to Model Details Page on the Playground
Another useful feature of NLP Lab Playground is the ability to quickly and easily access information on the models being used. This information can be invaluable for users who are trying to gain a deeper understanding of the model’s inner workings and capabilities. In particular, by click on the model’s name it is now possible to navigate to the NLP Models hub page. This page provides users with additional details about the model, including its training data, architecture, and performance metrics. By exploring this information, users can gain a better understanding of the model’s strengths and weaknesses, and use this knowledge to make more informed decisions on how good the model is for the data they need to annotate. 

![model_link](https://user-images.githubusercontent.com/26042994/219068322-0b0ccc7a-6acb-4522-b0ca-13eba6973c40.gif)
