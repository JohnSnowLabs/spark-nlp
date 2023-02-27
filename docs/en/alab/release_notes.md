---
layout: docs
comment: no
header: true
seotitle: Release Notes | John Snow Labs
title: Release Notes
permalink: /docs/en/alab/release_notes
key: docs-training
modify_date: "2023-02-08"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.7.1

Release date: **22-02-2023**

The latest version of NLP Lab, version 4.7.1, brings several enhancements that are worth highlighting. One of the most notable improvements is in relation prompts. NLP Lab now offers support for combining NER models, prompts and rules when defining relation prompts. 

The playground feature in NLP Lab has also received some noteworthy upgrades in version 4.7.1. The "playground" environment was initially added to facilitate experiments with different NLP models, tweak prompts and rules, and explore the potential of language models in a safe, sandboxed environment. The improvements made to the playground in this version are expected to enhance the overall user experience, and to make the environment faster and more responsive.

In addition to these improvements, the latest version of NLP Lab has extended support for importing large task archives. This means that users can now work with bigger datasets more efficiently, which will undoubtedly save them time and effort.
Below are the specifics of the additions included in this release:


## Improvements in Prompts
### Build Relation Prompts using NER Models, Prompts and Rules
In previous version, relation prompts could be defined based on NER models and rules. In this release, NLP Lab allows for NER prompts to be reused when defining relation prompts. To include a NER prompt within a relation prompt, users need to navigate to the Questions section of the Relation Prompt creation page and search for the prompt to reuse. Once the NER prompt has been selected, users can start defining the question patterns. For example, users could create prompts that identify the relationship between people and the organizations they work for, or prompts that identify the relationship between a place and its geographic coordinates. The ability to incorporate NER prompts into relation prompts is a significant advancement in prompts engineering, and it opens up new possibilities for more sophisticated and accurate natural language processing.

<img width="774" alt="image" src="https://user-images.githubusercontent.com/46840490/218919293-d931ed1f-10d1-4a97-b3e4-4d705546fb26.png">

## Improvements in Playground
### Direct Navigation to Active Playground Sessions

Navigating between multiple projects to and from the playground experiments can be necessary, especially when you want to revisit a previously edited prompt or rule. This is why NLP Lab Playground now allow users to navigate to any active Playground session without having to redeploy the server. 
This feature enables users to check how their resources (models, rules and prompts) behave at project level, compare the preannotation results with ground truth, and quickly get back to experiments for modifying prompts or rules without losing progress or spending time on new deployments. This feature makes experimenting with NLP prompts and rules in a playground more efficient, streamlined, and productive.

![reopen_playground](https://user-images.githubusercontent.com/26042994/219060474-0c6fc8ab-f946-4ea5-886c-659f357b7f7d.gif)

### Automatic Deployment of Updated Rules/Prompts 

Another benefit of experimenting with NLP prompts and rules in the playground is the immediate feedback that you receive. When you make changes to the parameters of your rules or to the questions in your prompts, the updates are deployed instantly. Manually deploying the server is not necessary any more for changes made to Rules/Prompts to be reflected in the preannotation results. Once the changes are saved, by simply clicking on the Test button, updated results are presented. 
This allows you to experiment with a range of variables and see how each one affects the correctness and completeness of the results. The real-time feedback and immediate deployment of changes in the playground make it a powerful tool for pushing the boundaries of what is possible with language processing.

### Playground Server Destroyed after 5 Minutes of Inactivity

When active, the NLP playground consumes resources from your server. For this reason, NLP Lab defines an idle time limit of 5 minutes after which the playground is automatically destroyed. This is done to ensure that the server resources are not being wasted on idle sessions. When the server is destroyed, a message is displayed, so users are aware that the session has ended. Users can view information regarding the reason for the Playground's termination, and have the option to restart by pressing the Restart button.

![Screenshot 2023-02-15 at 9 02 36 PM](https://user-images.githubusercontent.com/26042994/219069508-fc241f70-1544-4c68-ba3e-9aa7158a065a.png)

### Playground Servers use Light Pipelines
The replacement of regular preannotation pipelines with Light Pipelines has a significant impact on the performance of the NLP playground. Light pipelines allow for faster initial deployment, quicker pipeline update and fast processing of text data, resulting in overall quicker results in the UI.


### Direct Access to Model Details Page on the Playground
Another useful feature of NLP Lab Playground is the ability to quickly and easily access information on the models being used. This information can be invaluable for users who are trying to gain a deeper understanding of the model’s inner workings and capabilities. In particular, by click on the model’s name it is now possible to navigate to the NLP Models hub page. This page provides users with additional details about the model, including its training data, architecture, and performance metrics. By exploring this information, users can gain a better understanding of the model’s strengths and weaknesses, and use this knowledge to make more informed decisions on how good the model is for the data they need to annotate. 

![model_link](https://user-images.githubusercontent.com/26042994/219068322-0b0ccc7a-6acb-4522-b0ca-13eba6973c40.gif)


## Improvements in Task Import

### Support for Large Document Import
One of the challenges when working on big annotation projects is dealing with large size tasks, especially when uploading them to the platform. This is particularly problematic for files/archives larger than 20 MB, which can often lead to timeouts and failed uploads. To address this issue, NLP Lab has implemented chunk file uploading on the task import page.
Chunk file uploading is a method that breaks large files into smaller, more manageable chunks. This process makes the uploading of large files smoother and more reliable, as it reduces the risk of timeouts and failed uploads. This is especially important for NLP practitioners who work with large datasets, as it allows them to upload and process their data more quickly and effectively.

![chunk-file-uploading](https://user-images.githubusercontent.com/10126570/218990805-8128a475-d33b-4fd8-9397-6aaa6895c9c7.gif)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
    <li class="active"><a href="annotation_labs_releases/release_notes_4_7_1">4.7.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_6_5">4.6.5</a></li>    
    <li><a href="annotation_labs_releases/release_notes_4_6_3">4.6.3</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_6_2">4.6.2</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_5_1">4.5.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_5_0">4.5.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_4_1">4.4.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_4_0">4.4.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_3_0">4.3.0</a></li>
	<li><a href="annotation_labs_releases/release_notes_4_2_0">4.2.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_1_0">4.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_5_0">3.5.0</a></li>
	<li><a href="annotation_labs_releases/release_notes_3_4_1">3.4.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_4_0">3.4.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_3_1">3.3.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_3_0">3.3.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_2_0">3.2.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_1_1">3.1.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_1_0">3.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_0_1">3.0.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_0_0">3.0.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_8_0">2.8.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_2">2.7.2</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_1">2.7.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_0">2.7.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_6_0">2.6.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_5_0">2.5.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_4_0">2.4.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_3_0">2.3.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_2_2">2.2.2</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_1_0">2.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_0_1">2.0.1</a></li>
</ul>
