---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Preannotation
permalink: /docs/en/alab/preannotation
key: docs-training
modify_date: "2022-12-01"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}

es {
  font-weight: 400;
  font-style: italic;
}
</style>

Annotation Lab offers out-of-the-box support for <es>Named Entity Recognition</es>, <es>Classification</es>, <es>Assertion Status</es>, and <es>Relations</es> preannotations. These are extremely useful for bootstrapping any annotation project as the annotation team does not need to start the labeling from scratch but can leverage the existing knowledge transfer from domain experts to models. This way, the annotation efforts are significantly reduced.

To run pre-annotation on one or several tasks, the <es>Project Owner</es> or the <es>Manager</es> must select the target tasks and click on the `Pre-Annotate` button from the top right side of the <es>Tasks</es> page. It will display a popup with information regarding the last deployment of the model server with the list of models deployed and the labels they predict.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/preannotation_ner.gif" style="width:100%;"/>

This information is crucial, especially when multiple users are doing training and deployment in parallel. So before doing preannotations on your tasks, carefully check the list of currently deployed models and their labels.

If needed, users can deploy the models defined in the current project (based on the current Labeling Config) by clicking the _Deploy_ button. After the deployment is complete, the preannotation can be triggered.

<img class="image image__shadow image__align--center" src="/assets/images/annotation_lab/4.1.0/preannotate_dialog.png" style="width:40%;"/>

Since <bl>Annotation Lab 3.0.0</bl>, multiple preannotation servers are available to preannotate the tasks of a project. The dialog box that opens when clicking the _Pre-Annotate_ button on the <es>Tasks</es> page now lists available model servers in the options. <es>Project Owners</es> or <es>Managers</es> can now select the server to use. On selecting a model server, information about the configuration deployed on the server is shown on the popup so users can make an informed decision on which server to use.

In case a preannotation server does not exist for the current project, the dialog box also offers the option to deploy a new server with the current project's configuration. If this option is selected and enough resources are available (infrastructure capacity and a license if required) the server is deployed, and preannotation can be started. If there are no free resources, users can delete one or several existing servers from <es>Clusters</es> page under the <es>Settings</es> menu.

![preannotation_dialog](https://user-images.githubusercontent.com/26042994/161700555-1a46ef82-1ed4-41b8-b518-9c97767b1469.gif)

Concurrency is not only supported between preannotation servers but also between training and preannotation. Users can have training running on one project and preannotation running on another project at the same time.

## Preannotation Approaches

### Pretrained Models

On the <es>Predefined Labels</es> step of the <es>Project Configuration</es> page we can find the list of available models with their respective prediction labels. By selecting the relevant labels for your project and clicking the `Add Label` button you can add the predefined labels to your project configuration and take advantage of the Spark NLP auto labeling capabilities.

In the example below, we are reusing the `ner_posology` model that comes with 7 labels related to drugs.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/pretrained_models.png" style="width:100%;"/>

<br />

In the same manner classification, assertion status or relation models can be added to the project configuration and used for preannotation purpose.

Starting from version 4.3.0, Finance and Legal models downloaded from the Models Hub can be used for pre-annotation of NER, assertion status and classification projects. Visual NER models can now be downloaded from the NLP Models Hub, and used for pre-annotating image-based documents. Once you download the models from the Models Hub page, you can see the model's label in the <es>Predefined Label</es> tab on the project configuration page.

<img class="image image__shadow" src="https://user-images.githubusercontent.com/45035063/203519370-04cd1b4a-d02d-43ee-aa1b-3b6adf10ebb7.gif" style="width:100%;"/>

<br />

### Rules

Preannotation of NER projects can also be done using <es>Rules</es>. Rules are used to speed up the manual annotation process. Once a rule is defined, it is available for use in any project. However, for defining and running the rules we will need a <bl>[Healthcare NLP](/docs/en/licensed_install)</bl> license.

In the example below, we are reusing the available rules for preannotation.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/available_rules.png" style="width:100%;"/>

Read more on how to create rules and reuse them to speed up the annotation process [here](https://medium.com/annotation-lab/using-rules-to-jump-start-text-annotation-projects-1-3-john-snow-labs-8277a9c7fbcb).

## Text Preannotation

Preannotation is available for projects with text contents as the tasks. When you setup a project to use existing Spark NLP models for pre-annotation, you can run the designated models on all of your tasks by pressing the `Pre-Annotate` button on the top-right corner of the <es>Tasks</es> page.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/text_preannotation.png" style="width:100%;"/>

As a result, all predicted labels for a given task will be available in the <es>Prediction</es> widget on the Labeling page. The predictions are not editable. You can only view and navigate those or compare those with older predictions. However, you can create a new completion based on a given prediction. All labels and relations from such a new completion are now editable.

## Visual Preannotation

For running pre-annotation on one or several tasks, the <es>Project Owner</es> or the <es>Manager</es> must select the target tasks and can click on the `Pre-Annotate` button from the upper right side of the <es>Tasks</es> Page. It will display a popup with information regarding the last deployment of the model server, including the list of models deployed and the labels they predict.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/visual_ner_preannotation.gif" style="width:100%;"/>

**Known Limitations:**

1. When bulk pre-annotation runs on many tasks, the pre-annotation can fail due to memory issues.
2. Preannotation currently works at the token level, and does not merge all tokens of a chunk into one entity.

## Pipeline Limitations

Loading too many models in the preannotation server is not memory efficient and may not be practically required. Starting from version <bl>1.8.0</bl>, Annotation Lab supports maximum of five different models to be used for the preannotation server deployment.

Another restriction for Annotation Lab versions older than 4.2.0 is that two models trained on different embeddings cannot be used together in the same project. The Labeling Config will throw validation errors in any of the cases above, and we cannot save the configuration preventing preannotation server deployment.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/5_models.png" style="width:100%;"/>
