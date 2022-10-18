---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Model Training  
permalink: /docs/en/alab/active_learning
key: docs-training
modify_date: "2022-04-05"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

A **Project Owner** or a **Manager** can use the completed tasks (completions) from a project for training a new Spark NLP model. The training feature can be found on the Setup page.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/train_setup_label.png" style="width:80%;"/>
<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/train_setup_pipeline.png" style="width:80%;"/>
<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/train_setup_model.png" style="width:100%;"/>

## Named Entity Recognition Projects
Named Entity Recognition (NER) projects usually include several labels. When the annotation team has generated a relevant sample of training data/examples for each one of the labels the Project Owner/Manager can use this data to train an DL model which can then be used to predict the labels on new tasks. 

The NER models can be easily trained as illustrated below. 

The "Train Now" button (item 5) can be used to trigger training of a new model when no other trainings or preannotations are in progress. Otherwise, the button is disabled. Information on the training progress is shown in the top right corner of Model Training tab. Here the user can get indications on the success or failure message depending on how the last training ended.

When triggering the training, users are prompted to choose either to immediately deploy models or just do training. If immediate deployment is chosen, then the Labeling config is updated according to the name of the new model (item 1 on the above image).

It is possible to download training logs by clicking on the download logs icon (see item 8 on the above image) of the recently trained NER model which includes information like training parameters and TF graph used along with precision, recall, f1 score, etc.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/train_ner.gif" style="width:100%;"/>

## Training parameters

In Annotation Lab versions prior to 1.8.0, for mixed projects containing multiple types of annotations in a single project like classifications, NER, and assertion status, multiple trainings were triggered at the same time using the same system resources and Spark NLP resources. In this case, the training component could fail because of resource limitations.

In order to improve the usability of the system, Annotation Lab 1.8.0 added dropdown options to choose which type of training to run next. The project Owner or Manager of a project can scroll down to Training Settings and choose the training type. The drop-down gives a list of possible training types for that particular project based on defined Labeling Config. Another drop-down also lists available embeddings which can be used for training the model.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/trainig_models.png" style="width:80%;"/>


It is possible to tune the most common training parameters (Validation split ratio, Epoch, Learning rate, Decay, Dropout, and Batch) by editing their values in Training Parameters.

It is also possible to train a model by using a sublist of tasks with predefined tags. This is done by specifying the targeted Tags on the Training Parameters (last option).

The Annotation Lab v1.8.0 includes additional filtering options for the training dataset based on the status of completions, either all submitted completions cab be used for training or only the reviewed ones.

## Transfer Learning
Annotation Lab 2.0.0+ supports the Transfer Learning feature offered by [Spark NLP for Healthcare 3.1.2](https://nlp.johnsnowlabs.com/docs/en/licensed_release_notes#support-for-fine-tuning-of-ner-models). 
This feature is available for project manages and project owners, but only if a valid Spark NLP for Healthcare license is loaded into the Annotation Lab. 
In this case, the feature can be activated for any project by navigating to the Setup->Training & Active Learning. It requires the presence of a `base model` trained with [MedicalNERModel](https://nlp.johnsnowlabs.com/docs/en/licensed_release_notes#1-medicalnermodel-annotator).

If a MedicalNER model is available on the Models Hub section of the Annotation Lab, it can be chosen as a starting point of the training process. This means the `base model` will be Fine Tuned with the new training data.

When Fine Tuning is enabled, the same embeddings used for training the `base model` will be used to train the new model. Those need to be available on the Models Hub section as well. If present, embeddings will be automatically selected, otherwise users must go to the Models Hub page and download or upload them.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/transfer_learning.gif" style="width:80%;"/>

## Custom Training Script
If users want to change the default Training script present within the Annotation Lab, they can upload their own training pipeline. In the Training section of the Project Setup Page, admin users can upload the training scripts. At the moment we are supporting custom training script just for NER projects.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/custom_script.png" style="width:80%;"/>

## Selection of Completions
During the annotation project lifetime, normally not all tasks/completions are ready to be used as a training dataset. This is why the training process selects completions based on their status:
- Filter tasks by tags (if defined in Training Parameters window, otherwise all tasks are considered)
- For completed tasks, completions to be taken into account are also selected based on the following criteria:
  - If a task has a completion accepted by a reviewer this is selected for training and all others are ignored
  - Completions rejected by a Reviewer are not used for training
  - If no reviewer is assigned to a task that has multiple submitted completions the most recent completion is selected for training purpose

## Assertion Status Projects

NER configurations for the healthcare domain are often mixed with Assertion Status labels. In this case Annotation Lab offers support for training both types of models in one go. After the training is complete, the models will be listed in the Spark NLP Pipeline Config. Hovering mouse over the model name in the Spark NLP pipeline Config, the user can see more information about the model such as when it was trained and if the training was manually initiated or by the Active Learning process.

Once the model(s) has been trained, the project configuration will be automatically updated to reference the new model for prediction. Notice below, for the Assertion Status **<Label>** tag the addition of model attribute to indicate which model will be used for task preannotation for this label.

```bash
    <Label value="Absent" assertion="true" model="assertion_jsl_annotation_manual.model"/>
    <Label value="Past" assertion="true" model="assertion_jsl_annotation_manual.model"/>
```

It is not possible to mark a label as an Assertion Status label and use a NER model to predict it. A validation error is shown in the Interface Preview in case an invalid Assertion model is used.
<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/as_notification.png" style="width:70%;"/>

The Annotation Lab only allows the use of one single Assertion Status model in the same project.
<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/one_as.png" style="width:70%;"/>


## Classification Project Models Training
Annotation Lab supports two types of classification training **Single Choice Classification** and **Multi-Choice Classification**. For doing so, it uses three important attributes of the **<Choices>** tag to drive the Classification Models training and preannotation. Those are **name**, 
	**choice** and **train**.
### Attribute name
The attribute name allows the naming of the different choices present in the project configuration, and thus the training of separate models based on the same project annotations. For example, in the sample configuration illustrated below, the name="age" attribute, tells the system to only consider age-related classification information when training an Age Classifier. The value specified by the name attribute is also used to name the resulting Classification model (classification_age_annotation_manual).

### Attribute choice
The choice attribute specifies the type of model that will be trained: multiple or single. For example, in the Labeling Config below, Age and Gender are Single Choice Classification categories while the Smoking Status is Multi-Choice Classification. Depending upon the value of this attribute, the respective model will be trained as a Single Choice Classifier or Multi-Choice Classifier.
```bash
<View>
  <View style="overflow: auto;">
    <Text name="text" value="$text"/>
  </View>
  <Header value="Smoking Status"/>
  <Choices name="smokingstatus" toName="text" choice="multiple" showInLine="true">
    <Choice value="Smoker"/>
    <Choice value="Past Smoker"/>
    <Choice value="Nonsmoker"/>
  </Choices>
  <Header value="Age"/>
  <Choices name="age" toName="text" choice="single" showInLine="true">
    <Choice value="Child (less than 18y)" hotkey="c"/>
    <Choice value="Adult (19-50y)" hotkey="a"/>
    <Choice value="Aged (50+y)" hotkey="o"/>
  </Choices>
  <Header value="Gender"/>
  <Choices name="gender" toName="text" choice="single" showInLine="true">
    <Choice value="Female" hotkey="f"/>
    <Choice value="Male" hotkey="m"/>
  </Choices>
</View>
```
### Attribute train
This version of Annotation Lab restricts the training of two or more Classification Models at the same time. If there are multiple Classification categories in a project (like the one above), only the category whose name comes first in alphabetical order will be trained by default. In the above example, based on the value of the name attribute, we conclude that the Age classifier model is trained.
The model to be trained can also be specified by setting the train="true" attribute for the targeted <Choices> tag (like the one defined in Gender category below).
```bash
<View>
  <View style="overflow: auto;">
    <Text name="text" value="$text"/>
  </View>
  <Header value="Smoking Status"/>
  <Choices name="smokingstatus" toName="text" choice="multiple" showInLine="true">
    ...
  </Choices>
  <Header value="Age"/>
  <Choices name="age" toName="text" choice="single" showInLine="true">
    ...
  </Choices>
  <Header value="Gender"/>
  <Choices name="gender" train="true" toName="text" choice="single" showInLine="true">
   ...
  </Choices>
</View>
```
The trained classification models are also available on the Spark NLP pipeline config list.

<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/classification_pipeline.png" style="width:80%;"/>



## Mixed Projects 

If a project is set up to include Classification, Named Entity Recognition and Assertion Status labels and the three kinds of annotations are present in the training data, it is possible to train three models: one for Named Entity Recognition, one for Assertion Status, and one for Classification at the same time. The training logs from all three trainings can be downloaded at once by clicking the download button present in the Training section of the Setup Page. The newly trained models will be added to the Spark NLP pipeline config.

## Active Learning
Project Owners or Managers can enable the Active Learning feature by clicking on the corresponding Switch (item 6 on the above image) available on Model Training tab. If this feature is enabled, the NER training gets triggered automatically on every 50 new completions. It is also possible to change the completions frequency by dropdown (item 7) which is visible only when Active Learning is enabled.

While enabling this feature, users are asked whether they want to deploy the newly trained model right after the training process or not.

<img class="image image__shadow" src="/assets/images/annotation_lab/2.6.0/deployAL.png" style="width:70%;"/>

If the user chooses not to automatically deploy the newly trained model, this can be done on demand by navigating to the Spark NLP pipeline Config and filtering the model by name of the project (item 3) and select that new model trained by Active Learning. This will update the Labeling Config (name of the model in tag is changed). Hovering on each trained model will show the training date and time.


<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/ner_config.png" style="width:70%;"/>

If the user opts to deploy the model after the training, the Project Configuration is automatically updated for each label that is not associated with a pretrained Spark NLP model, the model information is updated with the name of the new model.

If there is any mistake in the name of models, the validation error is displayed in the Interface Preview Section present on the right side of the Labeling Config area.


<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/config_update.png" style="width:70%;"/>

## Train German and Spanish Models

In earlier versions of the Annotation Lab, users could download German/Spanish pretrained models from the NLP Models Hub and use them for preannotation. From this version, Annotation Lab also offers support for training/tuning German and Spanish models.

## Deploy a new training job

With release 3.0.0, users can perform multiple training jobs at the same time, depending on the available resources/license(s). Users can opt to create new training jobs independently from already running training/preannotation/OCR jobs. If resources/licenses are available when pressing the `Train Now` button a new training server is launched. 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/10557387/161714065-931d2c90-fd46-42bc-b008-bea0e1cdfff3.gif)
