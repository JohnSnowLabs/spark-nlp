---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Training Configuratations  
permalink: /docs/en/alab/training_configurations
key: docs-training
modify_date: "2022-10-21"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

A **Project Owner** or a **Manager** can use the completed tasks (completions) from a project to train a new Spark NLP model. The training feature can be found on the train page. The Train page is a part of the Project Menu. It guide users on each step. Users can follow a step-wise wizard view or a synthesis view for initiating the training of a model. During the training, a progress bar is shown to give users basic information on the status of the training process.

![trainingProcessGIF](https://user-images.githubusercontent.com/45035063/193196897-fc20b3c6-920b-46cf-91d4-1b4c70dbf28b.gif)

## Deploy a new training job
Users can perform multiple training jobs at the same time, depending on the available resources/license(s). Users can opt to create new training jobs independently from already running training/pre-annotation/OCR jobs. If resources/licenses are available when pressing the `Train Model` button a new training server is launched.

## Named Entity Recognition Projects
Named Entity Recognition (NER) projects usually include several labels. When the annotation team has generated a relevant sample of training data/examples for each one of the labels the Project Owner/Manager can use this data to train an DL model which can then be used to predict the labels on new tasks. 

The NER models can be easily trained as illustrated below. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/train_setup_label.png" style="width:80%;"/>

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/train_setup_pipeline.png" style="width:80%;"/>

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/train_setup_model.png" style="width:100%;"/>

The "Train Now" button (item 5) can be used to trigger training of a new model. Information on the training progress is shown in the page. Here the user can get indications on the success or failure message depending on how the last training ended.

When triggering the training, users are prompted to choose either to immediately deploy models or just do training. If immediate deployment is chosen, then the Labeling config is updated according to the name of the new model (item 1 on the above image).

It is possible to download training logs by clicking on the download logs icon (see item 8 on the above image) of the recently trained NER model which includes information like training parameters and TF graph used along with precision, recall, f1 score, etc.

## Training parameters

In Annotation Lab, for mixed projects containing multiple types of annotations in a single project like classifications, NER, and assertion status, multiple trainings were triggered at the same time using the same system resources and Spark NLP resources. In this case, the training component could fail because of resource limitations.

In order to improve the usability of the system, dropdown options can be used to choose which type of training to run next. The project Owner or Manager of a project can scroll down to Training Settings and choose the training type. The drop-down gives a list of possible training types for that particular project based on defined Labeling Config. Another drop-down also lists available embeddings which can be used for training the model.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/trainingparameters.png" style="width:80%;"/>

It is possible to tune the most common training parameters (Validation split ratio, Epoch, Learning rate, Decay, Dropout, and Batch) by editing their values in Training Parameters.

It is also possible to train a model by using a sublist of tasks with predefined tags. This is done by specifying the targeted Tags on the Training Parameters (last option).

The Annotation Lab also includes additional filtering options for the training dataset based on the status of completions, either all submitted completions cab be used for training or only the reviewed ones.

## Custom Training Script
If users want to change the default Training script present within the Annotation Lab, they can upload their own training pipeline. In the Train Page, project owners can upload the training scripts. At the moment we are supporting custom training script just for NER projects.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/customScript.png" style="width:80%;"/>

## Selection of Completions
During the annotation project lifetime, normally not all tasks/completions are ready to be used as a training dataset. This is why the training process selects completions based on their status:
- Filter tasks by tags (if defined in Training Parameters window, otherwise all tasks are considered)
- For completed tasks, completions to be taken into account are also selected based on the following criteria:
  - If a task has a completion accepted by a reviewer this is selected for training and all others are ignored
  - Completions rejected by a Reviewer are not used for training
  - If no reviewer is assigned to a task that has multiple submitted completions the most recent completion is selected for training purpose

## Assertion Status Projects

NER configurations for the healthcare domain are often mixed with Assertion Status labels. In this case Annotation Lab offers support for training both types of models in one go. After the training is complete, the models will be listed in the Spark NLP Pipeline Config. Hovering mouse over the model name in the Spark NLP pipeline Config, the user can see more information about the model such as when it was trained and if the training was manually initiated or by the Active Learning process.

Once the model(s) has been trained, the project configuration will be automatically updated to reference the new model for prediction. Notice below, for the Assertion Status **<Label>** tag the addition of model attribute to indicate which model will be used for task pre-annotation for this label.

```bash
    <Label value="Absent" assertion="true" model="assertion_jsl_annotation_manual.model"/>
    <Label value="Past" assertion="true" model="assertion_jsl_annotation_manual.model"/>
```

It is not possible to mark a label as an Assertion Status label and use a NER model to predict it. A validation error is shown in the Interface Preview in case an invalid Assertion model is used.
<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/as_notification.png" style="width:70%;"/>

The Annotation Lab only allows the use of one single Assertion Status model in the same project.
<img class="image image__shadow" src="/assets/images/annotation_lab/1.6.0/one_as.png" style="width:70%;"/>


## Classification Project Models Training
Annotation Lab supports two types of classification training **Single Choice Classification** and **Multi-Choice Classification**. For doing so, it uses three important attributes of the **<Choices>** tag to drive the Classification Models training and pre-annotation. Those are **name**, **choice** and **train**.

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
Annotation Lab restricts the training of two or more Classification Models at the same time. If there are multiple Classification categories in a project (like the one above), only the category whose name comes first in alphabetical order will be trained by default. In the above example, based on the value of the name attribute, we conclude that the Age classifier model is trained.
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

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/classification_pipeline.png" style="width:80%;"/>


## Mixed Projects 

If a project is set up to include Classification, Named Entity Recognition and Assertion Status labels and the three kinds of annotations are present in the training data, it is possible to train three models: one for Named Entity Recognition, one for Assertion Status, and one for Classification at the same time. The training logs from all three trainings can be downloaded at once by clicking the download button present in the Training section of the Setup Page. The newly trained models will be added to the Spark NLP pipeline config.


## Train German and Spanish Models

In earlier versions of the Annotation Lab, users could download German/Spanish pretrained models from the NLP Models Hub and use them for pre-annotation. From this version, Annotation Lab also offers support for training/tuning German and Spanish models.




