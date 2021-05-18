---
layout: docs
comment: no
header: true
title: Active Learning  
permalink: /docs/en/active_learning
key: docs-training
modify_date: "2021-05-11"
use_language_switcher: "Python-Scala"
---

A **Project Owner** or a **Manager** can use the completed tasks (completions) from a project for training a new Spark NLP model. The training feature can be found on the Setup page.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/train_setup.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Named Entity Recognition Projects
Named Entity Recognition (NER) projects usually include several labels. When the annotation team has generated a relevant sample of training data/examples for each one of the labels the Project Owner/Manager can use this data to train an DL model which can then be used to predict the labels on new tasks. 

The NER models can be easily trained as illustrated below. 

The "Train Now" button (See Arrow 6) can be used to trigger training of a new model when no other trainings or preannotations are in progress. Otherwise, the button is disabled. Information on the training progress is shown in the top right corner of Model Training tab. Here the user can get indications on the success or failure message depending on how the last training ended.

When triggering the training, users are prompted to choose either to immediately deploy models or just do training. If immediate deployment is chosen, then the Labeling config is updated according to the name of the new model (item 1 on the above image).

It is possible to download training logs by clicking on the download logs icon (see item 9 on the above image) of the recently trained NER model which includes information like training parameters and TF graph used along with precision, recall, f1 score, etc.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/train_ner.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Training parameters
It is possible to tune the most common training parameters (Validation split ratio, Epoch, Learning rate, Decay, Dropout, and Batch) by editing their values on the popup window activated by the gear icon (see item 4 on the above image).

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/training_params.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

It is also possible to train a model by using a sublist of tasks with predefined tags. This is done by specifing the targeted Tags on the Training Parameters popup window (last option).

## Selection of Completions
During the annotation project lifetime, normally not all tasks/completions are ready to be used as a training dataset. This is why the training process selects completions based on their status:
- Filter tasks by tags (if defined in Training Parameters window, otherwise all tasks are considered)
- For completed tasks, completions to be taken into account are also selected based on the following criteria:
  - If a task has a completion accepted by a reviewer this is selected for training and all others are ignored
  - Completions rejected by a Reviewer are not used for training
  - If no reviewer is assigned to a task that has multiple submitted completions the most recent completion is selected for training purpose

## Assertion Status Projects

NER configurations for the healthcare domain are often mixed with Assertion Status labels. In this case Annotation Lab offers support for training both types of models in one go. After the training is complete, the models will be listed in the Spark NLP Pipeline Config. On mouse over the model name in the Spark NLP pipeline config, the user can see more information about the model such as when it was trained and if the training was manually initiated or by the Active Learning process.

Once the model(s) has been trained, the project configuration will be automatically updated to reference the new model for prediction. Notice below, for the Assertion Status **<Label>** tag the addition of model attribute to indicate which model will be used for task preannotation for this label.

```bash
    <Label value="Absent" assertion="true" model="assertion_jsl_annotation_manual.model"/>
    <Label value="Past" assertion="true" model="assertion_jsl_annotation_manual.model"/>
```

It is not possible to mark a label as an Assertion Status label and use a NER model to predict it. A validation error is shown in the Interface Preview in case an invalid Assertion model is used.
<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/as_notification.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The Annotation Lab only allows the use of one single Assertion Status model in the same project.
<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/one_as.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


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

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/classification_pipeline.png" style="width:80%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>



## Mixed Projects 

If a project is set up to include Classification, Named Entity Recognition and Assertion Status labels and the three kinds of annotations are present in the training data, it is possible to train three models: one for Named Entity Recognition, one for Assertion Status, and one for Classification at the same time. The training logs from all three trainings can be downloaded at once by clicking the download button present in the Training section of the Setup Page. The newly trained models will be added to the Spark NLP pipeline config.

## Active Learning
Project Owners or Managers can enable the Active Learning feature by clicking on the corresponding Switch (item 7 on the above image) available on Model Training tab. If this feature is enabled, the NER training gets triggered automatically on every 50 new completions. It is also possible to change the completions frequency by dropdown (8) which is visible only when Active Learning is enabled.

While enabling this feature, users are asked whether they want to deploy the newly trained model right after the training process or not.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/deployAL.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

If the user chooses not to automatically deploy the newly trained model, this can be done on demand by navigating to the Spark NLP pipeline config and filtering the model by name of the project (3) and select that new model trained by Active Learning. This will update the Labeling Config (name of the model in tag is changed). Hovering on each trained model will show the training date and time.


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/ner_config.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

If the user opts for deploying the model after the training, the Project Configuration is automatically updated for each label that is not associated with a pretrained Spark NLP model, the model information is updated with the name of the new model.

If there is any mistake in the name of models, the validation error is displayed in the Interface Preview Section present on the right side of the Labeling Config area.


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/config_update.png" style="width:70%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>