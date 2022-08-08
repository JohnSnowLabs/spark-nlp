---
layout: docs
comment: no
header: true
seotitle: Visual NER | John Snow Labs
title: Visual NER
permalink: /docs/en/alab/visual_ner
key: docs-training
modify_date: "2022-04-05"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

Annotating text included in image documents (e.g. scanned documents) is a common use case in many verticals but comes with several challenges. With the new Visual NER Labeling config, we aim to ease the work of annotators by allowing them to simply select text from an image and assign the corresponding label to it.
This feature is powered by Spark OCR 3.5.0; thus a valid Spark OCR license is required to get access to it.

Here is how this can be used:
1.  Upload a valid Spark OCR license. See how to do this [here](https://nlp.johnsnowlabs.com/docs/en/alab/byol).
2.  Create a new project, specify a name for your project, add team members if necessary, and from the list of predefined templates (Default Project Configs) choose “Visual NER Labeling”.
3.  Update the configuration if necessary. This might be useful if you want to use other labels than the currently defined ones. Click the save button. While saving the project, a confirmation dialog is displayed to let you know that the Spark OCR pipeline for Visual NER is being deployed.
4.  Import the tasks you want to annotate (images).
5.  Start annotating text on top of the image by clicking on the text tokens or by drawing bounding boxes on top of chunks or image areas.
6.  Export annotations in your preferred format.

The entire process is illustrated below: 

<img class="image image--xl" src="/assets/images/annotation_lab/2.1.0/invoice_annotation.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Support for multi-page PDF documents

When a valid Saprk OCR license is available, Annotation Lab offers support for multi-page PDF annotation. The complete flow of import, annotation, and export for multi-page PDF files is currently supported.

Users have two options for importing a new PDF file into the Visual NER project
- Import PDF file from local storage;
- Add a link to the PDF file in the file attribute.

<img class="image image--xl" src="/assets/images/annotation_lab/2.3.0/import_pdf.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

After import, the task becomes available on the `Tasks Page`. The title of the new task is the name of the imported file. 

<img class="image image--xl" src="/assets/images/annotation_lab/2.3.0/import_pdf_2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

On the labeling page, the PDF file is displayed with pagination so that annotators can annotate on the PDF document one page at a time.


## OCR and Visual NER servers

Just like (preannotation servers)[], Annotation Lab 3.0.0 also supports the deployment of multiple OCR servers. If a user has uploaded a Spark OCR license, be it airgap or floating, OCR inference is enabled. 

To create a Visual NER project, users have to deploy at least one OCR server. Any OCR server can perform preannotation. To select the OCR server, users have to go to the Import page, toggle the OCR option and from the popup, choose one of the available OCR servers. In no suitable OCR server is available, one can be created by choosing the “Create Server” option.

![ocr_dialog](https://user-images.githubusercontent.com/26042994/161700598-fd2c8887-3bf9-4c71-9cb2-c47fc065a42a.gif)

## Visual NER Training And Preannotation

With v3.4.0 with support for Visual NER Automated Preannotation and Model Training. 

### Visual NER Training support

Version 3.4.0 of the Annotation Lab offers the ability to train Visual NER models, apply active learning for automatic model training, and preannotate image-based tasks with existing models in order to accelerate annotation work.

#### License Requirements

Visual NER annotation, training and preannotation features are dependent on the presence of a Spark OCR license. Floating or airgap licenses with scope ocr: inference and ocr: training are required for preannotation and training respectively.
![licenseVisualNER](https://user-images.githubusercontent.com/33893292/181743592-62b705d5-5730-4225-9541-e1d96d997e7d.png)

### Model Training

The training feature for Visual NER projects can be activated from the Setup page via the “Train Now” button (See 1). From the Training Settings sections, users can tune the training parameters (e.g. Epoch, Batch) and choose the tasks to use for training the Visual NER model (See 3).

Information on the training progress is shown in the top right corner of the Model Training tab (See 2). Users can check detailed information regarding the success or failure of the last training.

Training Failure can occur because of:
* Insufficient number of completions
* Poor quality of completions
* Insufficient CPU and Memory
* Wrong training parameters

![VisualNERTraining](https://user-images.githubusercontent.com/33893292/181743623-c3c62d98-7cda-41a1-9d4f-0951a35b8027.png)

When triggering the training, users can choose to immediately deploy the model or just train it without deploying. If immediate deployment is chosen, then the labeling config is updated with references to the new model so that it will be used for preannotations.

![VisualNERConfig](https://user-images.githubusercontent.com/33893292/181781047-0d1e68ea-a88d-40d2-a557-11b81a459aaa.png)

#### Training Server Specification

The minimal required training configuration is 64 GB RAM, 16 Core CPU for Visual NER Training.

### Visual NER Preannotation

For running preannotation on one or several tasks, the Project Owner or the Manager must select the target tasks and can click on the Preannotate button from the upper right side of the Tasks Page. This will display a popup with information regarding the last deployment including the list of models deployed and the labels they predict.

![VisualNERPreannotationGIF](https://user-images.githubusercontent.com/33893292/181766298-28643f8f-dc6e-4ef6-a426-b454ab0a1db3.gif)

Known Limitations:

* When bulk preannotation is run on a lot of tasks, the preannotation can fail due to memory issues.
* Preannotation currently works at token level, and does not merge all tokens of a chunk into one entity.

#### Preannotation Server Specification

The minimal required training configuration is 32 GB RAM, 2 Core CPU for Visual NER Model.