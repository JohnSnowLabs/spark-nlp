---
layout: docs
comment: no
header: true
title: Preannotations with Spark NLP  
permalink: /docs/en/alab/preannotations
key: docs-training
modify_date: "2021-05-11"
use_language_switcher: "Python-Scala"
sidebar:
    nav: annotation-lab
---


**Annotation Lab** offers out-of-the-box support for **Named Entity Recognition, Classification and Assertion Status Preannotations**. Those are extremely useful for bootstraping any annotation project, as the annotation team does not start the labeling from scratch but can leverage the existing knowledge transfer from domain experts to models. This way, the annotation efforts are significantly reduced.

For running preannotation on one or several tasks, the **Project Owner** or the **Manager** must select the target tasks and can click on the Preannotate Button from the upper right side of the Tasks List Page. This will display a popup with information regarding the last deployment including the list of models deployed and the labels they predict.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/preannotate_ner.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

This information is very important, especially when multiple users are doing training and deployment in parallel. So before doing preannotations on your tasks, carefully check the list of currently deployed models and their labels.

If needed, users can deploy the models defined in the current project (based on the current Labeling Config) by clicking the "Deploy" button. After the deployment is complete, the preannotation can be triggered.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/preannotate.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Pretrained Models 

On the project setup screen you can find a Spark NLP pipeline config widget which lists all available models together with the labels those are predicting. By simply selecting the relevant labels for your project and clicking the add button you can add the predefined labels to your project and take advantage of the Spark NLP auto labeling capabilities. 


In the below example we are reusing the posology model that comes with 7 labels related to drugs.  
<img class="image image--xl" src="/assets/images/annotation_lab/spark_nlp_models.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## Pipeline Limitations

Loading too many models in the preannotation server is not memory efficient and may not be practically required. Starting from version 1.8.0, Annotation Lab supports maximum of five different models to be used for the preannotation server deployment.
Another restriction for the server deployment is that two models trained on different embeddings cannot be used together in the same project.
The Labeling Config will throw validation errors in any of the cases above and hence Labeling Config cannot be saved and the preannotation server deployment will fail.

<img class="image image--xl" src="/assets/images/annotation_lab/1.8.0/5_models.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
