---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Active Learning  
permalink: /docs/en/alab/active_learning
key: docs-training
modify_date: "2022-10-21"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

Project Owners or Managers can enable the Active Learning feature by clicking on the corresponding Switch (item 6 on the above image) available on Model Training tab. If this feature is enabled, the NER training gets triggered automatically on every 50 new completions. It is also possible to change the completions frequency by dropdown (item 7) which is visible only when Active Learning is enabled.

While enabling this feature, users are asked whether they want to deploy the newly trained model right after the training process or not.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/activeLearning.png" style="width:100%;"/>

If the user chooses not to automatically deploy the newly trained model, this can be done on demand by navigating to the Spark NLP pipeline Config and filtering the model by name of the project (item 3) and select that new model trained by Active Learning. This will update the Labeling Config (name of the model in tag is changed). Training date and time of each trained model is displayed in the predefined labels.


<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/modelinPredefined.png" style="width:100%;"/>

If the user opts to deploy the model after the training, the Project Configuration is automatically updated for each label that is not associated with a pretrained Spark NLP model, the model information is updated with the name of the new model.

If there is any mistake in the name of models, the validation error is displayed in the Interface Preview Section present on the right side of the Labeling Config area.


<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/config_update.png" style="width:100%;"/>
