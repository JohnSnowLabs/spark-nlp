---
layout: docs
comment: no
header: true
title: Tasks
permalink: /docs/en/tasks
key: docs-training
modify_date: "2020-11-19"
use_language_switcher: "Python-Scala"
---

The **Tasks** screen shows a list of all documents that have been imported into the current project. Each task has one of the following two statuses: **completed**(green) or **incomplete**(red). 


<img class="image image--xl" src="/assets/images/annotation_lab/tasks.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


The following information is available for each task:
- creation date and user who added the task;
- date when the last completion was created for the task;
- the tags which were associated with the task;
- the initials of the users that created completions for this task.


## Preannotations

For projects that use labels from pre-trained Spark NLP models, the Annotation Lab offers the option to bootstrap the annotation project by automatically running the referred models and adding their results as **predictions** on the tasks. 


This can be done using the **Preannotate** button on the upper right side of the **Tasks** screen. The status of the pre-annotation process can be checked either by pushing the **Preannotate** button again or by clicking on the **Preannotation** drop-down as shown below. 


<img class="image image--xl" src="/assets/images/annotation_lab/preannotations.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>