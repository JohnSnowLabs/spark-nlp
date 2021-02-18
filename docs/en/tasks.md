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

## Task Assignment 

Annotation Lab 1.2.0 release added some important new features such as Task Assignment. With this feature, project owners can assign tasks to annotator(s) in order to better plan/distribute project work. Annotators can only view tasks that are assigned to them which means there is no chance of accidental work overlap.
 
For assigning a task to an annotator, from the task page select one or more tasks and from the Assign dropdown choose an  annotator. 
You can only assign a task to annotators that have already been added to the project. For adding an annotator to the project, go to the Setup page and share the project with the annotator by giving him/her the update right.


<img class="image image--xl" src="/assets/images/annotation_lab/assign.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Once an annotator is assigned to a task his/her name will be listed below the task name on the tasks screen. 

<img class="image image--xl" src="/assets/images/annotation_lab/assigned.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


When upgrading from an older version of the Annotation Lab, the annotators will no longer have access to the tasks they worked on unless they will be assigned to those explicitely by the admin user who created the project. Once they are assigned, they can resume work and no information will be lost.  