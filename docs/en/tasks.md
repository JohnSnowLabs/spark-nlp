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

The **Tasks** screen shows a list of all documents that have been imported into the current project. 

Under each task you can see meta data about the task: the time of import, the user who imported the task and the annotators and reviewers assigned to the task. 

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/tasks_annotator.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
## Task Status

Each task has one of the following four statuses: 
- **Incomplete**, when none of the assigne annotators has started working on the task. 
- **In Progress**, when at least one of the assigned annotators has submitted at least one completion for this task.
- **Submitted**, when all annotators which were assigned to the task have submitted a completion which is set as ground truth (starred).
- **Reviewed**, in the case there is a reviewer assigned to the task, and the reviewer has reviewed and accepted the submited completion.
- **To Correct**, in the case the assigned reviewer has rejected the completion created by the Annotator. 


As normally annotation projects involve a large number of tasks, the Task page includes filtering and sorting options which will help the user identify the tasks he/she needs faster. 
Tasks cam be sorted by time of import ascending or descending. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/sort.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
Tasks can be filtered by the assigned tags, by the user who imported the task and by the status.


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/tags.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
There is also a search functionality which will identify the tasks having a given string on their name. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/search.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
The number of tasks visible on the screeen is customizable by selecting the predefined values from the Tasks per page drop-down. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/tasks_filter.png" style="width:50%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>



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