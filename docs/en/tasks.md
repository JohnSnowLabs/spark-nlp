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


## Task Assignment 

Project Owners/Managers can assign tasks to annotator(s) and reviewer(s) in order to better plan/distribute project work. Annotators and Reviewers can only view tasks that are assigned to them which means there is no chance of accidental work overlap.
 
For assigning a task to an annotator, from the task page select one or more tasks and from the Assign dropdown choose an annotator. 
You can only assign a task to annotators that have already been added to the project. For adding an annotator to the project, go to the Setup page and share the project with the annotator by giving him/her the update right.

Once an annotator is assigned to a task his/her name will be listed below the task name on the tasks screen. 

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/task_assignment.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


When upgrading from an older version of the Annotation Lab, the annotators will no longer have access to the tasks they worked on unless they will be assigned to those explicitely by the admin user who created the project. Once they are assigned, they can resume work and no information will be lost.  

## Task Status

At high level, each task can have one of the following statuses: 
- **Incomplete**, when none of the assigne annotators has started working on the task. 
- **In Progress**, when at least one of the assigned annotators has submitted at least one completion for this task.
- **Submitted**, when all annotators which were assigned to the task have submitted a completion which is set as ground truth (starred).
- **Reviewed**, in the case there is a reviewer assigned to the task, and the reviewer has reviewed and accepted the submited completion.
- **To Correct**, in the case the assigned reviewer has rejected the completion created by the Annotator. 


The status of a task varies according to the type of account the logged in user has (his/her visibility over the project) and according to the tasks that have been assigned to him/her. 

### For Project Owner, Manager and Reviewer
On the Analytics page and Tasks page, the Project Owner/Manager/Reviewer will see the general overview of the projects which will take into consideration the task level statuses as follows:
- **Incomplete** - Annotators have not started working on this task
- **In Progress** - At least one annotator still has not starred (marked as ground truth) any submitted completions
- **Submitted** - All annotators that are assigned to a task have starred (marked as ground truth) one submitted completion
- **Reviewed** - Reviewer has approved all starred submitted completions for the task


### For Annotators 
On the Annotator's Task page, the task status will be shown with regards to the context of the logged-in Annotator's work. As such, if the same task is assigned to two annotators then:
- if annotator1 is still working and not submitted the task, then he/she will see task status as In-progress

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/ann1.png" style="width:90%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

- if annotator2 submits the task from his/her side then he/she will see task status as Submitted

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/ann2.png" style="width:90%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

The following statuses are available on the Annotator's view.
- **Incomplete** – Current logged-in annotator has not started working on this task.
- **In Progress** - At least one saved/submitted completions exist, but there is no starred submitted completion.
- **Submitted** - Annotator has at least one starred submitted completion.
- **Reviewed** - Reviewer has approved the starred submitted completion for the task.
- **To Correct** - Reviewer has rejected the submitted work. In this case, the star is removed from the reviewed completion. The annotator should start working on the task and resubmit.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/reject_completion.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Note:
 - The status of a task is maintained/available only for the annotators assigned to the task.
 - The Project Owner completions state are not considered while deciding the status of a task.
When multiple Annotators are assigned to a task, the reviewer will see the task as submitted when all annotators submit and star their completions. Otherwise, if one of the assigned Annotators has not submitted or has not starred one completion, then the Reviewer will see the task as In Progress.

## Task filters

As normally annotation projects involve a large number of tasks, the Task page includes filtering and sorting options which will help the user identify the tasks he/she needs faster. 
Tasks cam be sorted by time of import ascending or descending. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/sort.png" style="width:60%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Tasks can be filtered by the assigned tags, by the user who imported the task and by the status.


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/tags.png" style="width:80%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

There is also a search functionality which will identify the tasks having a given string on their name. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/search.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
The number of tasks visible on the screeen is customizable by selecting the predefined values from the Tasks per page drop-down. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/task_filters.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Comments
Comment can be added to each task by Project Owner or Manager. This is done by clicking the comment icon present on the rightmost side of each Task in the Tasks List page. It is important to notice that these comments are visible to everyone who can view the particular task.

<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/comments.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

*Dark icon = A comment is present on the task.
Light icon = No comment is present