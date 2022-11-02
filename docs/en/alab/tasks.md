---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Tasks
permalink: /docs/en/alab/tasks
key: docs-training
modify_date: "2022-11-2"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

The **Tasks** screen shows a list of all documents that have been imported into the current project. 

Under each task you can see meta data about the task: the time of import, the user who imported the task and the annotators and reviewers assigned to the task. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/tasks.png" style="width:100%;"/>


## Task Assignment 

Project Owners/Managers can assign tasks to annotator(s) and reviewer(s) in order to better plan/distribute project work. Annotators and Reviewers can only view tasks that are assigned to them which means there is no chance of accidental work overlap.
 
For assigning a task to an annotator, from the task page select one or more tasks and from the `Assign` dropdown choose an annotator. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_assignment.gif" style="width:100%;"/>

You can only assign a task to annotators that have already been added to the project team. For adding an annotator to the project team, select your project and navigate to the `Setup>Team` menu item. On the `Add Team Member` page, search for the user you want to add, select the role you want to assign to him/her and click on `Add To Team` button. 

> **NOTE:** When upgrading from an older version of the Annotation Lab, the annotators will no longer have access to the tasks they worked on unless they are assigned to those explicitely by the admin user who created the project. Once they are assigned, they can resume work and no information is lost.

## Task Status

At high level, each task can have one of the following statuses: 
- **Incomplete**, when none of the assigned annotators has started working on the task. 
- **In Progress**, when at least one of the assigned annotators has submitted at least one completion for this task.
- **Submitted**, when all annotators which were assigned to the task have submitted a completion which is set as ground truth (starred).
- **Reviewed**, in the case there is a reviewer assigned to the task, and the reviewer has reviewed and accepted the submited completion.
- **To Correct**, in the case the assigned reviewer has rejected the completion created by the Annotator. 


The status of a task varies according to the type of account the logged in user has (his/her visibility over the project) and according to the tasks that have been assigned to him/her. 

### For Project Owner, Manager and Reviewer
On the Analytics page and Tasks page, the Project Owner/Manager/Reviewer will see the general overview of the projects which will take into consideration the task level statuses as follows:
- **Incomplete** - Assigned annotators have not started working on this task
- **In Progress** - At least one annotator still has not starred (marked as ground truth) one submitted completion
- **Submitted** - All annotators that are assigned to the task have starred (marked as ground truth) one submitted completion
- **Reviewed** - Reviewer has approved all starred submitted completions for the task


### For Annotators 
On the Annotator's Task page, the task status will be shown with regards to the context of the logged-in Annotator's work. As such, if the same task is assigned to two annotators then:
- if annotator1 is still working and not submitted the task, then he/she will see task status as *In-progress*

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_in_progress.png" style="width:100%;"/>

- if annotator2 submits the task from his/her side then he/she will see task status as *Submitted*

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_submitted.png" style="width:100%;"/>

The following statuses are available on the Annotator's view.
- **Incomplete** – Current logged-in annotator has not started working on this task.
- **In Progress** - At least one saved/submitted completions exist, but there is no starred submitted completion.
- **Submitted** - Annotator has at least one starred submitted completion.
- **Reviewed** - Reviewer has approved the starred submitted completion for the task.
- **To Correct** - Reviewer has rejected the submitted work. In this case, the star is removed from the reviewed completion. The annotator should start working on the task and resubmit.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_reject_completion.gif" style="width:100%;"/>

> **Note:** The status of a task is maintained/available only for the annotators assigned to the task.
 
When multiple Annotators are assigned to a task, the reviewer will see the task as submitted when all annotators submit and star their completions. Otherwise, if one of the assigned Annotators has not submitted or has not starred one completion, then the Reviewer will see the task as *In Progress*.

## Task Filters

As normally annotation projects involve a large number of tasks, the Task page includes filtering and sorting options which will help the user identify the tasks he/she needs faster. 
Tasks can be sorted by time of import ascending or descending. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_sort.png" style="width:100%;"/>

Tasks can be filtered by the assigned tags, by the user who imported the task and by the status.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_tag.png" style="width:100%;"/>

There is also a search functionality which will identify the tasks having a given string on their name. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search.png" style="width:100%;"/>

The number of tasks visible on the screeen is customizable by selecting the predefined values from the Tasks per page drop-down. 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_filters.gif" style="width:100%;"/>

## Task Search by Text, Label and Choice

Annotation Lab offers advanced search features that help users identify the tasks they need based on the text or based on the annotations defined so far. Currently supported search queries are:
- text: patient -> returns all tasks which contain the string "patient";
- label: ABC -> returns all tasks that have at least one completion containing a chunk with label ABC;
- label: ABC=DEF -> returns all tasks that have at least one completion containing the text DEF labeled as ABC;
- choice: Sport -> returns all tasks that have at least one completion which classified the task as Sport;
- choice: Sport,Politics -> returns all tasks that have at least one completion containing multiple choices Sport and Politics.

Search functionality is case insensitive, thus the following queries `label: ABC=DEF` , `label: Abc=Def` or `label: abc=def` are considered equivalent.

**Example:**

Consider a project with 3 tasks which are annotated as below:

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_task1.png" style="width:100%;"/>
<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_task2.png" style="width:100%;"/>
<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_task3.png" style="width:100%;"/>

Search-query "label:LOC" will list as results Task 1 and Task 3.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_loc.png" style="width:100%;"/>

Search-query "label:WORK_OF_ART" will list as result Task 1 and Task 2.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_work_of_art.png" style="width:100%;"/>

Search-query "label:PERSON=Leonardo" will list as result Task 1.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_search_person_leonardo.png" style="width:100%;"/>

## Comments

Comments can be added to each task by any team member. This is done by clicking the `_View comments_` link present on the rightmost side of each Task in the Tasks List page. It is important to notice that these comments are visible to everyone who can view the particular task.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/task_comment.png" style="width:100%;"/>