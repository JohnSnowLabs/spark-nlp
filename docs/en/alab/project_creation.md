---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Project Creation
permalink: /docs/en/alab/project_creation
key: docs-training
modify_date: "2022-10-19"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## New project 
You can create a new project using a creation wizard which will guide users through each step of the project creation and configuration. Those are illustrated below:

**Project Description**

Every project in Annotation Lab should have the following information:

- a unique name and a short description;
- a team of annotators, reviewers and a manager which will colaborate on the project;
- a configuration which specifies the type of annotations that will be created.

To create a new project, click on the Create Project button on the Home Page and choose a name for it. The project can include a short description and annotation instructions/guidelines.

![Description](https://user-images.githubusercontent.com/46840490/193200891-d2098f36-c300-4693-924d-168801ae5acd.png)

Reserved words cannot be used as project names. The use of keywords like count, permission, or name as project names generated UI glitches. To avoid such issues, these keywords are no longer accepted as project names.

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/reserved.gif" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

**Adding Team Members**

When working in teams, projects can be shared with other team members.

The user who creates a project is called a Project Owner. He/She has complete visibility and ownership of the project for its entire lifecycle. If the Project Owner is removed from the user database, then all his/her projects are transfered to a new project owner. The Project Owner can edit the project configuration, can import/export tasks, can create a project team that will work on his project and can access project analytics.
When defining the project team, a project owner has access to three distinct roles: Annotator, Reviewer, and Manager. These are very useful for most of the workflows that our users follow.

- An Annotator is able to see the tasks which have been assigned to him or her and can create annotations on the documents.
- The Reviewer is able to see the work of the annotators and approve it or reject in case he finds issues that need to be solved.
- The Manager is able to see the work of the Annotators and of the Reviewers and he can assign tasks to team members. This is useful for eliminating work overlap and for a better management of the work load.

To add a user to your project team, navigate to the Project Setup page. On the Manage Project Team tab, start typing the name of a user in the available text box. This will populate a list of available users having the username start with the characters you typed. From the dropdown select the user you want to add to your team. Select a role for the user and click on the “Add to team” button.

In the Project Team page users can add/remove/update the team members even in the case of a large number of members. The team members are displayed in a tabular view. Each member has a priority assigned to them for CONLL export which can be changed by dragging them across the list.
![teamMembers](https://user-images.githubusercontent.com/46840490/193060010-3394cccd-93ab-40a8-8479-98155ea8b417.gif)

**Project Configuration**

The Project Configuration itself is a multi-step process. The wizard will guide users through each step and provide information and hints for each step.
![projectConfiguration](https://user-images.githubusercontent.com/46840490/193033349-534cc2ab-2e5a-4caa-a050-0ee650949b21.gif)

## Clone
You can create a copy of a project, by using the Clone option. The option to clone the project is also listed in the kebab menu of each project. The cloned project is differentiated as it contains cloned suffix in its project name.

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardCloneImportExport.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
  
## Export  
Entire Projects can be exported. The option to export the project is listed in the kebab menu of each project. All project-related items such as tasks, project configuration, project members, task assignments, and comments are included in the export file.

> **NOTE:**  Project export does not contain the model trained in the project as models are independent and not attached to a particular project.
  
## Import  
The project can be imported by uploading the zip in the upload dialog box. When the project is imported back to ALAB, all the item can be seen as it was present when exported. 
  
## Project Grouping
As the number of projects can grow significantly over time, for an easier management and organization of those, the Annotation Lab allows project grouping. As such, it is possible to assign a project to an existing group or to a new group. Each group can be assigned a color which will be used to highlight projects included in that group. Once a project is assigned to a group, the group name proceeds the name of the project. At any time a project can be remove from one group and added to another group.

The list of visible projects can be filtered based on group name, or using the search functionality which applies to both group name and project name. 

<img class="image image--xl" src="/assets/images/annotation_lab/4.1.0/dashboardGroup.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Projects can be organized in custom groups, and each project card will inherit the group color so that the users can visually distinguish the projects easily in a large cluster of projects. The new color picker for the group is user-friendly and customizable.

![DashboardGroupGIF](https://user-images.githubusercontent.com/46840490/193201637-57a7e7b6-9d25-48b4-9196-e6bed61fa2ad.gif)



