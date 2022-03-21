---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Infrastructure
permalink: /docs/en/alab/infrastructure
key: docs-training
modify_date: "2022-03-21"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## Infrastructure configuration

 Two new tabs have been added to the Settings page to ease the infrastructure definition for the prediction and training tasks and for defining backup schedules. 

### Resource allocation for Training and Preannotation 

This release of Annotationlab gives users the ability to change the configuration for the training and preannotation processes. This is done from the `Settings page` > `Infrastructure tab`. The settings can be edited by admin users and they are read-only for the other users. The Infrastructure tab consists of three sections named `Training Resources`, `Prenotation Server Resources`, `Prenotation Pipeline Resources`.

**Resources Inclusion:**
1.  Memory Limit – Represents the maximum memory size to allocate for the training/preannotation processes.
2.  CPU Limit – Specifies this maximum number of CPUs to use by the training/preannotation server.
3.  Spark Drive Memory – Defines the memory allocated for the Spark driver.
4.  Spark Kry Buff Max – Specifies the maximum memory size to allocate for the Kryo serialization buffer.
5.  Spark Driver Maximum Result Size – Represents the total size of the serialized results of all the partitions for spark.

 ![image](https://user-images.githubusercontent.com/73094423/158771801-e6155f07-4aaa-4a2d-8683-b5d42d8509a1.png)

**NOTE:-** If the specified configurations exceed the available resources, the server will not start.
 
### Backup settings in UI
In this release, AnnotationLab adds support for defining database and files backups via the UI. Any user with the admin role can view and edit the backup settings under the Settings tab. Users can select different backup periods and can specify a target S3 bucket for storing the backup files. New backups will be automatically generated and saved to the S3 bucket following the defined schedule. 
 
 ![image](https://user-images.githubusercontent.com/73094423/158530658-89adf6c2-70c0-489a-868d-5b17e5c71ee7.png)