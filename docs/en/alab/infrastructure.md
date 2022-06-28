---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Infrastructure
permalink: /docs/en/alab/infrastructure
key: docs-training
modify_date: "2022-04-05"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## Infrastructure configuration

 Two new tabs have been added to the Settings page to ease the infrastructure definition for the prediction and training tasks and for defining backup schedules. 

### Resource allocation for Training and Preannotation 

Since version 2.8.0, Annotation Lab gives users the ability to change the configuration for the training and preannotation processes. This is done from the `Settings page` > `Infrastructure tab`. The settings can be edited by admin users and they are read-only for the other users. The Infrastructure tab consists of three sections named `Training Resources`, `Prenotation Server Resources`, `Prenotation Pipeline Resources`.

**Resources Inclusion:**
1.  Memory Limit – Represents the maximum memory size to allocate for the training/preannotation processes.
2.  CPU Limit – Specifies this maximum number of CPUs to use by the training/preannotation server.
3.  Spark Drive Memory – Defines the memory allocated for the Spark driver.
4.  Spark Kry Buff Max – Specifies the maximum memory size to allocate for the Kryo serialization buffer.
5.  Spark Driver Maximum Result Size – Represents the total size of the serialized results of all the partitions for spark.

 ![image](https://user-images.githubusercontent.com/73094423/158771801-e6155f07-4aaa-4a2d-8683-b5d42d8509a1.png)

**NOTE:-** If the specified configurations exceed the available resources, the server will not start.
 
### Backup settings in UI
In 2.8.0 release, Annotation Lab added support for defining database and files backups via the UI. Any user with the admin role can view and edit the backup settings under the Settings tab. Users can select different backup periods and can specify a target S3 bucket for storing the backup files. New backups will be automatically generated and saved to the S3 bucket following the defined schedule. 
 
 ![image](https://user-images.githubusercontent.com/73094423/158530658-89adf6c2-70c0-489a-868d-5b17e5c71ee7.png)

### Management of Preannotation and Training Servers  

Annotationlab 3.0.0 gives users the ability to view the list of all active servers. Any user can access the Server List page by navigating to the Settings page > Server tab. This page gives the following details.

- A summary of the status/limitations of the current infrastructure to run Spark NLP for Healthcare training jobs and/or preannotation servers.
- Ability to delete a server and free up resources when required, so that another training job and/or preannotation server can be started.
- Shows details of the server
   - Server Name: Gives the name of the server that can help identify it while running preannotation or importing files.
   - License Details: The license that is being used in the server and its scope.
   - Usage: Let the user know the usage of the server. A server can be used for preannotation, training or OCR.
   - Deployed by: The user who deployed the server. This information might be useful for contacting the user who deployed a server before deleting it.
   - Deployed at: The deployed time of the server.
       
![server_page](https://user-images.githubusercontent.com/26042994/161711565-798bf34d-92ba-41c8-b0b8-2778a3e61561.gif)

#### Statuses of Training and Preannotation Server
A new column, status, is added to the server page that gives the status of training and preannotation servers. The available preannotation server statuses are:
* green=idle
* yellow=busy
* red=stopped

Users can visualize which servers is busy and which are idle. This is very useful information when the user intends to deploy a new server in replacement of an idle one. In this situation, the user can delete an idle server and deploy another preannotation/ training server.
This information is also available on the preannotation popup when the user selects the deployed server to use for preannotation.
<img class="image image--xl" src="/assets/images/annotation_lab/serverStatus1.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Also, if any issues are encountered during server initialization, those are displayed on mouse-over the status value. Depending on the issue, changes might be required in the infrastructure settings and user will have to manually redeploy the training/preannotation server.
<img class="image image--xl" src="/assets/images/annotation_lab/serverStatus2.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
