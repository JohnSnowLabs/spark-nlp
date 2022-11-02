---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: User Management
permalink: /docs/en/alab/user_management
key: docs-training
modify_date: "2021-10-12"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

Basic user management features are present in the Annotation Lab. The user with the admin privilege can add or remove other users from the system or can edit user information if necessary. This feature is available by selecting the _Users_ option under the _Settings_ menu from the navigation panel.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/user_management.png" style="width:25;"/>

All user accounts created on the Annotation Lab can be seen on the Users page. The table shows the username, first name, last name, and email address of all created user accounts. A user with the admin privilege can edit or delete that information, add a user to a group or change the user’s password.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/users.png" style="width:110%;"/>

## User Details

Annotation Lab stores basic information for each user. Such as the _First Name_, _Last Name_, and _Email_. It is editable from the _Details_ section by any user with admin privilege.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/user_details.png" style="width:110%;"/>

## User Groups

Currently, two user groups are available: _Annotators_ and _UserAdmins_. By default, a new user gets added to the _Annotators_ group. It means the user will not have access to any admin features, such as user management or other settings.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/user_group.png" style="width:110%;"/>

To add a user to the admin group, a user with admin privilege needs to navigate to the _Users_ page, click on the concerned username or select the _Edit_ option from the _More Actions_ icon, then go to the _Group_ section and check the _UserAdmins_ checkbox.

## Reset User Credentials

A user with the admin privilege can change the login credentials for another user by navigating to the _Credentials_ section of the edit user page and defining a new (temporary) password. For extra protection, the user with the admin privilege can enforce the password change on the next login.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/user_credentials.png" style="width:110%;"/>
