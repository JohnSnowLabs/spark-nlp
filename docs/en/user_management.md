---
layout: docs
comment: no
header: true
title: User Management
permalink: /docs/en/user_management
key: docs-training
modify_date: "2021-05-10"
use_language_switcher: "Python-Scala"
---

The Annotation Lab offers user management features. The admin user can add or remove a user from the data base or can edit user information if necessary. This feature is available by navigating to the lower left side menu and selecting User Management feature. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/user_management.png" style="width:25%; align:left; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

All users that have beed added to the current Annotation Lab instance can be seen on the Users screen. First Name, Last Name and e-mail address information should be available for all users. An admin user can edit those information, add a user to a group or change a user's password. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/users.png" style="width:110%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## User Details
For each user, the Annotation Lab stores basic information such as: the First Name, Last Name, e-mail address. Those can be edited from the User Details page by any Admin User. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/user_details.png" style="width:110%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## User Groups
Currently the Annotation Lab defines two user groups: Annotators and UserAdmins. By default a new user is added to the Annotators group. This means the user will not have access to any admin features such as: User Management or Settings. 


<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/user_group.png" style="width:110%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


For adding a user to the Admin group, a admin user needs to navigate to the Users screen, click on the edit button for the concerned user, then click on the Groups tab and check the Admin checkbox.  


## Reset User Credentials 

An admin user can change the login credentials for another user, by navigating to the User Credentials tab and by defining a new (temporary) password. For extra protection, the admin user can also enforce the password change on next user login. 
<img class="image image--xl" src="/assets/images/annotation_lab/1.6.0/user_credentials.png" style="width:110%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>