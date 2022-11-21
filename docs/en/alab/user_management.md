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

## SAML Integration

AnnotationLab supports Security Assertion Markup Language (SAML). To login to AnnotationLab using SAML, follow the steps below.

### SAML Server Setup

Run the following command to setup a sample SAML server in a Docker environment: 

```
docker run --rm --name mysamlserver -p 8081:8080 -p 8443:8443 -e SIMPLESAMLPHP_SP_ENTITY_ID=http://{IP}/auth/realms/master -e SIMPLESAMLPHP_SP_ASSERTION_CONSUMER_SERVICE=http://{IP}/auth/realms/master/broker/saml/endpoint --network annotationlab kristophjunge/test-saml-idp
```

### SAML Configuration

Follow the steps described below to setup a SAML connection. 

 1) Goto AnnotationLab Keyclock console and navigate to `Identity Providers` under `Configure` on the left-side menu. 

 2) Choose `SAML v2.0` from Add Provider drop-down menu and a configuration page should appear.

![Screen Shot 2022-02-16 at 11 52 23 AM](https://user-images.githubusercontent.com/17021686/154219230-726c3787-ce1e-4902-a90e-0228859a71b6.png)

 3) Provide values for `Alias`(e.g: saml) and `Display Name`(e.g: SAML). The value for `Display Name` will be seen in the login page.

 4) Now, set the value of the following attributes as shown below:
   - Enabled: On
   - Store Tokens: On
   - First Login Flow : first broker login
   - Sync Mode: force

 5) Under SAML Config specify values for the following parameters as provided by SAML sever:
   - Service Provider Entity ID 
   - Single Sign-On Service URL
   - Single Logout Service URL

 6) Choose a `Principal Type`(e.g: Attribute[Name]) and add value to `Principal Attribute`(e.g. email) according to the data provided by SAML server

 7) Click on the `Save` button to save the changes.



### Identity Provider Mapper

An Identity Provider Mapper must be defined for importing SAML data provided by the External Identity Provider (IDP) and using it for authenticating into Annotation Lab. This allows user profile and other user information to be imported and made available into Annotation Lab.

On Identity Providers > SAML page click on the `Mappers` tab located next to the `Settings` tab and follow the steps  below:

 1. Click on `Create`. This should open a form to add a new `Identity Provider Mapper`
 2. Set the value for the following attributes:
   - Name(e.g: uma_protection mapper)
   - Sync Mode Override: inherit
   - Mapper Type: Hardcoded Role
 3. Click on the `Select Role` button and under the `Client Roles` menu put `annotationlab`. Now, select `uma_protection` and click on `Select client role`. `annotationlab.uma_protection` should be the value displayed for `Role` 
 4. Save the changes

### Default Group

Default groups are used for assigning group membership automatically whenever any new user is created. Add `Annotators` as the default group using the following steps:

1. Goto `Groups`, on the left side panel under `Manages`
2. Select the `Default Groups` tab
3. Under `Available Groups` select `Annotators` and then click on the `Add` button

Now, `Annotators` should be listed under Default Groups.  

![Screen Shot 2022-02-16 at 12 30 23 PM](https://user-images.githubusercontent.com/17021686/154219740-75a1214e-cc5f-452f-8f0b-5f8e71871c12.png)


### Login to Annotation Lab

Goto the Annotation Lab's login dashboard and click on the display name which was set earlier(e.g: SAML). This is displayed under `Or sign in with`.  

![Screen Shot 2022-02-16 at 11 59 49 AM](https://user-images.githubusercontent.com/17021686/154219826-0615f052-0a81-45ff-a856-57386fc4007c.png)

Login with the data provided by the SAML server here:  

![Screen Shot 2022-02-16 at 10 50 02 AM](https://user-images.githubusercontent.com/17021686/154219900-54c3f829-0041-4c33-88e1-e71c4231e8f9.png)


The user account information is updated and the user is redirected to Annotation Lab and presented with the `Project` dashboard.

> NOTES: Users added as an IDP will be available in the `Users` tab on the left side under `Manages`

