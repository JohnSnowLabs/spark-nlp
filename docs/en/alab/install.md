---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Installation
permalink: /docs/en/alab/install
key: docs-training
modify_date: "2021-09-29"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## Deploy on a dedicated server

Install Annotation Lab on a dedicated server to reduce the likelihood of conflicts or unexpected behavior.

### Fresh install

To install Annotation Lab run the following command:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s VERSION
```

For installing the latest available version of the Annotation Lab use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s --
```

To upgrade your Annotation Lab installation to a newer version, run the following command on a terminal:

### Upgrade 

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s VERSION
```
Replace VERSION within the above one liners with the version you want to install.  

For upgrading to the latest version of the Annotation Lab, use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s --
```

> **NOTE:**  The install/upgrade script displays the login credentials for the admin user on the terminal.


After running the install/upgrade script the Annotation Lab is available at http://INSTANCE_IP  or https://INSTANCE_IP 

<img class="image image--xl image__shadow" src="/assets/images/annotation_lab/4.1.0/loginScreenALAB.png" style="width:100%;"/>

We have an aesthetically pleasing Login Page with a section highlighting the key features of Annotation Lab. Sign-In page highlights main features of Annotation Lab with animated GIFs.

## Deploy on the AWS Marketplace

Visit the [product page on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-nsww5rdpvou4w?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) and follow the instructions on the video below to subscribe and deploy. 

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='ebaewU4BcQA' -%}<div class="video-descr">Deploy Annotation Lab via AWS Marketplace</div></div></div>

## Deploy on the Azure Marketplace

Visit the [product page on Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.annotation_lab?tab=Overview) and follow the instructions on the video below to subscribe and deploy. 

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='e6aB3z5tB0k' -%}<div class="video-descr">Deploy Annotation Lab via Azure Marketplace</div></div></div>

## Deploy on AirGap environment 

### Get artifact

```bash
wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/annotationlab/annotationlab-VERSION.tar.gz
```
replace VERSION with the version you want to download and install. 

### Fresh installation

```bash
$ sudo su
$ ./annotationlab-installer.sh
```
### Upgrade version

```bash
$ sudo su
$ ./annotationlab-updater.sh
```
### Work over proxy

- Custom CA certificate

You can provide a custom CA certificate chain to be included into the deployment. To do it add `--set-file custom_cacert=./cachain.pem` options to `helm install/upgrade` command inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.
cachain.pem must include a certificate in the following format:
```bash
-----BEGIN CERTIFICATE-----
....
-----END CERTIFICATE-----
```

- Proxy env variables

You can provide a proxy to use for external communications. To do that add 

    `--set proxy.http=[protocol://]<host>[:port]`, 
    `--set proxy.https=[protocol://]<host>[:port]`, 
    `--set proxy.no=<comma-separated list of hosts/domains>` 

commands inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.


## Recommended Configurations

**System requirements**. You can install Annotation Lab on a Ubuntu 20+ machine.

**Port requirements**. Annotation Lab expects ports 443 and 80 to be open by default. 

**Server requirements**. The minimal required configuration is **32GB RAM, 8 Core CPU, 512 SSD**. 
The ideal configuration in case model training and preannotations are required on a large number of tasks is **64 GiB, 16 Core CPU, 2TB HDD, 512 SSD**. 

**Web browser support**. Annotation Lab is tested with the latest version of Google Chrome and is expected to work in the latest versions of:

•   Google Chrome

•   Apple Safari

•   Mozilla Firefox
