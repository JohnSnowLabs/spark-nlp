---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Installation
permalink: /docs/en/alab/install
key: docs-training
modify_date: "2022-11-06"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}
th {
  width: 200px;
  text-align: left;
  background-color: #f7f7f7;
  vertical-align: "top";
}
</style>

## Dedicated Server

Install Annotation Lab on a dedicated server to reduce the likelihood of conflicts or unexpected behavior.

### Fresh install

To install Annotation Lab run the following command:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s $VERSION
```

Replace `$VERSION` in the above one liners with the version you want to install.

For installing the latest available version of the Annotation Lab use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s --
```

<br />

### Upgrade

To upgrade your Annotation Lab installation to a newer version, run the following command on a terminal:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s $VERSION
```

Replace `$VERSION` in the above one liners with the version you want to upgrade to.

For upgrading to the latest version of the Annotation Lab, use:

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s --
```

> **NOTE:** The install/upgrade script displays the login credentials for the _admin_ user on the terminal.

After running the install/upgrade script, the Annotation Lab is available at http://INSTANCE_IP or https://INSTANCE_IP

<img class="image image--xl image__shadow" src="/assets/images/annotation_lab/4.1.0/loginScreenALAB.png" style="width:100%;"/>

We have an aesthetically pleasing Sign-In Page with a section highlighting the key features of Annotation Lab using animated GIFs.

## AWS Marketplace

Visit the [product page on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-nsww5rdpvou4w?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) and follow the instructions on the video below to subscribe and deploy.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='ebaewU4BcQA' -%}<div class="video-descr">Deploy Annotation Lab via AWS Marketplace</div></div></div>

## Azure Marketplace

Visit the [product page on Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/johnsnowlabsinc1646051154808.annotation_lab?tab=Overview) and follow the instructions on the video below to subscribe and deploy.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='e6aB3z5tB0k' -%}<div class="video-descr">Deploy Annotation Lab via Azure Marketplace</div></div></div>

## AirGap Environment

### Get Artifact

Run the following command on a terminal to fetch the compressed artifact (_tarball_) of the Annotation Lab.

```bash
wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/annotationlab/annotationlab-$VERSION.tar.gz
```

Extract the tarball and the change directory to the extracted folder (_artifacts_):

```bash
tar -xzf annotationlab-$VERSION.tar.gz
cd artifacts
```

Replace `$VERSION` with the version you want to download and install.

<br />

### Fresh Install

Run the installer script `annotationlab-installer.sh` with `sudo` privileges.

```bash
$ sudo su
$ ./annotationlab-installer.sh
```

<br />

### Upgrade

Run the upgrade script `annotationlab-updater.sh` with `sudo` privileges.

```bash
$ sudo su
$ ./annotationlab-updater.sh
```

<br />

### Work over proxy

**Custom CA certificate**

You can provide a custom CA certificate chain to be included into the deployment. To do it add `--set-file custom_cacert=./cachain.pem` options to `helm install/upgrade` command inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.

_cachain.pem_ must include a certificate in the following format:

```bash
-----BEGIN CERTIFICATE-----
....
-----END CERTIFICATE-----
```

<br />

**Proxy env variables**

You can provide a proxy to use for external communications. To do that add

    `--set proxy.http=[protocol://]<host>[:port]`,
    `--set proxy.https=[protocol://]<host>[:port]`,
    `--set proxy.no=<comma-separated list of hosts/domains>`

commands inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.

## Recommended Configurations

<table>
  <tr>
    <th>System requirements</th>
    <td>You can install Annotation Lab on a Ubuntu 20+ machine.</td>
  </tr>
  <tr>
    <th>Port requirements</th>
    <td>Annotation Lab expects ports <bl>443</bl> and <bl>80</bl> to be open by default.</td>
  </tr>
  <tr>
    <th>Server requirements</th>
    <td>The minimal required configuration is <bl>32GB RAM, 8 Core CPU, 512 SSD</bl>.<br /><br />

    The ideal configuration in case model training and preannotations are required on a large number of tasks is <bl>64 GiB, 16 Core CPU, 2TB HDD, 512 SSD</bl>.
    </td>

  </tr>
  <tr>
    <th>Web browser support</th>
    <td>Annotation Lab is tested with the latest version of Google Chrome and is expected to work in the latest versions of:
      <ul>
      <li>Google Chrome</li>
      <li>Apple Safari</li>
      <li>Mozilla Firefox</li>
      </ul>
    </td>
  </tr>
</table>
