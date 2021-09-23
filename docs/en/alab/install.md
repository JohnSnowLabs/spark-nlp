---
layout: docs
comment: no
header: true
title: Installation
permalink: /docs/en/alab/install
key: docs-training
modify_date: "2021-09-22"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## System requirements
You can install Annotation Lab on a Ubuntu 20+ machine.

## Port requirements
Annotation Lab expects ports 443 and 80 to be open by default. 

## Server requirements
The minimal required configuration is *32GB RAM, 8 Core CPU, 512 SSD*. 
The ideal configuration in case model training and preannotations are required on a large number of tasks is *64 GiB, 16 Core CPU, 2TB HDD, 512 SSD*. 

## Web browser support
Annotation Lab is tested with the latest version of Google Chrome and is expected to work in the latest versions of:
•   Google Chrome
•   Apple Safari
•   Mozilla Firefox

## Install via script

### Install prerequisite
Install Annotation Lab on a dedicated server to reduce the likelihood of conflicts or unexpected behavior.

### One liner install 
To install Annotation Lab run the following command:
```bash
wget https://setup.johnsnowlabs.com/annotationlab/install.sh -O - | sudo bash -s -- --version 2.0.1
```

### One liner upgrade

```bash
wget https://setup.johnsnowlabs.com/annotationlab/upgrade.sh -O - | sudo bash -s -- --version 2.0.1
```

After running the install/upgrade script the Annotation Lab is available at http://INSTANCE_IP  or https://INSTANCE_IP 
The install/upgrade script display the login credentials for the admin user on the terminal. 

## Custom installation 

### Get artifact

```bash
Wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/annotationlab/annotationlab-"$VERSION".tar.gz
```
replace $VERSION with the version you want to download and install. 

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
You can provide a proxy to use for external communications. To do it add `--set proxy.http=[protocol://]<host>[:port]`, `--set proxy.https=[protocol://]<host>[:port]`, `--set proxy.no=<comma-separated list of hosts/domains>` commands inside `annotationlab-installer.sh` and `annotationlab-updater.sh` files.

### Backup and restore

- Backup

You can enable daily backups by adding several variables with --set option to helm command in `annotationlab-updater.sh`:
```bash
backup.enable=true
backup.s3_access_key="<ACCESS_KEY>"
backup.s3_secret_key="<SECRET_KEY>"
backup.s3_bucket_fullpath="<FULL_PATH>"
```

`<ACCESS_KEY>` - your access key for aws s3 access
`<SECRET_KEY>` - your secret key for aws s3 access
`<FULL_PATH>` - full path to your backup in s3 bucket (f.e. s3://example.com/path/to/my/backup/dir)

- Restore

To restore from backup you need new clear installation of Annotation Lab. Do it with `annotationlab-install.sh`.
Next, you need to download latest backup from your s3 bucket and unpack an archive. There should be 3 sql backup files:

```bash
annotationlab.sql
keycloak.sql
airflow.sql
```
Run commands below to get PostgreSQL passwords.
airflow-postgres password for user `airflow`:
```bash
kubectl get secret $(k get secret | grep postgr | grep airflow | awk '{print $1}') -o jsonpath='{.data.postgresql-password}' | base64 -d
```
annotationlab-postgres password for user `annotationlab`:
```bash
kubectl get secret $(k get secret | grep annotationlab-postgre | awk '{print $1}') -o jsonpath='{.data.postgresql-password}' | base64 -d
```
keycloak-postgress password for user `keycloak`:
```bash
kubectl get secret $(k get secret | grep postgre | grep keyclo | awk '{print $1}') -o jsonpath='{.data.postgresql-password}' | base64 -d
```
Now you can restore your databases with `psql`, `pg_restore`, etc.

