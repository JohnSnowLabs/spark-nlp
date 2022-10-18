---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Backup and Restore
permalink: /docs/en/alab/backup_restore
key: docs-training
modify_date: "2022-10-13"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

## Backup

You can enable daily backups by adding several variables with `--set` option to helm command in `annotationlab-updater.sh`:

```bash
backup.enable=true
backup.files=true
backup.s3_access_key="<ACCESS_KEY>"
backup.s3_secret_key="<SECRET_KEY>"
backup.s3_bucket_fullpath="<FULL_PATH>"
```

`<ACCESS_KEY>` - your access key for AWS S3 access

`<SECRET_KEY>` - your secret key for AWS S3 access

`<FULL_PATH>` - full path to your backup in s3 bucket (f.e. `s3://example.com/path/to/my/backup/dir`)

**Note:** File backup is enabled by default. If you don't need to backup files, you have to change

```bash
backup.files=true
```

to

```bash
backup.files=false
```

<br>

#### Configure Backup from the UI

In 2.8.0 release, Annotation Lab added support for defining database and files backups via the UI. An _admin_ user can view and edit the backup settings under the **Settings** menu. Users can select different backup periods and can specify a target S3 bucket for storing the backup files. New backups will be automatically generated and saved to the S3 bucket following the defined schedule.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/backupRestoreUI.png" style="width:100%;"/>

## Restore

#### Database

To restore Annotation Lab from a backup you need a fresh installation of Annotation Lab. Install it using `annotationlab-install.sh`. Now, download the latest backup from your S3 bucket and move the archive to `restore/database/` directory. Next, go to the `restore/database/` directory and execute script `restore_all_databases.sh` with the name of your backup archive as the argument.

For example:

```
cd restore/database/
sudo ./restore_all_databases.sh 2022-04-14-annotationlab-all-databases.tar.xz
```

> **Note:** <br>
>
> 1. You need `xz` and `bash` installed to execute this script. <br>
> 2. This script works only with backups created by Annotation Lab backup system. <br>
> 3. Run this script with `sudo` command

After database restore complete you can check logs in `restore_log` directory created by restore script.

<br>

#### Files

Download your files backup and move it to `restore/files/` directory. Go to `restore/files/` directory and execute script `restore_files.sh` with the name of your backup archive as the argument. For example:

```
cd restore/files/
sudo ./restore_files.sh 2022-04-14-annotationlab-files.tar
```

> **Note:** <br>
>
> 1. You need `bash` installed to execute this script. <br>
> 2. This script works only with backups created by Annotation Lab backup system. <br>
> 3. Run this script with `sudo` command

<br>

#### Reboot

After restoring database and files, reboot Annotation Lab:

```
sudo reboot
```
