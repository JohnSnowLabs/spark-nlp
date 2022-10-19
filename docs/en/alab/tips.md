---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Settings
permalink: /docs/en/alab/tips
key: docs-training
modify_date: "2021-12-09"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---


## Optimize view for large taxonomy

For projects that include a large number of labels, we have created a way to optimize the taxonomy display so that users can quickly find the label they are searching for. 

<img class="image image__shadow" src="/assets/images/annotation_lab/settings/large_taxonomy.png" style="width:100%;"/>

To obtain the above display please use the following configuration:

```xml
<View>
    <Filter name="fl" toName="label" hotkey="shift+f" minlength="1" />
    <View style="
        background:white;
        height: 100px;
        overflow-y:scroll;
        resize:vertical;
        position:sticky;
        top:0;"
    >
        <Labels name="label" toName="text">
            <Label value="Person" background="red"></Label>
            <Label value="Organization" background="darkorange"></Label>
        </Labels>
    </View>
    <View style="
        resize:vertical;
        margin-top:10px; 
        max-height:400px;
        overflow-y:scroll;"
    >
        <Text name="text" value="$text"></Text>
    </View>
</View>
```
