---
layout: article
title: Contribute
permalink: /contribute
key: contribute
aside:
    toc: false
sidebar:
    nav: extras    
license: false
show_edit_on_github: true
show_date: true
header:
    theme: light
    background: "#ffffff"
modify_date: "2019-06-30"
---

Refer to our GitHub page to take a look at the GH Issues, as the project is yet small. You can create in there your own issues to either work on them yourself or simply propose them.

Feel free to clone the repository locally and submit pull requests so we can review them and work together.

* feedback, ideas and bug reports
* testing and development
* training and testing nlp corpora
* documentation and research

Help is always welcome, for any further questions, contact nlp@johnsnowlabs.com.

## Your own annotator model

Creating your first annotator transformer should not be hard, here are a few guidelines to get you started. Lets assume we want a wrapper annotator, which puts a character surrounding tokens provided by a Tokenizer

### WordWrapper

uid is utilized for transformer serialization, AnnotatorModel[MyAnnotator] will contain the common annotator logic We need to use standard constructor for java and python compatibility

```scala
class WordWrapper(override val uid: String) extends AnnotatorModel[WordWrapper] {
    def this() = this(Identifiable.randomUID("WORD_WRAPPER"))
}
```

### Annotator attributes

This annotator is not flexible if we don't provide parameters

```scala
import com.johnsnowlabs.nlp.AnnotatorType._
override val annotatorType: AnnotatorType = TOKEN
override val requiredAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](TOKEN)

```

### Annotator parameters

This annotator is not flexible if we don't provide parameters

```scala
protected val character: Param[String] = new Param(this, "character", "this is the character used to wrap a token")
def setCharacter(value: String): this.type = set(pattern, value)
def getCharacter: String = $(pattern)
    setDefault(character, "@")
```

### Annotator logic

Here is how we act, annotations will automatically provide our required annotations We generally use annotatorType for metadata keys

```scala
override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map(annotation => {
            Annotation(
            annotatorType,
            annotation.begin,
            annotation.end,
            Map(annotatorType -> $(character) + annotation.result + $(character))
        })
    }
```
