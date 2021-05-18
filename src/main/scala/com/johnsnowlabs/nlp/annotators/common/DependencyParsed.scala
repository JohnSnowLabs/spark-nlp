package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}


object DependencyParsed extends Annotated[DependencyParsedSentence]{

  override def annotatorType: String = AnnotatorType.DEPENDENCY

  override def unpack(annotations: Seq[Annotation]): Seq[DependencyParsedSentence] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    val depAnnotations = annotations
      .filter(a => a.annotatorType == annotatorType)
      .sortBy(a => a.begin)

    var last = 0
    sentences.map{sentence =>
      val sorted = sentence.indexedTokens.sortBy(t => t.begin)
      val dependencies = (last until (last + sorted.length)).map { i =>
        depAnnotations(i).metadata("head").toInt
      }

      last += sorted.length

      val words = sorted.zip(dependencies).map{
        case (token, head) =>
          WordWithDependency(token.token, head, "", token.begin, token.end)
      }

      DependencyParsedSentence(words)
    }
  }

  override def pack(items: Seq[DependencyParsedSentence]): Seq[Annotation] = {
    items.flatMap{ dependencyParsedSentence =>
      dependencyParsedSentence.wordsWithDependency.map { wordWithDependency =>
        val headData = getHeadData(wordWithDependency.head, dependencyParsedSentence)
        val realHead = if (wordWithDependency.head == dependencyParsedSentence.wordsWithDependency.length) 0
                       else wordWithDependency.head + 1
        Annotation(annotatorType, wordWithDependency.begin, wordWithDependency.end, headData.word,
          Map("head" -> realHead.toString, "head.begin" -> headData.begin.toString, "head.end" -> headData.end.toString))
      }
    }
  }

  def packDependencyRelations(dependencyRelations: Seq[DependencyParsedSentence]): Seq[Annotation] = {
    dependencyRelations.flatMap{ dependencyParsedSentence =>
      dependencyParsedSentence.wordsWithDependency.map { wordWithDependency =>
        val realHead = wordWithDependency.head - 1
        val headData = dependencyParsedSentence.wordsWithDependency.lift(realHead)
          .getOrElse( WordWithDependency("ROOT", -1, wordWithDependency.dependencyRelation, -1, -1))

        Annotation(annotatorType, wordWithDependency.begin, wordWithDependency.end, headData.word,
          Map("head" -> realHead.toString, "head.begin" -> headData.begin.toString, "head.end" -> headData.end.toString))
      }
    }
  }

  def getHeadData(head: Int, sentence: DependencyParsedSentence): WordWithDependency = {
    val root: WordWithDependency = WordWithDependency("ROOT", -1, "", -1, -1)
    sentence.wordsWithDependency.lift(head).getOrElse(root)
  }

}