package com.johnsnowlabs.nlp.annotators.common
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object LabeledDependency extends Annotated[ConllSentence] {

  val ROOT_HEAD: Int = -1
  val ROOT_INDEX: Int = 1

  override def annotatorType: String = AnnotatorType.LABELED_DEPENDENCY

  override def unpack(annotations: Seq[Annotation]): Seq[ConllSentence] = {
    val posTagged = annotations.filter(_.annotatorType == AnnotatorType.POS).toArray
    val tokens = annotations.filter(_.annotatorType == AnnotatorType.TOKEN).toArray
    val unlabeledDependencies = annotations.filter(_.annotatorType == AnnotatorType.DEPENDENCY).toArray

    val conll = unlabeledDependencies.zipWithIndex.map{ case(unlabeledDependency, index) =>
      val form = tokens(index).result
      val lemma = tokens(index).result.toLowerCase
      val pos = posTagged(index).result
      val head = unlabeledDependency.metadata.getOrElse("head", "-1").toInt
      val sentence = tokens(index).metadata.getOrElse("sentence", "0").toInt
      ConllSentence(form, lemma, pos, pos, "_", head, sentence, unlabeledDependency.begin, unlabeledDependency.end)
    }

   conll
  }

  def unpackHeadAndRelation(annotations: Seq[Annotation]): Seq[DependencyInfo] = {

    val tokens = annotations.filter(_.annotatorType == AnnotatorType.TOKEN)
    val unlabeledDependencies = annotations.filter(_.annotatorType == AnnotatorType.DEPENDENCY).toArray
    val labeledDependency = annotations.filter(_.annotatorType == annotatorType).toArray

    if (unlabeledDependencies.length != labeledDependency.length) {
      throw new IndexOutOfBoundsException("Dependency and Typed Dependency Parser have different length.")
    }

    unlabeledDependencies.zipWithIndex.map{ case (unlabeledDependency, index) =>
      val token = tokens(index)
      val headIndex = unlabeledDependency.metadata("head").toInt
      val headBegin = unlabeledDependency.metadata("head.begin").toInt
      val headEnd = unlabeledDependency.metadata("head.end").toInt
      val head = if (headIndex == 0) "*" + unlabeledDependency.result + "*" else unlabeledDependency.result
      val relation = if (headIndex == 0) "*" + labeledDependency(index).result + "*" else labeledDependency(index).result

      DependencyInfo(token.begin, token.end, token.result, headBegin, headEnd, headIndex, head, relation)
    }
  }

  override def pack(conllSentences: Seq[ConllSentence]): Seq[Annotation] = {

    val root = conllSentences.last
    val arrangedSentences = moveToFront(root, conllSentences.toList)

    val annotations = arrangedSentences.map{ arrangedSentence =>
      val head = arrangedSentence.head
      if (head != ROOT_HEAD) {
        val label = arrangedSentence.deprel
        Annotation(annotatorType, arrangedSentence.begin, arrangedSentence.end,
          label, Map("sentence" -> arrangedSentence.sentence.toString))
      }
    }
    annotations.drop(ROOT_INDEX).asInstanceOf[Seq[Annotation]]
  }

  def moveToFront[A](y: A, xs: List[A]): List[A] = {
    xs.span(_ != y) match {
      case (as, h::bs) => h :: as ++ bs
      case _           => xs
    }
  }

  case class DependencyInfo(beginToken: Int, endToken: Int, token: String, beginHead: Int, endHead: Int, headIndex: Int,
                            head: String, relation: String)

}
