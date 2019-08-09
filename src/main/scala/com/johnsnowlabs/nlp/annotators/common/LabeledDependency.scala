package com.johnsnowlabs.nlp.annotators.common
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}


object LabeledDependency extends Annotated[ConllSentence] {

  val ROOT_HEAD: Int = -1
  val ROOT_INDEX: Int = 1

  override def annotatorType: String = AnnotatorType.POS

  override def unpack(annotations: Seq[Annotation]): Seq[ConllSentence] = {
    val posTagged = annotations.filter(_.annotatorType == annotatorType).toArray
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

  override def pack(conllSentences: Seq[ConllSentence]): Seq[Annotation] = {

    val root = conllSentences.last
    val arrangedSentences = moveToFront(root, conllSentences.toList)

    val annotations = arrangedSentences.map{arrangedSentence =>
      val head = arrangedSentence.head
      if (head != ROOT_HEAD) {
        val label = arrangedSentence.deprel
        Annotation(AnnotatorType.LABELED_DEPENDENCY, arrangedSentence.begin, arrangedSentence.end,
          label, Map())
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

}
