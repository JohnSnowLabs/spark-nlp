package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{DependencyParsed, DependencyParsedSentence, PosTagged}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DependencyParserModel(override val uid: String) extends AnnotatorModel[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val annotatorType: String = DEPENDENCY

  override val requiredAnnotatorTypes =  Array[String](DOCUMENT, POS, TOKEN)

  val sourcePath = new Param[String](this, "sourcePath", "source file for dependency model")

  def setSourcePath(value: String): this.type = set(sourcePath, value)

  def tag(sentence: PosTaggedSentence): DependencyParsedSentence = {
    val model = new GreedyTransitionApproach()
    model.parse(sentence, $(sourcePath))
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val posTaggedSentences = PosTagged.unpack(annotations)
    val sentencesWithDependency = posTaggedSentences.map{sentence => tag(sentence)}
    DependencyParsed.pack(sentencesWithDependency)
  }
}

object DependencyParserModel extends DefaultParamsReadable[DependencyParserModel]
