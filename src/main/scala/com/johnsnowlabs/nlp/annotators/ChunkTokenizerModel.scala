package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.ChunkSplit
import com.johnsnowlabs.nlp.{Annotation, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.Identifiable


/**
  *
  *
  * @param uid required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class ChunkTokenizerModel(override val uid: String) extends TokenizerModel {

  def this() = this(Identifiable.randomUID("CHUNK_TOKENIZER"))

  /** Output Annotator Type : CHUNK
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)
  /** Output Annotator Type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** one to many annotation */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = ChunkSplit.unpack(annotations)
    val tokenized = tag(sentences)

    tokenized.zipWithIndex.flatMap { case (sentence, sentenceIndex) =>
      sentence.indexedTokens.map { token =>
        Annotation(outputAnnotatorType, token.begin, token.end, token.token,
          Map("chunk" -> sentenceIndex.toString, "sentence" -> sentence.sentenceIndex.toString))
      }}

  }

}

object ChunkTokenizerModel extends ParamsAndFeaturesReadable[ChunkTokenizerModel]