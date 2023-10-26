/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.{TokenPieceEmbeddings, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.param.{IntParam, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, FloatType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/** Word2Vec model that creates vector representations of words in a text corpus.
  *
  * The algorithm first constructs a vocabulary from the corpus and then learns vector
  * representation of words in the vocabulary. The vector representation can be used as features
  * in natural language processing and machine learning algorithms.
  *
  * We use Word2Vec implemented in Spark ML. It uses skip-gram model in our implementation and a
  * hierarchical softmax method to train the model. The variable names in the implementation match
  * the original C implementation.
  *
  * This is the instantiated model of the [[Word2VecApproach]]. For training your own model,
  * please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = Word2VecModel.pretrained()
  *   .setInputCols("token")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"word2vec_gigaword_300"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * '''Sources''' :
  *
  * For the original C implementation, see https://code.google.com/p/word2vec/
  *
  * For the research paper, see
  * [[https://arxiv.org/abs/1301.3781 Efficient Estimation of Word Representations in Vector Space]]
  * and
  * [[https://arxiv.org/pdf/1310.4546v1.pdf Distributed Representations of Words and Phrases and their Compositionality]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.{Tokenizer, Word2VecModel}
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  *
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val embeddings = Word2VecModel.pretrained()
  *   .setInputCols("token")
  *   .setOutputCol("embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[0.06222493574023247,0.011579325422644615,0.009919632226228714,0.109361454844...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class Word2VecModel(override val uid: String)
    extends AnnotatorModel[Word2VecModel]
    with HasSimpleAnnotate[Word2VecModel]
    with HasStorageRef
    with HasEmbeddingsProperties
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("Word2VecModel"))

  /** Input annotator type : TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** Output annotator type : WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: String = WORD_EMBEDDINGS

  /** The dimension of codes after transforming from words (> 0) (Default: `100`)
    *
    * @group param
    */
  val vectorSize = new IntParam(
    this,
    "vectorSize",
    "the dimension of codes after transforming from words (> 0)",
    ParamValidators.gt(0)).setProtected()

  /** @group getParam */
  def getVectorSize: Int = $(vectorSize)

  /** @group setParam */
  def setVectorSize(value: Int): this.type = {
    set(vectorSize, value)
  }

  /** Dictionary of words with their vectors
    *
    * @group param
    */
  val wordVectors: MapFeature[String, Array[Float]] = new MapFeature(this, "wordVectors")

  /** @group setParam */
  def setWordVectors(value: Map[String, Array[Float]]): this.type = set(wordVectors, value)

  private var sparkSession: Option[SparkSession] = None

  def getVectors: DataFrame = {
    val vectors: Map[String, Array[Float]] = $$(wordVectors)
    val rows = vectors.toSeq.map { case (key, values) => Row(key, values) }
    val schema = StructType(
      StructField("word", StringType, nullable = false) ::
        StructField("vector", ArrayType(FloatType), nullable = false) :: Nil)
    if (sparkSession.isEmpty) {
      throw new UnsupportedOperationException(
        "Vector representation empty. Please run Word2VecModel in some pipeline before accessing vector vocabulary.")
    }
    sparkSession.get.createDataFrame(sparkSession.get.sparkContext.parallelize(rows), schema)
  }

  setDefault(inputCols -> Array(TOKEN), outputCol -> "word2vec", vectorSize -> 100)

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    sparkSession = Some(dataset.sparkSession)
    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val sentences = annotations
      .filter(_.annotatorType == TOKEN)
      .groupBy(token => token.metadata("sentence").toInt)
      .toSeq
      .sortBy(_._1)

    if (sentences.nonEmpty) {
      val sentenceWithEmbeddings = sentences.map { case (index, sentence) =>
        val oovVector = Array.fill($(vectorSize))(0.0f)
        val tokensWithEmbeddings = sentence.map { token =>
          val embeddings = $$(wordVectors).getOrElse(token.result, oovVector)
          val isOOV = embeddings.sameElements(oovVector)

          TokenPieceEmbeddings(
            wordpiece = token.result,
            token = token.result,
            pieceId = -1,
            isWordStart = true,
            begin = token.begin,
            end = token.end,
            embeddings = embeddings,
            isOOV = isOOV)
        }.toArray
        WordpieceEmbeddingsSentence(tokensWithEmbeddings, index)
      }
      WordpieceEmbeddingsSentence.pack(sentenceWithEmbeddings)
    } else Seq.empty[Annotation]
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(vectorSize), Some($(storageRef))))
  }
}

trait ReadablePretrainedWord2Vec
    extends ParamsAndFeaturesReadable[Word2VecModel]
    with HasPretrained[Word2VecModel] {
  override val defaultModelName: Some[String] = Some("word2vec_gigaword_300")

  override def pretrained(name: String, lang: String, remoteLoc: String): Word2VecModel = {
    ResourceDownloader.downloadModel(Word2VecModel, name, Option(lang), remoteLoc)
  }

  /** Java compliant-overrides */
  override def pretrained(): Word2VecModel =
    pretrained(defaultModelName.get, defaultLang, defaultLoc)

  override def pretrained(name: String): Word2VecModel = pretrained(name, defaultLang, defaultLoc)

  override def pretrained(name: String, lang: String): Word2VecModel =
    pretrained(name, lang, defaultLoc)
}

/** This is the companion object of [[Word2VecModel]]. Please refer to that class for the
  * documentation.
  */
object Word2VecModel extends ReadablePretrainedWord2Vec
