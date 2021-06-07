package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row}

/**
  * This transformer is designed to deal with embedding annotators, for example:
  * [[com.johnsnowlabs.nlp.embeddings.WordEmbeddings WordEmbeddings]],
  * [[com.johnsnowlabs.nlp.embeddings.BertEmbeddings BertEmbeddings]],
  * [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] and
  * [[com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings ChunkEmbeddings]].
  * By using `EmbeddingsFinisher` you can easily transform your embeddings into array of floats or vectors which are
  * compatible with Spark ML functions such as LDA, K-mean, Random Forest classifier or any other functions that require
  * `featureCol`.
  *
  * For more extended examples see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.1_Text_classification_examples_in_SparkML_SparkNLP.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import org.apache.spark.ml.Pipeline
  * import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
  * import com.johnsnowlabs.nlp.annotator.{Normalizer, StopWordsCleaner, Tokenizer, WordEmbeddingsModel}
  *
  * // First the embeddings are extracted using the WordEmbeddingsModel
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val normalizer = new Normalizer()
  *   .setInputCols("token")
  *   .setOutputCol("normalized")
  *
  * val stopwordsCleaner = new StopWordsCleaner()
  *   .setInputCols("normalized")
  *   .setOutputCol("cleanTokens")
  *   .setCaseSensitive(false)
  *
  * val gloveEmbeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("document", "cleanTokens")
  *   .setOutputCol("embeddings")
  *   .setCaseSensitive(false)
  *
  * // Then the embeddings can be turned into a vector using the EmbeddingsFinisher
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_sentence_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val data = Seq("Spark NLP is an open-source text processing library.")
  *   .toDF("text")
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   normalizer,
  *   stopwordsCleaner,
  *   gloveEmbeddings,
  *   embeddingsFinisher
  * )).fit(data)
  *
  * val result = pipeline.transform(data)
  * result.select("finished_sentence_embeddings").show(false)
  * +--------------------------------------------------------------------------------------------------------+
  * |finished_sentence_embeddings                                                                            |
  * +--------------------------------------------------------------------------------------------------------+
  * |[[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.6856099963188171,0.5442799925804138...|
  * +--------------------------------------------------------------------------------------------------------+
  * }}}
  *
  * @see [[com.johnsnowlabs.nlp.Finisher Finisher]] for finishing Strings
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
class EmbeddingsFinisher(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  /**
    * Name of input annotation cols containing embeddings
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "Name of input annotation cols containing embeddings")

  /**
    * Name of EmbeddingsFinisher output cols
    * @group param
    */
  val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "Name of EmbeddingsFinisher output cols")

  /**
    * Whether to remove all the existing annotation columns (Default: `true`)
    * @group param
    */
  val cleanAnnotations: BooleanParam =
    new BooleanParam(this, "cleanAnnotations", "Whether to remove all the existing annotation columns (Default: `true`)")

  /**
    * If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    * @group param
    */
  val outputAsVector: BooleanParam =
    new BooleanParam(this, "outputAsVector", "If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)")

  /**
    * Name of input annotation cols containing embeddings
    * @group setParam
    */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /**
    * Name of input annotation cols containing embeddings
    * @group setParam
    */
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /**
    * Name of EmbeddingsFinisher output cols
    * @group setParam
    */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /**
    * Name of EmbeddingsFinisher output cols
    * @group setParam
    */
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)

  /**
    * Whether to remove all the existing annotation columns (Default: `true`)
    * @group setParam
    */
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)

  /**
    * If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    * @group setParam
    */
  def setOutputAsVector(value: Boolean): this.type = set(outputAsVector, value)

  /**
    * Name of input annotation cols containing embeddings
    * @group getParam
    */
  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))

  /**
    * Name of EmbeddingsFinisher output cols
    * @group getParam
    */
  def getInputCols: Array[String] = $(inputCols)

  /**
    * Whether to remove all the existing annotation columns (Default: `true`)
    * @group getParam
    */
  def getCleanAnnotations: Boolean = $(cleanAnnotations)

  /**
    * If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    * @group getParam
    */
  def getOutputAsVector: Boolean = $(outputAsVector)

  setDefault(
    cleanAnnotations -> true,
    outputAsVector -> false
  )

  def this() = this(Identifiable.randomUID("embeddings_finisher"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")

    val embeddingsAnnotators = Seq(
      AnnotatorType.WORD_EMBEDDINGS,
      AnnotatorType.SENTENCE_EMBEDDINGS
    )

    getInputCols.foreach {
      annotationColumn =>

        /**
          * Check if the inputCols exist
          */
        require(getInputCols.forall(schema.fieldNames.contains),
          s"pipeline annotator stages incomplete. " +
            s"expected: ${getInputCols.mkString(", ")}, " +
            s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
            s"among available: ${schema.fieldNames.mkString(", ")}")

        /**
          * Check if the annotationColumn is(are) Spark NLP annotations
          */
        require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
          s"column [$annotationColumn] must be an NLP Annotation column")

        /**
          * Check if the annotationColumn has embeddings
          * It must be at least of one these annotators: WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings
          */
        require(embeddingsAnnotators.contains(schema(annotationColumn).metadata.getString("annotatorType")),
          s"column [$annotationColumn] must be a type of either WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings")

    }
    val metadataFields =  getOutputCols.flatMap(outputCol => {
      if ($(outputAsVector))
        Some(StructField(outputCol + "_metadata", MapType(StringType, StringType), nullable = false))
      else
        None
    })

    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => {
        if ($(outputAsVector))
          StructField(outputCol, ArrayType(VectorType), nullable = false)
        else
          StructField(outputCol, ArrayType(FloatType), nullable = false)
      }) ++ metadataFields

    val cleanFields = if ($(cleanAnnotations)) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
    StructType(cleanFields)
  }

  private def vectorsAsArray: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings
  }

  private def vectorsAsVectorType: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings.map(embedding =>
      Vectors.dense(embedding.toArray.map(_.toDouble))
    )
  }

  override def transform(dataset: Dataset[_]): Dataset[Row] = {
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")

    val cols = getInputCols.zip(getOutputCols)
    var flattened = dataset
    cols.foreach { case (inputCol, outputCol) =>
      flattened = {
        flattened.withColumn(
          outputCol, {
            if ($(outputAsVector))
              vectorsAsVectorType(flattened.col(inputCol + ".embeddings"))
            else
              vectorsAsArray(flattened.col(inputCol + ".embeddings"))
          }
        )
      }
    }

    if ($(cleanAnnotations)) flattened.drop(
      flattened.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name):_*)
    else flattened.toDF()
  }

}
object EmbeddingsFinisher extends DefaultParamsReadable[EmbeddingsFinisher]