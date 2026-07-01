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

package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.annotator.{PerceptronModel, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.util.{AnnotationRowUtils, SparkNlpConfig}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler, Finisher}
import com.johnsnowlabs.util.Version
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/** The PubTator format includes medical papers’ titles, abstracts, and tagged chunks.
  *
  * For more information see
  * [[http://bioportal.bioontology.org/ontologies/EDAM?p=classes&conceptid=format_3783 PubTator Docs]]
  * and [[http://github.com/chanzuckerberg/MedMentions MedMentions Docs]].
  *
  * `readDataset` is used to create a Spark DataFrame from a PubTator text file.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.training.PubTator
  *
  * val pubTatorFile = "./src/test/resources/corpus_pubtator_sample.txt"
  * val pubTatorDataSet = PubTator().readDataset(ResourceHelper.spark, pubTatorFile)
  * pubTatorDataSet.show(1)
  * +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
  * |  doc_id|      finished_token|        finished_pos|        finished_ner|finished_token_metadata|finished_pos_metadata|finished_label_metadata|
  * +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
  * |25763772|[DCTN4, as, a, mo...|[NNP, IN, DT, NN,...|[B-T116, O, O, O,...|   [[sentence, 0], [...| [[word, DCTN4], [...|   [[word, DCTN4], [...|
  * +--------+--------------------+--------------------+--------------------+-----------------------+---------------------+-----------------------+
  * }}}
  */
case class PubTator() {

  private def isSpark4OrNewer(spark: SparkSession): Boolean =
    Version.parse(spark.version).toFloat >= 4.0f

  private def iobTaggingAnnotations(
      tokens: scala.collection.Seq[Row],
      chunkLabels: scala.collection.Seq[Row],
      isPaddedToken: Boolean): Seq[Annotation] = {
    val tokenAnnotations = tokens.map(Annotation(_))
    val labelAnnotations = chunkLabels.map(Annotation(_))
    tokenAnnotations
      .map(ta => {
        val tokenLabel = labelAnnotations.find(la => la.begin <= ta.begin && la.end >= ta.end)
        val tokenTag = {
          if (tokenLabel.isEmpty) "O"
          else {
            val tokenCSV = tokenLabel.get.metadata("entity")
            if (tokenCSV == "UnknownType") "O"
            else {
              val tokenPrefix = if (ta.begin == tokenLabel.get.begin) "B-" else "I-"
              val token = if (isPaddedToken) {
                "T" + "%03d".format(tokenCSV.split(",")(0).slice(1, 4).toInt)
              } else tokenCSV
              tokenPrefix + token
            }
          }
        }

        Annotation(
          AnnotatorType.NAMED_ENTITY,
          ta.begin,
          ta.end,
          tokenTag,
          Map("word" -> ta.result))
      })
      .toSeq
  }

  private def tagWithRows(
      alignedDf: DataFrame,
      labelMeta: Metadata,
      isPaddedToken: Boolean): DataFrame = {
    val inputDataFrame = alignedDf.toDF()
    val outputSchema = StructType(
      inputDataFrame.schema.fields :+
        StructField("label", Annotation.arrayType, nullable = true, labelMeta))
    val tokenIndex = inputDataFrame.schema.fieldIndex("token")
    val chunkIndex = inputDataFrame.schema.fieldIndex("chunk")

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame
      .mapPartitions { rows =>
        rows.map { row =>
          val tokens = AnnotationRowUtils.extractAnnotationRows(row, tokenIndex)
          val chunkLabels = AnnotationRowUtils.extractAnnotationRows(row, chunkIndex)
          val labelRows = AnnotationRowUtils
            .annotationsToRows(iobTaggingAnnotations(tokens, chunkLabels, isPaddedToken))
            .toVector
          Row.fromSeq(row.toSeq :+ labelRows)
        }
      }
      .toDF()

    outputSchema.fields.foldLeft(mappedDataFrame)((dataFrame, field) => {
      dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
    })
  }

  def readDataset(spark: SparkSession, path: String, isPaddedToken: Boolean = true): DataFrame = {
    val pubtator = spark.sparkContext.textFile(path)
    val titles = pubtator.filter(x => x.contains("|a|") | x.contains("|t|"))
    val titlesText = titles
      .map(x => x.split("\\|"))
      .groupBy(_.head)
      .map(x => (x._1.toInt, x._2.foldLeft(Seq[String]())((a, b) => a ++ Seq(b.last))))
      .map(x => (x._1, x._2.mkString(" ")))
    val df = spark.createDataFrame(titlesText).toDF("doc_id", "text")
    val docAsm = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val setDet = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tknz = new Tokenizer().setInputCols("sentence").setOutputCol("token")
    val pl = new Pipeline().setStages(Array(docAsm, setDet, tknz))
    val nlpDf = pl.fit(df).transform(df)
    val annotations = pubtator.filter(x => !x.contains("|a|") & !x.contains("|t|") & x.nonEmpty)
    val splitAnnotations = annotations
      .map(_.split("\\t"))
      .map(x => (x(0), x(1).toInt, x(2).toInt - 1, x(3), x(4), x(5)))
    val docAnnotations = splitAnnotations
      .groupBy(_._1)
      .map(x => (x._1, x._2))
      .map(x =>
        (
          x._1.toInt,
          x._2.zipWithIndex
            .map(a =>
              (new Annotation(
                AnnotatorType.CHUNK,
                a._1._2,
                a._1._3,
                a._1._4,
                Map("entity" -> a._1._5, "chunk" -> a._2.toString),
                Array[Float]())))
            .toList))
    val chunkMeta = new MetadataBuilder().putString("annotatorType", AnnotatorType.CHUNK).build()
    val annDf = spark
      .createDataFrame(docAnnotations)
      .toDF("doc_id", "chunk")
      .withColumn("chunk", col("chunk").as("chunk", chunkMeta))
    val alignedDf =
      nlpDf.join(annDf, Seq("doc_id")).selectExpr("doc_id", "sentence", "token", "chunk")
    val iobTagging = udf((tokens: Seq[Row], chunkLabels: Seq[Row]) => {
      iobTaggingAnnotations(tokens, chunkLabels, isPaddedToken)
    })
    val labelMeta =
      new MetadataBuilder().putString("annotatorType", AnnotatorType.NAMED_ENTITY).build()
    val taggedDf =
      if (isSpark4OrNewer(spark)) {
        tagWithRows(alignedDf, labelMeta, isPaddedToken)
      } else {
        alignedDf.withColumn(
          "label",
          iobTagging(col("token"), col("chunk")).as("label", labelMeta))
      }

    val pos =
      PerceptronModel.pretrained().setInputCols(Array("sentence", "token")).setOutputCol("pos")
    val finisher = new Finisher().setInputCols("token", "pos", "label").setIncludeMetadata(true)
    val finishingPipeline = new Pipeline().setStages(Array(pos, finisher))
    finishingPipeline
      .fit(taggedDf)
      .transform(taggedDf)
      .withColumnRenamed("finished_label", "finished_ner") // CoNLL generator expects finished_ner
  }
}
