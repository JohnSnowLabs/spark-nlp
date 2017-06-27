package com.jsl.nlp

import java.io.File

import com.jsl.nlp.annotators.{EntityExtractor, Normalizer, RegexTokenizer, Stemmer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created by alext on 6/23/17.
  */
object Demo {

  def main(args: Array[String]): Unit = {

    println("## CREATING SPARK SESSION ##")
    val spark = SparkSession.builder().master("local[4]").appName("demo").getOrCreate()

    val toDocument = spark.udf.register("to_document", (id: String, text: String) => Document(id, text))

    println("## LOADING TEXT ##")
    val text = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/emr-example.txt")).mkString

    println(text)

    pause()

    println("## CREATING DATAFRAME ##")
    val df: DataFrame = DataBuilder.basicDataBuild(text)

    df.show()

    pause()

    println("## LOADING CONCEPTS ##")
    val pathToData = System.getProperty("user.dir") + File.separator + "src/test/resources/data.parquet"
    val concepts = spark.read.parquet(pathToData)
      .withColumn("id", concat_ws("-",
        col("Concept_Unique_Identifier"),
        col("Atom_Unique_Identifier"),
        col("String_Unique_Identifier")))
      .withColumn("document", toDocument(col("id"), col("String_Name")))

    concepts.show()

    pause()

    println("## CREATING TRANSFORMERS ##")
    val tokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setOutputAnnotationCol("tokens")
    val stemmer = new Stemmer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("tokens"))
      .setOutputAnnotationCol("stems")
    val normalizer = new Normalizer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("stems"))
      .setOutputAnnotationCol("ntokens")

    println("## PROCESSING CONCEPTS ##")

    val extractStems = spark.udf.register("extract_ntokens", (annos: Seq[Row]) => {
      annos.map {
        anno =>
          anno.getMap[String, String](anno.fieldIndex("metadata"))(Normalizer.aType)
      }
    })

    val procdConcepts = normalizer.transform(stemmer.transform(tokenizer.transform(concepts.select("document"))))
      .withColumn("ntoken_arr", extractStems(col("ntokens")))

    val localConcepts: Set[Seq[String]] =
      procdConcepts.select("ntoken_arr").dropDuplicates().collect().map(r => r.getSeq[String](0)).toSet

    println("## CREATING EXTRACTOR ##")

    val entityExtractor = new EntityExtractor()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("ntokens"))
      .setOutputAnnotationCol("entities")
      .setEntities(localConcepts)
      .setMaxLen(3)

    println("## PROCESSING DOCUMENT ##")

    val procdDoc = entityExtractor.transform(normalizer.transform(stemmer.transform(tokenizer.transform(df))))

    procdDoc.selectExpr("explode(entities)").collect().foreach(println)

    pause()

    println("## SHUTTING DOWN ##")

    spark.stop()
  }

  def pause(): Unit = {
    println("\n" * 5)
    println("## PRESS ANY BUTTON TO CONTINUE ##")
    scala.io.StdIn.readLine()
    println("\n" * 5)
  }
}
