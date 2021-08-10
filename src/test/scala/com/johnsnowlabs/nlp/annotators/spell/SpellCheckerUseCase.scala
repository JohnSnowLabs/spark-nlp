package com.johnsnowlabs.nlp.annotators.spell
import com.johnsnowlabs.nlp.annotator.RecursiveTokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.spell.context.{ContextSpellCheckerApproach, ContextSpellCheckerModel}
import org.apache.spark.ml.Pipeline

object SpellCheckerUseCase extends App  {

  // ==================================
  // Getting and cleaning all the data
  // ==================================

  // Let's use the Paisa corpus
  // https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3/paisa.raw.utf8.gz
  // (update with the path you downloaded the file to)
  val paisaCorpusPath = "/home/jose/Downloads/paisa/paisa.raw.utf8"

  // do some brief DS exploration, and preparation to get clean text
  val df = SparkAccessor.spark.read.text(paisaCorpusPath)

  val dataset = df.filter(!df("value").contains("</text")).
    filter(!df("value").contains("<text")).
    filter(!df("value").startsWith("#")).
    limit(10000)

  dataset.show(truncate=false)

  val names = List( "Achille", "Achillea", "Achilleo",  "Achillina", "Achiropita", "Acilia", "Acilio", "Acquisto",
  "Acrisio", "Ada", "Adalberta", "Adalberto", "Adalciso", "Adalgerio", "Adalgisa")

  import scala.collection.JavaConverters._
  val javaNames = new java.util.ArrayList[String](names.asJava)

  // ==================================
  // all the pipeline & training
  // ==================================
  val assembler = new DocumentAssembler()
  .setInputCol("value")
  .setOutputCol("document")

  val tokenizer = new RecursiveTokenizer()
  .setInputCols("document")
  .setOutputCol("token")
  .setPrefixes(Array("\"", "“", "(", "[", "\n", ".", "l’", "dell’", "nell’", "sull’", "all’", "d’", "un’"))
  .setSuffixes(Array("\"", "”", ".", ",", "?", ")", "]", "!", ";", ":"))

  val spellCheckerModel = new ContextSpellCheckerApproach()
    .setInputCols("token")
    .setOutputCol("checked")
    .addVocabClass("_NAME_", javaNames)
    .setLMClasses(1650)
    .setWordMaxDist(3)
    .setEpochs(2)

  val pipelineTok = new Pipeline().setStages(Array(assembler, tokenizer)).fit(dataset)
  val tokenized = pipelineTok.transform(dataset)
  spellCheckerModel.fit(tokenized).write.save("src/test/resources/spell/contextSpellCheckerModel")
  val spellChecker = ContextSpellCheckerModel.load("src/test/resources/spell/contextSpellCheckerModel")

}
