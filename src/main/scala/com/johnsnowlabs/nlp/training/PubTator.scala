package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DocumentAssembler}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper, ReadAs}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, SparkSession}
import java.io._

import scala.collection.mutable.ArrayBuffer

/*
PubTator document
Each paper or document ends with a blank line, and is represented as (without the spaces):

PMID | t | Title text
PMID | a | Abstract text
PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID

25763772|t|DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis
25763772|a|Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with worse long-term pulmonary disease and shorter survival, and chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline, increased rates of exacerbations and shorter survival. By using exome sequencing and extreme phenotype design, it was recently shown that isoforms of dynactin 4 (DCTN4) may influence Pa infection in CF, leading to worse respiratory disease. The purpose of this study was to investigate the role of DCTN4 missense variants on Pa infection incidence, age at first Pa infection and chronic Pa infection incidence in a cohort of adult CF patients from a single centre. Polymerase chain reaction and direct sequencing were used to screen DNA samples for DCTN4 variants. A total of 121 adult CF patients from the Cochin Hospital CF centre have been included, all of them carrying two CFTR defects: 103 developed at least 1 pulmonary infection with Pa, and 68 patients of them had CPA. DCTN4 variants were identified in 24% (29/121) CF patients with Pa infection and in only 17% (3/18) CF patients with no Pa infection. Of the patients with CPA, 29% (20/68) had DCTN4 missense variants vs 23% (8/35) in patients without CPA. Interestingly, p.Tyr263Cys tend to be more frequently observed in CF patients with CPA than in patients without CPA (4/68 vs 0/35), and DCTN4 missense variants tend to be more frequent in male CF patients with CPA bearing two class II mutations than in male CF patients without CPA bearing two class II mutations (P = 0.06). Our observations reinforce that DCTN4 missense variants, especially p.Tyr263Cys, may be involved in the pathogenesis of CPA in male CF.
25763772        0       5       DCTN4   T116,T123    C4308010
25763772        23      63      chronic Pseudomonas aeruginosa infection        T047    C0854135
25763772        67      82      cystic fibrosis T047    C0010674
25763772        83      120     Pseudomonas aeruginosa (Pa) infection   T047    C0854135


as output, we want something like
text | document | chunk/segment | NER(semantictpeID(comma-separated list of lists)) | entityID | PMID

NER (semantictypeID) col is something like [[B-T116,B-T123], start->0, end->5]
go word-by-word, assign "O" to words not in segments


parsing notes: document is title + "/n" or " " + abstract
how to semantictypeID and entityID match up to POS/NER?
semantictypeID is literally NER type, will just have to be custom-mapped for each 130 semantic types to desired NER types


 */

//                              # of chunks does not match # of ners
//pubtator document has text | [[chunk start, chunk end],] | [[word, word start, word end], ] | [[ner, ner start, ner end, ner], ] | [entityID, ], PMID
case class PubTatorDocument(text: String,
                            typeIDTagged: Seq[NerTaggedSentence],
                            entityIDTagged: Seq[NerTaggedSentence],//TODO: change this type to tagged resolution sentence?
                            PMID: String
                        )

case class PubTator (pubTatorTextCol: String = "text",
                     documentCol: String = "document",
                     segmentCol: String = "chunk",
                     typeIDCol: String = "typeID",
                     entityIDCol: String = "entityID",
                     PMIDCol: String = "pmid"
                    //TODO: define NER mapping type?
                    //or add functions to define and map?
                    ){

  def readDataset(spark: SparkSession,
                  path: String,
                  readAs: String = ReadAs.TEXT.toString
                 ) : Unit={//TODO: should be Dataset[_]={
    val er = ExternalResource(path, readAs, Map("format" -> "text"))
    val lines = ResourceHelper.parseLines(er)
    readLines(lines)
    //packDocs(readLines(lines), spark)
  }

  def readLines(lines: Array[String]): Unit={ //TODO: should be Seq[PubTatorDocument] = {
  //TODO: read lines, return pubtatordocument that can be read by packdocs
    //convert rows of the PubTator file into rows of the annotated DataFrame

    var rows = ArrayBuffer[PubTatorDocument]() //arraybuffer to hold all the rows
    var i=0 //line counter
    var line=""
    var allDocsNers = ArrayBuffer[Array[Tuple6[String, Int, Int, String, String, String]]]()
    while (i < lines.length ){
      //parse a new document (pubmed paper)
      var title=lines(i)
      if(!title.matches(".+[|]t[|].+")){print(i)}//title)}
      assert(title.matches(".+[|]t[|].+")) //first line should be a title
      i += 1
      var abstr = lines(i)
      assert(abstr.matches(".+[|]a[|].+")) //second line should be an abstract
      i += 1
      line=lines(i)
      var fullText = title + abstr
      var actualNERs = ArrayBuffer[Tuple6[String, Int, Int, String, String, String]]() //word, start, end, ner, entitytype, PMID
      //Assemble list of words, NERs found
      while(line != ""){ //Should it be "" or "/n" ?
        //parse
        val elements = line.split('\t')
        val PMID= elements(0)
        val segStart = elements(1).toInt
        val segEnd = elements(2).toInt
        val seg = elements(3)
        var typeID = "" //TODO: for now, we only get the first typeID in the CSV
        if (elements(4).contains(",")) { typeID = elements(4).split(',')(0) }
        else{ typeID = elements(4) }
        val entityID = elements(5)

        val wordsInLine = seg.split(" ")
        var j = segStart
        for (word <- wordsInLine){
          val wordStart = j
          j += word.length
          val wordEnd = j
          j+=1 //space between words
          var NERPrefix = ""
          if (wordStart == segStart){ //if it's the first word in the chunk
            NERPrefix = "B-"
          }
          else{NERPrefix = "I-"}
          val fullNER = NERPrefix + typeID
          actualNERs.append((word, wordStart, wordEnd, fullNER, entityID, PMID))
        }
        //if (i == lines.length){break}??
        i += 1
        line=lines(i)
      }
      print(line)
      i += 1 //do not read blank line
      line=lines(i)


      //all lines for this pubmed doc have been parsed, words in the chunks have been added.
      //time to add the words not mentioned in the chunks
      var words = fullText.split(" ") //TODO: this ignores periods, punctuation, etc.. In the future, use a tokenizer.
      var j:Int = 0 //current cursor
      var notYetAdded = ArrayBuffer[(String, Int, Int, String, String, String)]() //word, start, end, ner, entitytype, PMID
      val PMID = actualNERs(0)._6
      for (word <- words){
        var stringStart = j
        j += word.length
        val stringEnd = j
        j += 1 //for the space
        var wordAlreadyAdded = false
        for (tup <- actualNERs){
          if( (tup._1 == stringStart) && ((tup._2 - stringEnd).abs <= 2) ){
            wordAlreadyAdded = true
          }
        if (!wordAlreadyAdded){
          notYetAdded.append((word, stringStart, stringEnd, "O", "O", PMID))
        }

      }
    }

      //all words in the pubmed document have been parsed
      //convert to PubTatorDocument
      var allNer = actualNERs ++ notYetAdded
      //sort by word order (compare the start values)
      val sortedNer = scala.util.Sorting.stableSort(allNer,
        (e1: (String, Int, Int, String, String, String), e2: (String, Int, Int, String, String, String)) => e1._2 < e2._2).toArray
      allDocsNers.append(sortedNer)





      /*
      var typeIDTagged = ArrayBuffer[NerTaggedSentence]
      for (tup <- allNer){
      }*/
    }





    //temporary solution: write to CoNLL file (will have an extra
    new File("./pubtator-conll.txt").delete()
    val file = new File("./pubtator-conll.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("-DOCSTART- -X- -X- O\n")
    for (listForDoc <- allDocsNers) {
      bw.write("\n")
      for (nerWord <- listForDoc) {
        bw.write("%s O O %s\n".format(nerWord._1, nerWord._4))
      }
    }
    bw.close()



    //create tagged words, tagged sentences
    ""
    //Seq(PubTatorDocument("", Seq(""), Seq(Seq("")), "" ))
  }

  def packDocs(docs: Seq[PubTatorDocument], spark: SparkSession): Dataset[_] = {
    import spark.implicits._
    //TODO: put in dataframe
    val rows = docs.map { doc =>
      val text = doc.text
      val docs = new DocumentAssembler().assemble(text, Map("training" -> true.toString))
      val chunks = ""
      val tokens = ""
      val ners = ""
      val semanticTypeIDs = ""
      val entityIDs =""


      (text, docs, chunks, semanticTypeIDs, entityIDs)
    }.toDF.rdd

    val annotationType = ArrayType(Annotation.dataType)

    def getAnnotationType(column: String, annotatorType: String, addMetadata: Boolean = true): StructField = {
      if (!addMetadata)
        StructField(column, annotationType, nullable = false)
      else {
        val metadataBuilder: MetadataBuilder = new MetadataBuilder()
        metadataBuilder.putString("annotatorType", annotatorType)
        StructField(column, annotationType, nullable = false, metadataBuilder.build)
      }
    }

    def schema: StructType = {
      //TODO: Do we want metadata?
      val text = StructField(pubTatorTextCol, StringType)
      val doc = getAnnotationType(documentCol, AnnotatorType.DOCUMENT)
      val segment = getAnnotationType(segmentCol, AnnotatorType.CHUNK)
      val typeID = StructField(typeIDCol, ArrayType(StringType))
      val entityID = StructField(entityIDCol, StringType)
      StructType(Seq(text, doc, segment, typeID, entityID))
    }


    spark.createDataFrame(rows, schema)
  }

/*
  def getWords(fullText: String): ArrayBuffer[ Tuple4(String, Int, Int, String) ] ={
    //TODO: add something like: if word start matches and word end is +- 1, change the NER

    var words = fullText.split(" ") //TODO: this ignores periods, punctuation, etc.. In the future, use a tokenizer.
    var wordMeta = ArrayBuffer[Tuple3(String, Int, Int, String)]
    var j:Int = 0 //current cursor???
    for (word <- words){
      var stringStart = j
      j += word.length
      val stringEnd = j
      j += 1 //for the space???
      wordMeta.append((word, stringStart, stringEnd, "O"))
    }
   wordMeta
  }*/


}
