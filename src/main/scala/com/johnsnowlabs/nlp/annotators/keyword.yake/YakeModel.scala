package com.johnsnowlabs.nlp.annotators.keyword.yake

import com.johnsnowlabs.nlp.annotators.keyword.yake.util.Utilities.{getTag, medianCalculator}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap
import scala.collection.{immutable, mutable}
import scala.collection.mutable.ListBuffer
import com.johnsnowlabs.nlp.AnnotatorType.{KEYWORD, TOKEN}
import com.johnsnowlabs.nlp.annotators.keyword.yake.util.Token

import scala.math.sqrt

/**
  * Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.
  * Extracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a plethora of situations where access to training corpora is either limited or restricted.
  * The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be first sent through a Sentence Boundary Detector and then a tokenizer.
  * You can tweak the following parameters to get the best result from the annotator.
  *
  * setMinNGrams(int) Select the minimum length of a extracted keyword
  * setMaxNGrams(int) Select the maximum length of a extracted keyword
  * setNKeywords(int) Extract the top N keywords
  * setStopWords(list) Set the list of stop words
  * setThreshold(float) Each keyword will be given a keyword score greater than 0. (Lower the score better the keyword) Set an upper bound for the keyword score from this method.
  * setWindowSize(int) Yake will construct a co-occurence matrix. You can set the window size for the cooccurence matrix construction from this method. ex: windowSize=2 will look at two words to both left and right of a candidate word.
  *
  *
  *
  * See Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289.
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/keyword/yake/YakeTestSpec.scala]] for further reference on how to use this API.
  * Sources:
  *
  *
  * '''Sources''' :
  *
  * [[https://www.sciencedirect.com/science/article/pii/S0020025519308588]]
  *
  * ''' Paper abstract '''
  * As the amount of generated information grows, reading and summarizing texts of large collections turns into a challenging task. Many documents do not come with descriptive terms,
  * thus requiring humans to generate keywords on-the-fly. The need to automate this kind of task demands the development of keyword extraction systems with the ability to automatically
  * identify keywords within the text. One approach is to resort to machine-learning algorithms. These, however, depend on large annotated text corpora, which are not always available.
  * An alternative solution is to consider an unsupervised approach. In this article, we describe YAKE!, a light-weight unsupervised automatic keyword extraction method which rests on
  * statistical text features extracted from single documents to select the most relevant keywords of a text. Our system does not need to be trained on a particular set of documents,
  * nor does it depend on dictionaries, external corpora, text size, language, or domain. To demonstrate the merits and significance of YAKE!, we compare it against ten state-of-the-art
  * unsupervised approaches and one supervised method. Experimental results carried out on top of twenty datasets show that YAKE! significantly outperforms other unsupervised methods on
  * texts of different sizes, languages, and domains.
  *
  *
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class YakeModel(override val uid: String) extends AnnotatorModel[YakeModel] with HasSimpleAnnotate[YakeModel] with YakeParams {

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */

  def this() = this(Identifiable.randomUID("YAKE"))

  private val logger = LoggerFactory.getLogger("YakeModel")

  override val outputAnnotatorType: AnnotatorType = KEYWORD
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  setDefault(
    maxNGrams -> 3,
    minNGrams -> 1,
    nKeywords -> 30,
    windowSize -> 3,
    threshold -> -1,
    stopWords -> StopWordsRemover.loadDefaultStopWords("english")
  )

  /**
    * Calculates basic statistics like total Sentences in the document and assign a tag for each token
    * @param result Document to annotate as array of tokens with sentence metadata
    * @return Dataframe with columns SentenceID, token, totalSentences, tag
    */
  def getBasicStats(result: Array[Annotation]): Array[(String,Int)] = {
    val resultFlatten = result.map(x => (x.result, x.metadata.head._2))
    val resultFlattenIndexed = resultFlatten.map { row => (row._1, row._2.toInt) }
    resultFlattenIndexed
  }

  def assignTags(resultFlattenIndexed: Array[(String,Int)]): Array[(String, Int, Int, String)] = {
    var sentenceID = 0
    var position = 0
    var tags = resultFlattenIndexed.map{case (t,sID) =>
      if (sID.toInt == sentenceID) {
        val tag = getTag(t,position)
        position += 1
        (t,sentenceID,position,tag)
      }
      else {
        sentenceID += 1
        position = 0
        val tag = getTag(t,position)
        position += 1
        (t,sentenceID,position,tag)
      }
    }
    tags
  }

  /**
    * Calculate Co Occurrence for left to right given a window size
    * @param sentences DataFrame with tokens
    * @return Co Occurrence for token x from left to right as a Map
    */
  def getCoOccurrence(sentences: ListBuffer[ListBuffer[String]], left: Boolean): mutable.Map[String,mutable.Map[String,Int]] = {
    var coMap: mutable.Map[String, mutable.Map[String, Int]] = mutable.HashMap[String,mutable.Map[String,Int]]()
    val ngrams = sentences.zipWithIndex.flatMap { case (row, index) =>
      (for (i <- $(minNGrams) to $(maxNGrams)) yield row.sliding(i).map(p => p.toList)).flatten
    }
    ngrams.foreach{elem =>{
      var head = elem.head.toLowerCase
      if (!left) {
        head = elem.last.toLowerCase
      }
      elem.foreach(x=>{
        if (x.toLowerCase!=head){
          coMap.getOrElseUpdate(head,mutable.HashMap[String,Int]())
          coMap(head).getOrElseUpdate(x.toLowerCase,0)
          coMap(head)(x.toLowerCase) += 1
        }
      })
    }
    }
    coMap
  }

  /**
    * Calculate token scores given statistics
    *
    *  Refer [[https://doi.org/10.1016/j.ins.2019.09.013 YAKE Paper]]
    *
    * T Position = ln ( ln ( 3 + Median(Sentence Index))
    * T Case =  max(TF(U(t)) , TF(A(t))) / ln(TF(t))
    * TF Norm =TF(t) / (MeanTF + 1 ∗ σ)
    * T Rel = 1 + ( DL + DR ) * TF(t)/MaxTF
    * T Sentence = SF(t)/# Sentences
    * TS = ( TPos ∗ TRel ) / ( TCase + (( TFNorm + TSent ) / TRel ))
    *
    * @param basicStats Basic stats
    * @param coOccurLeftAggregate Left Co Occurrence
    * @param coOccurRightAggregate Right Co Occurrence
    * @return
    */
  def calculateTokenScores(basicStats: Array[(String,Int)],
                           coOccurLeftAggregate: mutable.Map[String,mutable.Map[String,Int]],
                           coOccurRightAggregate: mutable.Map[String,mutable.Map[String,Int]]): immutable.Iterable[Token] = {
    val tags = assignTags(basicStats)
    val avg = basicStats
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(_.length)
      .foldLeft(0)(_+_._2).toDouble/
      basicStats
        .groupBy(x=>x._1.toLowerCase)
        .mapValues(_.length).size.toDouble
    val std = sqrt(basicStats
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(_.length)
      .map(_._2.toDouble)
      .map(a => math.pow(a - avg, 2)).sum /
      basicStats
        .groupBy(x=>x._1.toLowerCase)
        .mapValues(_.length).size.toDouble)
    val maxTF = basicStats.groupBy(x=>x._1.toLowerCase).mapValues(_.length).map(_._2.toDouble).max
    val nsent = basicStats.map(_._2).max +1
    val tokens=basicStats
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(_.length)
      .map(x=>
        new Token(x._1.toLowerCase,
          x._2,
          nsent,
          avg,
          std,
          maxTF,
          coOccurLeftAggregate.getOrElse(x._1.toLowerCase,mutable.HashMap[String,Int]()),
          coOccurRightAggregate.getOrElse(x._1.toLowerCase,mutable.HashMap[String,Int]())
        )
      )
    tags
      .filter(x=>x._4=="n")
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(_.length)
      .foreach(x=>tokens.filter(y=>y.token==x._1).head.nCount=x._2)
    tags
      .filter(x=>x._4=="a")
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(_.length)
      .foreach(x=>tokens.filter(y=>y.token==x._1).head.aCount=x._2)
    tags
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(x=>medianCalculator(x.map(y=>y._2)))
      .foreach(x=>tokens.filter(y=>y.token==x._1).head.medianSentenceOffset=x._2)
    tags
      .groupBy(x=>x._1.toLowerCase)
      .mapValues(x=>x.map(y => y._2).length)
      .foreach(x=>tokens.filter(y=>y.token==x._1).head.numberOfSentences=x._2)
    tokens
  }

  /**
    * Separate sentences given tokens with sentence metadata
    * @param tokenizedArray Tokens with sentence metadata
    * @return separated sentences
    */
  def getSentences(tokenizedArray: Array[Annotation]): ListBuffer[ListBuffer[String]] = {
    var sentences: ListBuffer[ListBuffer[String]] = ListBuffer(ListBuffer())
    var snt = 0
    tokenizedArray.map(x=>{
      if(x.metadata.getOrElse("sentence",null).toInt == snt) {
        sentences(snt) += x.result.toLowerCase
      }else {
        snt += 1
        sentences += ListBuffer()
        sentences(snt) += x.result.toLowerCase
      }})
    sentences
  }

  /**
    * Generate candidate keywords
    * @param sentences sentences as a list
    * @return candidate keywords
    */
  def getCandidateKeywords(sentences:Array[(String, Int, Int, String)]):mutable.Map[String, Int] = {
    var candidate = mutable.HashMap[String, Int]().withDefaultValue(0)
    sentences.groupBy(_._2).map(row=>{
      val ngrams = (for (i <- $(minNGrams) to $(maxNGrams)) yield row._2.sliding(i).map(p => p.toList)).flatten
      ngrams.filter(y=>(!y.map(x=>x._4).contains("u")) && (!y.map(x=>x._4).contains("d"))).map(x=>{
        val firstWord = x.head._1.toLowerCase
        val lastWord =  x.last._1.toLowerCase
        if(!$(stopWords).contains(firstWord) && !$(stopWords).contains(lastWord) ){
          candidate(x.map(_._1).mkString(",").toLowerCase) += 1
        }
      })
    })
    candidate
  }

  /**
    * Extract keywords
    * @param candidate candidate keywords
    * @param tokens tokens with scores
    * @return keywords
    */
  def getKeywords(candidate: mutable.Map[String, Int],tokens:immutable.Iterable[Token]): ListMap[String, Double] = {
    val keywords = candidate.map{case (x,kf) =>
      var prod_s:Double = 1
      var sum_s:Double = 0
      val xi = x.split(",")
      xi.zipWithIndex.foreach{case (y,ind) =>
        val word = tokens.filter(k=>k.token==y)
        if(!$(stopWords).contains(y) && word.nonEmpty){
          prod_s *= word.head.TScore
          sum_s+=word.head.TScore
        }
        else{
          val prev_token = tokens.filter(k => k.token == xi(ind - 1))
          var prev=0.0
          var prev_prob=0.0
          if(prev_token.nonEmpty){
            prev = prev_token.head.rightCO.getOrElse(y,0).toDouble
            prev_prob = prev /prev_token.head.termFrequency
          }
          val next_token = tokens.filter(k => k.token == y)
          var next=0.0
          var next_prob =0.0
          if(next_token.nonEmpty){
            next = next_token.head.rightCO.getOrElse(xi(ind+1),0).toDouble
            next_prob = next /next_token.head.termFrequency
          }
          val bi_probability = prev_prob * next_prob
          prod_s = prod_s * (1+(1-bi_probability))
          sum_s -= (1-bi_probability)
        }
      }
      val S_kw = prod_s/(kf*(1+sum_s))
      (xi.mkString(" ").toLowerCase,S_kw)
    }
    var topn = ListMap(keywords.toSeq.sortWith(_._2 < _._2):_*)
    topn = topn.slice(0,$(nKeywords))
    if ($(threshold) != -1){
      topn = topn.filter{case (_, score)=> score <= $(threshold)}
    }
    topn
  }

  /**
    * Execute the YAKE algorithm for each sentence
    * @param annotations token array to annotate
    * @return annotated token array
    */
  def processSentences(annotations:Seq[Annotation]):Seq[Annotation] = {
    val basicStat = getBasicStats(annotations.toArray)
    val sentences = getSentences(annotations.toArray)
    val coOccurMatLeft = getCoOccurrence(sentences,left = true)
    val coOccurMatRight = getCoOccurrence(sentences,left = false)
    val tokens = calculateTokenScores(basicStat,coOccurMatLeft,coOccurMatRight)
    val taggedSentence = assignTags(basicStat)
    val candidateKeywords = getCandidateKeywords(taggedSentence)
    val keywords = getKeywords(candidateKeywords,tokens)
    var annotatedKeywords:ListBuffer[Annotation] = new ListBuffer()
    val annotationNGram = (for (i <- $(minNGrams) to $(maxNGrams)) yield annotations.sliding(i).map(p => p.toList)).flatten
    annotationNGram.foreach(
      annotation => {
        val key :String = annotation.map(_.result.toLowerCase()).mkString(" ").toLowerCase
        if (keywords.isDefinedAt(key)) {
          annotatedKeywords += Annotation(
            outputAnnotatorType,
            annotation.head.begin,
            annotation.last.end,
            key.toString,
            Map("score" -> keywords.getOrElse(key,"").toString,"sentence"->annotation.head.metadata.getOrElse("sentence",0).toString)
          )
        }
      }
    )
    annotatedKeywords.toSeq
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val keywords = processSentences(annotations)
    keywords
  }
}
object YakeModel extends ParamsAndFeaturesReadable[YakeModel]