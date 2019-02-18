package com.johnsnowlabs.nlp

/**
  * Created by saif on 02/05/17.
  */
object ContentProvider {

  lazy val parquetData = SparkAccessor.spark.read.parquet("./src/test/resources/sentiment.parquet")

  val latinBody: String = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut " +
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut " +
    "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum " +
    "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui " +
    "officia deserunt mollit anim id est laborum."

  val englishPhrase: String = "In the third category he included those Brothers (the majority) " +
    "who saw nothing in Freemasonry but the external forms and ceremonies, and prized the strict " +
    "performance of these forms without troubling about their purport or significance."

  val sbdTestParagraph: String = "In the third category he included those Brothers (the majority) who saw nothing " +
    "in Freemasonry but the external forms and ceremonies, and prized the strict performance of these forms without " +
    "troubling about their purport or significance.@@ Such were Willarski and even the Grand Master of the principal " +
    "lodge.@@ Finally, to the fourth category also a great many Brothers belonged, particularly those who had lately " +
    "joined.@@ These according to Pierre's observations were men who had no belief in anything, nor desire for anything, " +
    "but joined the Freemasons merely to associate with the wealthy young Brothers who were influential through their" +
    " connections or rank, and of whom there were very many in the lodge.@@ Pierre began to feel dissatisfied with what" +
    " he was doing.@@ Freemasonry, at any rate as he saw it here, sometimes seemed to him based merely on externals.@@" +
    " He did not think of doubting Freemasonry itself, but suspected that Russian Masonry had taken a wrong path" +
    " and deviated from its original principles.@@ And so toward the end of the year he went abroad to be initiated" +
    " into the higher secrets of the order.@@ What is to be done in these circumstances?@@ To favor revolutions, overthrow" +
    " everything, repel force by force?@@ No!@@ We are very far from that.@@ Every violent reform deserves censure, for it" +
    " quite fails to remedy evil while men remain what they are, and also because wisdom needs no violence.@@ \"But" +
    " what is there in running across it like that?\" said Ilagin's groom.@@ \"Once she had missed it and turned it" +
    " away, any mongrel could take it,\" Ilagin was saying at the same time, breathless from his gallop and his" +
    " excitement."

  val wsjTrainingCorpus: String = "Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ ,|, will|MD " +
    "join|VB the|DT board|NN as|IN a|DT nonexecutive|JJ director|NN " +
    "Nov.|NNP 29|CD .|.\nMr.|NNP Vinken|NNP is|VBZ chairman|NN of|IN " +
    "Elsevier|NNP N.V.|NNP ,|, the|DT Dutch|NNP publishing|VBG " +
    "group|NN .|. Rudolph|NNP Agnew|NNP ,|, 55|CD years|NNS old|JJ " +
    "and|CC former|JJ chairman|NN of|IN Consolidated|NNP Gold|NNP " +
    "Fields|NNP PLC|NNP ,|, was|VBD named|VBN a|DT nonexecutive|JJ " +
    "director|NN of|IN this|DT British|JJ industrial|JJ conglomerate|NN " +
    ".|.\nA|DT form|NN of|IN asbestos|NN once|RB used|VBN to|TO make|VB " +
    "Kent|NNP cigarette|NN filters|NNS has|VBZ caused|VBN a|DT high|JJ " +
    "percentage|NN of|IN cancer|NN deaths|NNS among|IN a|DT group|NN " +
    "of|IN workers|NNS exposed|VBN to|TO it|PRP more|RBR than|IN " +
    "30|CD years|NNS ago|IN ,|, researchers|NNS reported|VBD .|."

  val targetSentencesFromWsj = Array("A form of asbestos once used to make " +
    "Kent cigarette filters has caused a high percentage of cancer deaths among a group " +
    "of workers exposed to it more than 30 years ago researchers reported")

  val nerSentence = "John Smith works at Airbus in Germany."

  val nerCorpus = """
                   |-DOCSTART- O
                   |
                   |John PER
                   |Smith PER
                   |works O
                   |at O
                   |Airbus ORG
                   |Germany LOC
                   |. O
                   |
                  """.stripMargin

  val depSentence = "One morning I shot an elephant in my pajamas. How he got into my pajamas I’ll never know."

  val conllEightSentences = "EU rejects German call to boycott British lamb.\n\nPeter Blackburn\n\nBRUSSELS 1996-08-22\n\nThe European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep.\n\nGermany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer.\n\n\" We don't support any such recommendation because we don't see any grounds for it , \" the Commission's chief spokesman Nikolaus van der Pas told a news briefing.\n\nHe said further scientific study was required and if it was found that action was needed it should be taken by the European Union.\n\nHe said a proposal last month by EU Farm Commissioner Franz Fischler to ban sheep brains , spleens and spinal cords from the human and animal food chains was a highly specific and precautionary move to protect human health.\n\n"
}
