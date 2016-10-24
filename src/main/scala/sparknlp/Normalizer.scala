package sparknlp

/**
  * Created by alext on 10/23/16.
  */
class Normalizer() extends Annotator {
  override val aType: String = "ntoken"

  override def annotate(
    document: Document, annos: Seq[Annotation]
  ): Seq[Annotation] =
    annos.collect {
      case token: Annotation if token.aType == "stem" =>
        val ntoken = document.text.substring(token.begin, token.end).toLowerCase
          .replaceAll("[^a-zA-Z]", " ").trim
        Annotation(aType, token.begin, token.end, Map(aType -> ntoken))
    }.filter(_.metadata("ntoken").nonEmpty)

  override val requiredAnnotationTypes = Seq("stem")
}
