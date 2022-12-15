import sbt._

object M2Resolvers {
  val m2CentralRepo = "Maven Central" at "https://repo1.maven.org/maven2/"

  val m2Resolvers: Seq[MavenRepository] = Seq(m2CentralRepo)
}
