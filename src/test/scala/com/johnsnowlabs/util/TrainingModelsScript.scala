import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.pretrained.EnModelsTraining

// ToDo 1. Test Loading
// ToDo 2. Downlaod datasets before if needs
// ToDo 3. Use Build.version in sbt ?
// ToDo 4. Create Uploader

object TrainingModelsScript extends App {

  val training = new EnModelsTraining(SparkAccessor.spark)
  training.trainAllAndSave("models", true)
}