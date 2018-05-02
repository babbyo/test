import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.linalg.Vectors
import java.util.Scanner


object Test{
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("NORM")
      .master("local")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
//
//    val dataFrame = spark.createDataFrame(Seq(
//      (0, Vectors.dense(1.0, 0.5, -1.0)),
//      (1, Vectors.dense(2.0, 1.0, 1.0)),
//      (2, Vectors.dense(4.0, 10.0, 2.0))
//    )).toDF("id", "features")
//
//    // Normalize each Vector using $L^1$ norm.
//    val normalizer = new Normalizer()
//      .setInputCol("features")
//      .setOutputCol("normFeatures")
//      .setP(1.0)
//
//    val l1NormData = normalizer.transform(dataFrame)
//    println("Normalized using L^1 norm")

    val loadModel = RandomForestClassificationModel.load("/Users/liujinchen/Desktop/Qos/ml")
    println("RTT, PL, NACK, Plis, label:")
    val input = scala.io.StdIn.readLine()
    val line = new Scanner(input)
    val rtt = line.nextInt
    val pl = line.nextInt
    val nack = line.nextInt
    val plis = line.nextInt
    val label = line.nextInt
    //println(rtt, pl, nack , plis , label)

    val dataframe = spark.createDataFrame(Seq(
      (rtt, pl, nack, plis, label)
    )).toDF("rtt","pl","nack", "plis", "label")

    val assembler =  new VectorAssembler()
      .setInputCols(Array("rtt", "pl", "nack", "plis"))
      .setOutputCol("scaledFeatures")
    val inputDF = assembler.transform(dataframe)

//    val scaler = new StandardScaler()
//      .setInputCol("features")
//      .setOutputCol("scaledFeatures")
//      .setWithMean(true)
//      .setWithStd(true)
//      .fit(inputDF)
//    // Scale features using the scaler model
//    val normalizedDF = scaler.transform(inputDF)

    loadModel.transform(inputDF).toDF().show()



  }
}
