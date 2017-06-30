# Databricks notebook source
# MAGIC %md https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# MAGIC 
# MAGIC Competition Description:
# MAGIC 
# MAGIC With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# MAGIC 
# MAGIC Practice Skills:
# MAGIC 
# MAGIC Creative feature engineering Advanced regression techniques like random forest and gradient boosting

# COMMAND ----------

# MAGIC %md download data sets from the Kaggle and import csv files 

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/tables/iz4qo8a01498148128281/

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC train = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/tables/iz4qo8a01498148128281/train.csv')
# MAGIC test = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/tables/iz4qo8a01498148128281/test.csv')
# MAGIC 
# MAGIC display(train)

# COMMAND ----------

train.registerTempTable("train")
test.registerTempTable("test")

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %md Checking the data formats for each columns

# COMMAND ----------

train.printSchema()

# COMMAND ----------

print(train.count())
print(test.count())

# COMMAND ----------

# MAGIC %python
# MAGIC list(train)

# COMMAND ----------

# MAGIC %md Data transformation, merging tables, and cleaning

# COMMAND ----------

# MAGIC %scala
# MAGIC val train = sqlContext.table("train")
# MAGIC val test = sqlContext.table("test")

# COMMAND ----------

# MAGIC %scala
# MAGIC sqlContext.cacheTable("train")
# MAGIC sqlContext.cacheTable("test")

# COMMAND ----------

# MAGIC %md Converting categorical variables from string to numeric forms. 

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.feature.StringIndexer 
# MAGIC 
# MAGIC val val1 = new StringIndexer().setInputCol("MSZoning").setOutputCol("MSZoning1") 
# MAGIC val val2 = new StringIndexer().setInputCol("Neighborhood").setOutputCol("Neighborhood1") 
# MAGIC val val3 = new StringIndexer().setInputCol("LotShape").setOutputCol("LotShape1") 
# MAGIC val val4 = new StringIndexer().setInputCol("LandContour").setOutputCol("LandContour1")
# MAGIC val val5 = new StringIndexer().setInputCol("LandSlope").setOutputCol("LandSlope1") 
# MAGIC val val6 = new StringIndexer().setInputCol("PavedDrive").setOutputCol("PavedDrive1")
# MAGIC val val7 = new StringIndexer().setInputCol("Condition2").setOutputCol("Condition21") 
# MAGIC val val8 = new StringIndexer().setInputCol("ExterCond").setOutputCol("ExterCond1")
# MAGIC val val9 = new StringIndexer().setInputCol("BsmtQual").setOutputCol("BsmtQual1") 
# MAGIC val val10 = new StringIndexer().setInputCol("KitchenQual").setOutputCol("KitchenQual1")
# MAGIC val val11 = new StringIndexer().setInputCol("RoofMatl").setOutputCol("RoofMatl1") 
# MAGIC val val12 = new StringIndexer().setInputCol("CentralAir").setOutputCol("CentralAir1")
# MAGIC val val13 = new StringIndexer().setInputCol("Functional").setOutputCol("Functional1")
# MAGIC val val14 = new StringIndexer().setInputCol("FireplaceQu").setOutputCol("FireplaceQu1")
# MAGIC val val15 = new StringIndexer().setInputCol("GarageType").setOutputCol("GarageType1") 
# MAGIC val val16 = new StringIndexer().setInputCol("GarageFinish").setOutputCol("GarageFinish1")
# MAGIC val val17 = new StringIndexer().setInputCol("GarageQual").setOutputCol("GarageQual1") 
# MAGIC val val18 = new StringIndexer().setInputCol("GarageCond").setOutputCol("GarageCond1")
# MAGIC val val19 = new StringIndexer().setInputCol("SaleCondition").setOutputCol("SaleCondition1") 
# MAGIC 
# MAGIC val train1 = val1.fit(train).transform(train)
# MAGIC val train2 = val2.fit(train1).transform(train1)
# MAGIC val train3 = val3.fit(train2).transform(train2)
# MAGIC val train4 = val4.fit(train3).transform(train3)
# MAGIC val train5 = val5.fit(train4).transform(train4)
# MAGIC val train6 = val6.fit(train5).transform(train5)
# MAGIC val train7 = val7.fit(train6).transform(train6)
# MAGIC val train8 = val8.fit(train7).transform(train7)
# MAGIC val train9 = val9.fit(train8).transform(train8)
# MAGIC val train10 = val10.fit(train9).transform(train9)
# MAGIC val train11 = val11.fit(train10).transform(train10)
# MAGIC val train12 = val12.fit(train11).transform(train11)
# MAGIC val train13 = val13.fit(train12).transform(train12)
# MAGIC val train14 = val14.fit(train13).transform(train13)
# MAGIC val train15 = val15.fit(train14).transform(train14)
# MAGIC val train16 = val16.fit(train15).transform(train15)
# MAGIC val train17 = val17.fit(train16).transform(train16)
# MAGIC val train18 = val18.fit(train17).transform(train17)
# MAGIC val train19 = val19.fit(train18).transform(train18)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC // indexed.show()
# MAGIC // indexed.printSchema()
# MAGIC display(train19)
# MAGIC train19.registerTempTable("trainCat")
# MAGIC 
# MAGIC // Qurious how the 'MSZoning' values below are not numerical but still showing the string values.

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %sql SELECT * FROM traincat limit 30

# COMMAND ----------

# MAGIC %sql
# MAGIC show columns from traincat

# COMMAND ----------

# MAGIC %md Creating a new table with variabels that are selected from a preliminary variable selection step.  The preliminary variable selection step is not included here.

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists cleaned_train;
# MAGIC 
# MAGIC create table cleaned_train as
# MAGIC select MSZoning1, Neighborhood1,  
# MAGIC LotShape1, LandContour1, LandSlope1, PavedDrive1,
# MAGIC Condition21, ExterCond1, BsmtQual1, KitchenQual1,
# MAGIC RoofMatl1, CentralAir1, FireplaceQu1,
# MAGIC GarageType1, GarageFinish1,  GarageQual1, GarageCond1,
# MAGIC SaleCondition1, Functional1,
# MAGIC -- # Numerical below
# MAGIC MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, 
# MAGIC BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, 1stFlrSF, 2ndFlrSF, GrLivArea,
# MAGIC BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, Fireplaces,
# MAGIC GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, 
# MAGIC PoolArea, YrSold, YearBuilt, YearRemodAdd, MoSold, SalePrice
# MAGIC from traincat

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %scala
# MAGIC val cleaned_train = sqlContext.table("cleaned_train")
# MAGIC display(cleaned_train.describe())

# COMMAND ----------

# MAGIC %md Trials to remove NA values from 'MasVnrArea' column

# COMMAND ----------

# MAGIC %scala
# MAGIC val cleaned_train2 = cleaned_train.na.fill(0)
# MAGIC cleaned_train2.registerTempTable("cleaned_train2")
# MAGIC display(cleaned_train2.describe())

# COMMAND ----------

# MAGIC %r # could not replace the NAs from MasVnrArea and this r fnc fails also
# MAGIC cleaned_train2[is.na(cleaned_train2)] <- 0

# COMMAND ----------

# MAGIC %md  NA removing tentative solution:  'MasVnrArea' variable is not selected from the above table 'cleaned_train'

# COMMAND ----------

# MAGIC %md We're going to have to put all of it into one column of a vector type for Spark MLLib. 
# MAGIC 
# MAGIC This makes it easy to embed a prediction right in a DataFrame and also makes it very clear as to what is getting passed into the model and what isn't without have to convert it to a numpy array or specify an R formula. 
# MAGIC 
# MAGIC This also makes it easy to incrementally add new features, simply by adding to the vector. 
# MAGIC 
# MAGIC In the below case rather than specifically adding them in, I'm going to create a exclusionary group and just remove what is NOT a feature.

# COMMAND ----------

# MAGIC %scala
# MAGIC val nonFeatureCols = Array("SalePrice")
# MAGIC val featureCols = cleaned_train2.columns.diff(nonFeatureCols)

# COMMAND ----------

# MAGIC %scala
# MAGIC cleaned_train2.printSchema()

# COMMAND ----------

# MAGIC %md Now I'm going to use the VectorAssembler in Apache Spark to Assemble all of these columns into one single vector. To do this I'll have to set the input columns and output column. Then I'll use that assembler to transform the prepped data to my final dataset.

# COMMAND ----------

# MAGIC %scala 
# MAGIC import org.apache.spark.ml.feature.VectorAssembler
# MAGIC 
# MAGIC val assembler = new VectorAssembler()
# MAGIC   .setInputCols(featureCols)
# MAGIC   .setOutputCol("features")
# MAGIC val finalPrep = assembler.transform(cleaned_train2)
# MAGIC 
# MAGIC val Array(training, testing) = finalPrep.randomSplit(Array(0.7, 0.3))
# MAGIC 
# MAGIC // Going to cache the data to make sure things stay snappy!
# MAGIC training.cache()
# MAGIC testing.cache()
# MAGIC 
# MAGIC println(training.count())
# MAGIC println(testing.count())

# COMMAND ----------

# MAGIC %scala display(finalPrep)

# COMMAND ----------

# MAGIC %scala training.select("MSZoning1").show()

# COMMAND ----------

Now we're going to get into the core of Apache Spark MLLib. At a high level, we're going to create an instance of a regressor or classifier, that in turn will then be trained and return a Model type. Whenever you access Spark MLLib you should be sure to import/train on the name of the algorithm you want as opposed to the Model type. 

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.ml.regression.LinearRegression
# MAGIC 
# MAGIC val lrModel = new LinearRegression()
# MAGIC   .setLabelCol("SalePrice")
# MAGIC   .setFeaturesCol("features")
# MAGIC   .setElasticNetParam(0.5)
# MAGIC 
# MAGIC println("Printing out the model Parameters:")
# MAGIC println("-"*20)
# MAGIC println(lrModel.explainParams)
# MAGIC println("-"*20)

# COMMAND ----------

# MAGIC %md Now finally we can go about fitting our model! You'll see that we're going to do this in a series of steps. First we'll fit it, then we'll use it to make predictions via the transform method. This is the same way you would make predictions with your model in the future however in this case we're using it to evaluate how our model is doing. We'll be using regression metrics to get some idea of how our model is performing, we'll then print out those values to be able to evaluate how it performs.

# COMMAND ----------

# MAGIC %scala 
# MAGIC import org.apache.spark.mllib.evaluation.RegressionMetrics
# MAGIC val lrFitted = lrModel.fit(training)

# COMMAND ----------

# MAGIC %md Prediction vs SalePrice

# COMMAND ----------

# MAGIC %scala
# MAGIC val holdout = lrFitted
# MAGIC   .transform(testing)
# MAGIC   .selectExpr("SalePrice as SalePrice",
# MAGIC   "prediction as Prediction")
# MAGIC display(holdout)

# COMMAND ----------

# MAGIC %scala
# MAGIC val rm = new RegressionMetrics(
# MAGIC   holdout.select("Prediction", "SalePrice").rdd.map(x =>
# MAGIC   (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
# MAGIC 
# MAGIC println("MSE: " + rm.meanSquaredError)
# MAGIC println("MAE: " + rm.meanAbsoluteError)
# MAGIC println("RMSE Squared: " + rm.rootMeanSquaredError)
# MAGIC println("R Squared: " + rm.r2)
# MAGIC println("Explained Variance: " + rm.explainedVariance + "\n")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 2nd method

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.regression.RandomForestRegressor
# MAGIC import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
# MAGIC 
# MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
# MAGIC 
# MAGIC import org.apache.spark.ml.{Pipeline, PipelineStage}
# MAGIC 
# MAGIC val rfModel = new RandomForestRegressor()
# MAGIC   .setLabelCol("SalePrice")
# MAGIC   .setFeaturesCol("features")
# MAGIC 
# MAGIC val paramGrid = new ParamGridBuilder()
# MAGIC   .addGrid(rfModel.maxDepth, Array(5, 10))
# MAGIC   .addGrid(rfModel.numTrees, Array(20, 60))
# MAGIC   .build()
# MAGIC // Note, that this parameter grid will take a long time
# MAGIC // to run in the community edition due to limited number
# MAGIC // of workers available! Be patient for it to run!
# MAGIC // If you want it to run faster, remove some of
# MAGIC // the above parameters and it'll speed right up!
# MAGIC 
# MAGIC val steps:Array[PipelineStage] = Array(rfModel)
# MAGIC 
# MAGIC val pipeline = new Pipeline().setStages(steps)
# MAGIC 
# MAGIC val cv = new CrossValidator() // you can feel free to change the number of folds used in cross validation as well
# MAGIC   .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
# MAGIC   .setEstimatorParamMaps(paramGrid)
# MAGIC   .setEvaluator(new RegressionEvaluator().setLabelCol("SalePrice"))
# MAGIC 
# MAGIC val pipelineFitted = cv.fit(training)

# COMMAND ----------

# MAGIC %md Now we've trained our model! Let's take a look at which version performed best!

# COMMAND ----------

# MAGIC %scala
# MAGIC println("The Best Parameters:\n--------------------")
# MAGIC println(pipelineFitted.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
# MAGIC pipelineFitted
# MAGIC   .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
# MAGIC   .stages(0)
# MAGIC   .extractParamMap

# COMMAND ----------

# MAGIC %scala
# MAGIC val holdout2 = pipelineFitted.bestModel
# MAGIC   .transform(testing)
# MAGIC   .selectExpr("SalePrice as SalePrice",
# MAGIC   "prediction as Prediction")
# MAGIC display(holdout2)

# COMMAND ----------


