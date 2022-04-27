"""
# Title : PySpark Script for Wine Quality Data Machine Learning Prediction
# Description : Wine Quality Data Machine Learning Prediction
# Author : Luis Hernandeez
# Date : 25-April-2022
# Version : 1.0 (Initial Draft)
# Usage : spark-submit --executor-memory 4G --executor-cores 4 PySpark_Script_Template.py > S3//luis4bucket logging and data
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from operator import add
import sys
## Constants
APP_NAME = "Wine Quality"
##OTHER FUNCTIONS/CLASSES

if __name__ == "__main__":
    """
        Usage: pa2_luis
    """
    spark = SparkSession\
        .builder\
        .appName("pa2_luis")\
        .getOrCreate()





sc  #spark context

sc.install_pypi_package("pandas==0.25.1") #Install pandas 



sc.list_packages() #packages available











import pandas
from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .appName ('Wine quality data prediction') \
        .getOrCreate()

trainingRawData = spark.read \
            .format('csv') \
            .option('delimiter',";") \
            .option('header', 'true') \
            .load('s3://luis4bucket/TrainingDataset.csv')

testingRawData = spark.read \
            .format('csv') \
            .option('delimiter',";") \
            .option('header', 'true') \
            .load('s3://luis4bucket/ValidationDataset.csv')


#drop invalid values
trainingRawData=trainingRawData.dropna(how='any')

#drop invalid values
testingRawData=testingRawData.dropna(how='any')

#rename columns
new_col_names = [ 'fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_ioide','density','pH','sulphates','alcohol','quality']
trainingRawData1= trainingRawData.toDF(*new_col_names)

trainingRawData2 = trainingRawData1.toPandas()





trainingRawData2.head()

trainingRawData2.columns

trainingRawData1.toPandas()['quality'].unique()

trainingRawData1.toPandas()['quality'].value_counts()





#trainingRawData2.count()







# change column names in testing data
testingRawData1=testingRawData.toDF(*new_col_names)

testingRawData2 = trainingRawData1.toPandas()

testingRawData2.head()

from pyspark.sql.types import FloatType
#change STRING data type to FLOAT datatype in the numeric field
trainingdataset = trainingRawData1
trainingdataset = trainingdataset.withColumn('fixed_acidity',
                                 trainingdataset['fixed_acidity'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('volatile_acidity',
                                 trainingdataset['volatile_acidity'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('citric_acid',
                                 trainingdataset['citric_acid'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('residual_sugar',
                                 trainingdataset['residual_sugar'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('chlorides',
                                 trainingdataset['chlorides'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('free_sulfur_dioxide',
                                 trainingdataset['free_sulfur_dioxide'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('total_sulfur_ioide',
                                 trainingdataset['total_sulfur_ioide'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('density',
                                 trainingdataset['density'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('pH',
                                 trainingdataset['pH'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('sulphates',
                                 trainingdataset['sulphates'].cast(FloatType()))
trainingdataset = trainingdataset.withColumn('alcohol',
                                 trainingdataset['alcohol'].cast(FloatType()))
                    
                    
 

from pyspark.sql.functions import col
testingdataset = testingRawData1.select(col('fixed_acidity').cast('float'),
                    col('volatile_acidity').cast('float'),
                    col('citric_acid').cast('float'),
                    col('residual_sugar').cast('float'),
                    col('chlorides').cast('float'),
                    col('free_sulfur_dioxide').cast('float'),
                    col('total_sulfur_ioide').cast('float'),
                    col('density').cast('float'),
                    col('pH').cast('float'),
                    col('sulphates').cast('float'),
                    col('alcohol').cast('float'),
                    col('quality')
                    )

#trainingdataset.describe()

trainingdataset.toPandas().head()

#testingdataset.describe()

testingdataset.toPandas()





#shuffle rows in the dataset
#from pyspark.sql.functions import rand
#trainingdataset=trainingdataset.orderBy(rand())


trainingdataset.toPandas().head()







from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

qualityIndexer = [StringIndexer(
    inputCol = 'quality', outputCol = 'quality_index')]

from pyspark.ml import Pipeline

pipeline = Pipeline(stages = qualityIndexer)

transformedDF = pipeline.fit(trainingdataset).transform(trainingdataset)
transformedDF.toPandas().head()

requiredFeatures = ['fixed_acidity',
                    'volatile_acidity',
                    'citric_acid',
                    'residual_sugar',
                    'chlorides',
                    'free_sulfur_dioxide',
                    'total_sulfur_ioide',
                    'density',
                    'pH',
                    'sulphates',
                    'alcohol'
                   ]

features = ['fixed_acidity',
                    'volatile_acidity',
                    'citric_acid',
                    'residual_sugar',
                    'chlorides',
                    'free_sulfur_dioxide',
                    'total_sulfur_ioide',
                    'density',
                    'pH',
                    'sulphates',
                    'alcohol'
                   ]

assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')

transformedDF = assembler.transform(transformedDF)
transformedDF.toPandas().head()

transformedDF.select('features').toPandas().head()

rf = RandomForestClassifier(labelCol='quality_index',
                            featuresCol='features',
                            maxDepth=24)
                            

pipeline=Pipeline(
        stages = qualityIndexer + [assembler,rf]
)











model = pipeline.fit(trainingdataset)

predictions = model.transform(testingdataset)
predictionsDF=predictions.toPandas()
predictionsDF.head()

predictions=predictions.select(
    'quality_index',
    'prediction'
)
    

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='quality_index',
    predictionCol='prediction',
    metricName='accuracy')



#evaluare the accuracy of the model
accuracy=evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)

from pyspark.mllib.evaluation import MulticlassMetrics
predictionAndLabels = predictions.select("prediction","quality_index").rdd
# Instantiate metrics objects
multi_metrics = MulticlassMetrics(predictionAndLabels)
precision_score = multi_metrics.weightedPrecision
recall_score = multi_metrics.weightedRecall
f_measure = multi_metrics.fMeasure








#predictionAndLabels = prediction.select("prediction","label").rdd
# Instantiate metrics objects
#multi_metrics = MulticlassMetrics(predictionAndLabels)
#precision_score = multi_metrics.weightedPrecision
#recall_score = multi_metrics.weightedRecall


#confusion_matrix = multi_metrics.confusionMatrix().toArray()



#multi_metrics = MulticlassMetrics(rdd)
#print ('fMeasure: ', multi_metrics.fMeasure(1.0,1.0))

print("precision_score", precision_score)
print("recall_score", recall_score)
print("f_Measure for quality 5", multi_metrics.fMeasure(0.0))
print("f_Measure for quality 6", multi_metrics.fMeasure(1.0))
print("f_Measure for quality 7", multi_metrics.fMeasure(2.0))
print("f_Measure for quality 4", multi_metrics.fMeasure(3.0))
print("f_Measure for quality 8", multi_metrics.fMeasure(4.0))
print("f_Measure for quality 3", multi_metrics.fMeasure(5.0))
print("weighted_f_Measeure", multi_metrics.weightedFMeasure())

import pyspark.sql.functions as F

printdf = sqlContext.createDataFrame([("precision_score", precision_score),("recall_score", recall_score),("f_Measure for quality 5", multi_metrics.fMeasure(0.0)),
                                     ("f_Measure for quality 6", multi_metrics.fMeasure(1.0)),("f_Measure for quality 7", multi_metrics.fMeasure(2.0)),
                                     ("f_Measure for quality 4", multi_metrics.fMeasure(3.0)),("f_Measure for quality 8", multi_metrics.fMeasure(4.0)),
                                     ("f_Measure for quality 3", multi_metrics.fMeasure(5.0)),("weighted_f_Measeure", multi_metrics.weightedFMeasure())],['Measure','Score']) 
printdf.show() 


###printdf = printdf.withColumn('Score', F.round(f.col('em'), 3))



#printdf.withColumn("Score",printdf.Score.cast('string'))



printdf = printdf.withColumn("Score", F.round(printdf["Score"], 6)).withColumnRenamed("Score","Score")

printdf.show()


printdf.repartition(1).write.mode('overwrite').csv("s3://luis4bucket/data.csv")

#printdf.toPandas().to_csv('mycsv.csv')
#printdf.write.csv('mycsv.csv')

spark.stop()
























































































