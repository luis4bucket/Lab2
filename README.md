# Lab2
Lab 2 requirements
Training dataset s3://luis4bucket/TrainingDataset.csv
Testing dataset  s3://luis4bucket/ValidationDataset.csv
This PySpark script uses a RandomForest classificationalgorithm to predict the wine quality on a dataset
runs in a Jupyter notebook in an EMR cluster from AWS
place the result scores in s3://luis4bucket/data.csv
pull the image docker run -it luis4docker/school
run the aws commands as follows
apt-get install awscli -y -11 -5 #install aws cli
#copy file credentials to /.aws/credentials
run 

aws emr create-cluster --auto-scaling-role EMR_AutoScaling_DefaultRole --termination-protected --applications Name=Hadoop Name=Hive Name=Pig Name=Hue Name=JupyterHub Name=JupyterEnterpriseGateway Name=Spark --ebs-root-volume-size 10 --ec2-attributes '{"InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-0616743c287267acd","EmrManagedSlaveSecurityGroup":"sg-08fa40b3db4548856","EmrManagedMasterSecurityGroup":"sg-0114e9293c01ded4f"}' --service-role EMR_DefaultRole --enable-debugging --release-label emr-6.5.0 --log-uri 's3n://luis4bucket/' --name 'LCHC Final cluster' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master - 1"},{"InstanceCount":3,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core - 2"}]' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1

generate a notebook with the data in the script and obtain the data.csv for the scores
