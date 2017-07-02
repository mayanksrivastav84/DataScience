sqoop import  -Dhadoop.security.credential.provider.path=jceks://hdfs/user/root/mysql.password.jceks \
--connect jdbc:mysql://10.37.129.7:3306/retail_db \
--username retail_dba \
--password-alias mydb.password.alias \
--table customers \
--num-mappers 10