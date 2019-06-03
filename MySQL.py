import mysql.connector


mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="lewis",
  passwd="Aa1995841214",
  database='mydatabase'
)


mycursor = mydb.cursor()

mycursor.execute("SHOW TABLES")

for x in mycursor:
  print(x)