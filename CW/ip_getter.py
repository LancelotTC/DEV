import requests
from api_tools import open_db
from time import sleep
from sqlite3 import IntegrityError

BUFFER = 60 * 1
while True:
	ip = requests.get('https://api.ipify.org').content.decode('utf8')
	with open_db("ip_addresses.db") as (db, connection):
		try:
			db.execute("insert into ip_addresses values (?)", (ip, ))
		except IntegrityError:
			pass
		connection.commit()
		print(ip)

	sleep(BUFFER)
