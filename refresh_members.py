from api_tools import open_db, get_clan_members
import sqlite3


clan_tag = "2R00CGYR0"

members = [(member["tag"], member["name"]) for member in get_clan_members(clan_tag)["items"]]
print(members)

with open_db(r"C:\Users\lance.000\Downloads\contributions.db") as (db, connection):
	request = ""
	for tag, name in members:
		try:
			db.execute("insert into member_contributions (tag, name) values (?, ?);", (tag, name))
		except sqlite3.IntegrityError:
			pass
	connection.commit()

# with open("response.json", "w") as file:
# 	json.dump(response, file)
