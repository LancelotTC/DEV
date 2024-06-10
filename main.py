from tools import flatten, ProgressBar
from tools.logger import Logger
from api_tools import get_player, open_db
from datetime import datetime
from time import sleep
from _tools import RecurringTasks


# from concurrent.futures import ThreadPoolExecutor, as_completed

logger = Logger("logs.log")

def get_contributions():
	with open_db() as (db, connection):
		tags: list[str] = flatten(db.execute("select tag from member_contributions").fetchall())

		# with ThreadPoolExecutor(max_workers=25) as pool:
		# 	results = [pool.submit(get_player, tag) for tag in tags]

		for tag in tags:
				player = get_player(tag)

			# for future in as_completed(results):
			# 	player = future.result()

				try:
					tag = player["tag"]
				except KeyError:
					print("Invalid IP address")
					quit()
				name = player["name"]
				new_contribution = player["clanCapitalContributions"]

				old_contribution = \
					flatten(db.execute("select contribution from member_contributions where tag = ?", (tag, )).fetchone())[0]
				# print(old_contribution)
				if old_contribution is None:
					old_contribution = 0

				if old_contribution != new_contribution:
					db.execute(
						"insert into contribution_logs values (?, ?, ?, ?)",
						(tag, name, new_contribution - old_contribution, datetime.now())
					)

					db.execute(
						"update member_contributions set contribution = ? where tag = ?",
						(new_contribution, tag)
					)

					logger.info(f"{name} contributed {new_contribution - old_contribution}")

				connection.commit()


rt = RecurringTasks()
rt.add_task(get_contributions, 60 * 5, "Contributions")
# rt.add_task(get_contributions, 60 * 5, "Contributions")
rt.start()


