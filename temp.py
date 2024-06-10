from _tools import sep
from itertools import cycle
from json import JSONEncoder
from functools import cmp_to_key
# from itertools import groupby
import attrs
from api_tools import _api_request, get_key
import requests
import json
import asyncio
import time

import aiohttp
from aiohttp import ClientOSError, ServerDisconnectedError

from _tools import Logger
from typing import Optional
from typing import Optional, Iterable, Callable

logger = Logger(filemode="w")


@attrs.define
class Member:
	tag: str
	name: str
	townhall_level: Optional[int] = None
	map_position: Optional[int] = None
	total_stars: Optional[int] = None
	total_attacks: Optional[int] = None
	total_chances: Optional[int] = None
	total_destruction: Optional[int] = None

	@property
	def activity_rate(self):
		return self.total_attacks/self.total_chances

@attrs.define
class War:
	war_tag: str
	members: list[Member]


class SortByTH:
	pass


clan_tag = "2R00CGYR0"
response: dict = _api_request(f"clans/%23{clan_tag}/currentwar/leaguegroup")
rounds: list[list[str]] = [d["warTags"] for d in response["rounds"]]


async def _async_api_request(args: str) -> dict:
	while True:
		url = f"https://api.clashofclans.com/v1/{args}"
		try:
			async with aiohttp.ClientSession() as session:
				response = await session.get(url, headers={"Authorization": f"Bearer {get_key()}"})
				response = json.loads(await response.text())

			if getattr(response, "reason", None) == "accessDenied.invalidIp":
				raise requests.exceptions.ConnectionError
			break

		except (ClientOSError, ServerDisconnectedError, ConnectionResetError):
			time.sleep(1)

	return response


async def clan_in_war(war_tag: str):
	war_tag = war_tag.replace("#", "")
	print(f"Getting war tag: {war_tag}")
	response: dict[str, dict[str, str]] = await _async_api_request(f"clanwarleagues/wars/%23{war_tag}")
	clan1, clan2 = response["clan"], response["opponent"]

	def get_tag(x): return x["tag"].replace("#", "")
	clan_tag1, clan_tag2 = get_tag(clan1), get_tag(clan2)

	if clan_tag not in {clan_tag1, clan_tag2}:
		return None
	# logger.debug(f"{war_tag}: BLITZ ({clan_tag}) is in {clan1["name"]} ({clan_tag1}), {clan2["name"]} ({clan_tag2})")

	clan = clan1 if clan_tag1 == clan_tag else clan2
	opposing_clan = clan1 if clan_tag1 != clan_tag else clan2
	opposing_members = {
		member["tag"]:
		{"townhall_level": member["townhallLevel"], "map_position": n}
		for n, member in enumerate(sorted(opposing_clan["members"], key=lambda x: x["mapPosition"]), 1)

	}

	del clan1, clan2, response, clan_tag1, clan_tag2

	information = {}

	information["war_tag"] = war_tag
	information["members"] = []

	with open("response.json", "w") as file:
		json.dump(clan["members"], file)

	for n, member in enumerate(sorted(clan["members"], key=lambda x: x["mapPosition"]), 1):
		member: dict

		new_attack = {
			"stars": None,
			"destruction_percentage": None,
			"order": None,
			"opposing_member": {
				"townhall_level": None,
				"map_position": None
			}
		}

		if member.get("attacks", None) is not None:
			new_attack = {
				"stars": member["attacks"][0]["stars"],
				"destruction_percentage": member["attacks"][0]["destructionPercentage"],
				"order": member["attacks"][0]["order"],
				"opposing_member": opposing_members[member["attacks"][0]["defenderTag"]]
			}

		new_member = {
			"tag": member["tag"],
			"name": member["name"],
			"townhall_level": member["townhallLevel"],
			"map_position": n,
			"attack": new_attack
		}

		information["members"].append(new_member)

	return information


async def get_wars(rounds: list[list[str]]):
	tasks = {clan_in_war(war_tag) for round in rounds for war_tag in round}

	results = await asyncio.gather(*tasks - {asyncio.current_task()})
	while None in results:
		results.remove(None)

	return results

wars = asyncio.run(get_wars(rounds))

with open("response.json", "w") as file:
	json.dump(wars, file)

wars = [War(**war) for war in wars]

tag_to_members: dict[str, Member] = {}
member_list = []
for war in wars:
	for member in war.members:
		if tag_to_members.get(member["tag"], None) is None:
			tag_to_members[member["tag"]] = Member(
				member["tag"],
				member["name"],
				member["townhall_level"],
				member["map_position"],
				member["attack"]["stars"] or 0,
				int(member["attack"]["stars"] is not None),
				1,
				member["attack"]["destruction_percentage"] or 0
			)
			continue

		tag_to_members[member["tag"]].total_stars += member["attack"]["stars"] or 0
		tag_to_members[member["tag"]].total_attacks += member["attack"]["stars"] is not None
		tag_to_members[member["tag"]].total_chances += 1
		tag_to_members[member["tag"]].total_destruction += member["attack"]["destruction_percentage"] or 0

member_list = list(tag_to_members.values())


def sorter(m1: Member, m2: Member):
	return (m2.total_stars - m1.total_stars) or \
		(m2.total_attacks - m1.total_attacks) or \
		(m2.total_attacks/m2.total_chances - m1.total_attacks/m1.total_chances) or \
		(m2.total_destruction - m1.total_destruction)

def g(m1: Member, m2: Member):
	p1 = m1.total_attacks, m1.total_chances
	p2 = m2.total_attacks, m2.total_chances

	try:
		attack_diff = p1[0]/p2[0]
	except ZeroDivisionError:
		attack_diff = 0

	try:
		activity_rate2 = p2[0]/p2[1]
	except ZeroDivisionError:
		activity_rate2 = 0

	try:
		activity_rate1 = p1[0]/p1[1]
	except ZeroDivisionError:
		activity_rate1 = 0

	score = 1.015873 * p1[0] * (attack_diff - activity_rate2 + activity_rate1)
	return score

def g(couple: tuple[int, int]) -> float:
	identity = (0, 0)

	try:
		attack_diff = couple[0]/identity[0]
	except ZeroDivisionError:
		attack_diff = 0

	try:
		activity_rate2 = identity[0]/identity[1]
	except ZeroDivisionError:
		activity_rate2 = 0

	try:
		activity_rate1 = couple[0]/couple[1]
	except ZeroDivisionError:
		activity_rate1 = 0

	score = 1.015873 * couple[0] * (attack_diff - activity_rate2 + activity_rate1)
	return score

def groupby(__iterable: Iterable, /, key: Callable):
	lst_of_lst = []
	for _ in range(len(__iterable)):
		lst_of_lst.append([])

	grouped = {key(i): [] for i in __iterable}
	for i in __iterable:
		grouped[key(i)].append(i)

	return grouped


th_grouped: dict[str, Member] = groupby(member_list, key=lambda m: m.townhall_level)

for key in th_grouped:
	members: list

	members = th_grouped[key]

	# temp = sorted(members, key=lambda m: (g((m.total_attacks, m.total_chances)), -m.total_stars, -m.total_destruction))
	temp = sorted(members, key=cmp_to_key(sorter))
	print(temp)
	members.clear()
	members.extend(temp)


# th_grouped = {key: [attrs.asdict(v) for v in value]
# 			for key, value in th_grouped.items()}


class MyEncoder(JSONEncoder):
	def default(self, o):
		return attrs.asdict(o)


with open("results.json", "w") as file:
	json.dump(th_grouped, file, cls=MyEncoder)

groups_allocation = {ths: 0 for ths in th_grouped}
groups = cycle(sorted(th_grouped.keys()))

bonus_rewards = 7
names = []
for th, group in sorted(th_grouped.items()):
	for member in group:
		if bonus_rewards <= 0:
			break
		if member.activity_rate != 1:
			continue
		names.append(member.name)
		bonus_rewards -= 1




# for _ in range(bonus_rewards):
#     groups_allocation[next(groups)] += 1


# names = [th_grouped[key][:value] for key, value in groups_allocation.items()]
# names = [n for name in names for n in name]
# names = [i.name for i in names]


print(sep(names))
