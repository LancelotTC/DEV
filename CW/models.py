from typing import NamedTuple

class Location(NamedTuple):
	id: int
	name: str
	is_country: bool



class Clan(NamedTuple):
	tag: str
	name: str
	type: str
	description: str
	location: Location