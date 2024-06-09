import requests
from aiohttp.client_exceptions import ClientOSError, ServerDisconnectedError

import aiohttp
import json
import sqlite3
from contextlib import contextmanager
from time import sleep
from typing import Any, Sequence, Mapping, Callable
from collections.abc import Iterable

from logging import Logger as _Logger, DEBUG, FileHandler, Formatter
from typing import Optional
from pathlib import Path
import inspect
import os
import asyncio


def get_key():
	with open("./api_key.txt", "r") as file:
		return file.readline().strip()

def _api_request(tag: str, obj: str, suffix: str="") -> dict:
	tag = tag.replace("#", "")
	while True:
		try:
			response = requests.get(
				f"https://api.clashofclans.com/v1/{obj}/%23{tag}/{suffix}",
				# f"https://af8cc933-3f4c-4bbc-9e27-cd1d204d736d.mock.pstmn.io/v1/{obj}/%23{tag}/{suffix}",
				headers={"Authorization": f"Bearer {get_key()}"}
			)
			response = json.loads(response.content)
			if getattr(response, "reason", None) == "accessDenied.invalidIp":
				raise requests.exceptions.ConnectionError
			break
		except requests.exceptions.ConnectionError:
			sleep(1)

	with open("response.json", "w") as file:
		json.dump(response, file)

	return response

def get_player(player_tag: str) -> dict:
	return _api_request(player_tag, "players")

def get_clan_members(clan_tag: str) -> dict:
	return _api_request(clan_tag, "clans", "members")


async def _api_request_async(tag: str, obj: str, suffix: str="") -> dict:
	tag = tag.replace("#", "")

	while True:
		async with aiohttp.ClientSession() as session:
			response = await session.get(
				f"https://api.clashofclans.com/v1/{obj}/%23{tag}/{suffix}",
				headers={"Authorization": f"Bearer {get_key()}"}
			)
			await asyncio.sleep(1)
		try:
			response = json.loads(await response.read())
			if getattr(response, "reason", None) == "accessDenied.invalidIp":
				raise requests.exceptions.ConnectionError
			break
		except requests.exceptions.ConnectionError:
			sleep(1)

	with open("response.json", "w") as file:
		json.dump(response, file)

	return response

async def get_player_async(player_tag: str) -> dict:
	return await _api_request_async(player_tag, "players")

async def get_clan_members_async(clan_tag: str) -> dict:
	return await _api_request_async(clan_tag, "clans", "members")


@contextmanager
def open_db(file: str="contributions.db"):
	try:
		connection = sqlite3.connect(file)
		db = connection.cursor()
		yield db, connection
	finally:
		db.close()
		connection.close()




def traverse(iterable: Iterable[Any], /):
	"""Returns a Generator that iterates through every non-Sequence object
	of a list or tuple, regardless of depth.

	Works only for 1 iterable for the sake of recursiveness.
	Use flatten to use for any number of iterables"""

	if isinstance(iterable, Sequence) and not isinstance(iterable, str):
		for value in iterable:
			for subvalue in traverse(value):
				yield subvalue
	elif isinstance(iterable, Mapping):
		for value in iterable.values():
			for subvalue in traverse(value):
				yield value
	else:
		yield iterable

# Can cause a MemoryError if the Sequence object is too big. Do not confuse with traverse
def flatten(*iterables: Iterable[Any], factory: Callable=list) -> list[Any]:
	"""
	factory(*traverse(iterables))

	Uses the 'traverse' function to return all the values of the iterables passed in as a list.
	Supports any number of iterables. Do not confuse with traverse (which you rather be using).
	traverse returns a generator, flatten iterates through it and puts the contents in the factory type.
	Beware the MemoryError with large lists.
	"""
	return factory(traverse(iterables))

class Logger(_Logger):
	"""Logger dispatcher class. Equivalent to logger.create_logger"""

	def __init__(self, filename: Optional[str]=None, /, *, name: Optional[str]=None, level: int=DEBUG,
	filemode: str="w", encoding: str="utf-8", folder_name: str | Path = "logs",
	format: str="%(asctime)s, in %(filename)s, %(funcName)s - %(levelname)s: %(message)s (line %(lineno)d)",
	time_format: str="%Y-%m-%d %H:%M:%S") -> _Logger:
		if None in (name, filename):
			frame = inspect.currentframe().f_back

			while frame.f_code.co_filename.startswith('<frozen'):
				frame = frame.f_back

			auto_name = frame.f_code.co_filename

			if name is None:
				name = auto_name

			if filename is None:
				filename = os.path.splitext(os.path.split(auto_name)[1])[0] + ".log"

			filename = Path(os.path.split(name)[0], folder_name, filename)


		super().__init__(name, level)

		os.makedirs(os.path.dirname(filename), exist_ok=True)


		handler = FileHandler(filename, filemode, encoding)
		handler.setFormatter(Formatter(format, time_format))

		self.addHandler(handler)

def create_logger(filename: Optional[str]=None, /, *, name: Optional[str]=None, level: int=DEBUG,
filemode: str="w", encoding: str="utf-8", folder_name: str | Path = "logs",
format: str="%(asctime)s, in %(filename)s, %(funcName)s - %(levelname)s: %(message)s (line %(lineno)d)",
time_format: str="%Y-%m-%d %H:%M:%S") -> _Logger:
	"""Logger dispatcher function. Equivalent to logger.Logger"""

	if None in (name, filename):
		frame = inspect.currentframe().f_back

		while frame.f_code.co_filename.startswith('<frozen'):
			frame = frame.f_back

		auto_name = frame.f_code.co_filename

		if name is None:
			name = auto_name

		if filename is None:
			filename = os.path.splitext(os.path.split(auto_name)[1])[0] + ".log"

	logger = _Logger(name, level)
	filename = Path(folder_name, filename)

	handler = FileHandler(filename, filemode, encoding)
	handler.setFormatter(Formatter(format, time_format))

	logger.addHandler(handler)

	return logger

class ProgressBar:
	def __init__(self, total: int, iteration: Optional[int]=0, prefix: Optional[str]='', suffix: Optional[str]='',
	decimals: Optional[int]=1, length: Optional[int]=50, fill: Optional[str]= '█', print_end: Optional[str]="\r") -> None:
		self.iteration = iteration
		self.total = total
		self.prefix = prefix
		self.suffix = suffix
		self.decimals = decimals
		self.length = length
		self.fill = fill
		self.print_end = print_end
		self._finished = False
		self.progress_bar_length = 0

	def init(self):
		if self.iteration > self.total:
			if not self._finished:
				self.finish()
			self._finished = True
			return

		try:
			self.percent = f"{self.iteration/self.total*100: .{self.decimals}f}"
		except ZeroDivisionError:
			return

		filled_length = int(self.length * self.iteration // self.total)
		bar = self.fill * filled_length + '-' * (self.length - filled_length)
		progress_bar = f"{self.prefix} {self.iteration}/{self.total} |{bar}| {self.percent}% {self.suffix}"
		self.progress_bar_length = len(progress_bar)
		print(f'\r{progress_bar}', end=self.print_end)

	def increment(self):
		self.iteration += 1
		self.init()

	def clear_line(self):
		print("\r" + " "*self.progress_bar_length, end="\r")

	def finish(self):
		print()
