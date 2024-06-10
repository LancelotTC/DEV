
import attrs, inspect, time

from typing import Optional, Callable
from collections.abc import Sequence
from threading import Thread, Event, current_thread

from pathlib import Path
from logging import Logger as _Logger, DEBUG, FileHandler, Formatter

import os


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


logger = Logger()

def reprs(obj: object="", /, default: Optional[str]=None) -> str:
	"""
	Gets all local-scoped variables with the same value, searches in globals if there aren't any in local,
	and returns all variable names that match the value. If default is provided, just return f"{default} = {var!r}".
	"""

	if default:
		return f"{default} = {obj!r}"

	if default is None:
		default = ""

	matches = []
	local_dict = inspect.currentframe().f_back.f_locals
	global_dict = inspect.currentframe().f_back.f_globals

	for key, value in local_dict.items():
		if value == obj:
			matches.append(key)

	if not matches:
		for key, value in global_dict.items():
			if value == obj:
				matches.append(key)

	result = sep(matches, " = ")

	return f"{result} = {obj!r}"

def sep(__iterable: Sequence[object], /, sep: Optional[str]=", ", *,
end: Optional[str]="") -> str:
	"""
	Adds 'sep' between each elements of of the iterable and returns the result
	"""

	return sep.join(__iterable) + end

@attrs.define
class ScheduledTask:
	function: Callable
	interval: int
	label: str

	def __hash__(self) -> int:
		return hash(self.function) + hash(self.interval)


class RecurringTasks:
	def __init__(self):
		self.running = True
		self.functions: list[ScheduledTask] = []
		self.threads: list[Thread] = []
		self.refresh_labels: dict[ScheduledTask, str] = {}
		self.len_last_label: int = 0

		import atexit
		atexit.register(self.stop)

	def add_task(self, func: Callable, interval: int, label):
		self.functions.append(ScheduledTask(func, interval, label))

	def start(self):

		event = Event()
		for scheduled_task in self.functions:
			thread = Thread(target=self._run_function, args=(scheduled_task, event), daemon=False)
			thread.start()

			self.threads.append(thread)

	def _run_function(self, scheduled_task: ScheduledTask, event: Event):
		remaining_interval = scheduled_task.interval
		# iters = 0
		self.refresh_labels[scheduled_task] = f"Refreshing {scheduled_task.label}"
		self._display(event)
		scheduled_task.function()
		while self.running:
			# iters += 1
			time.sleep(1)
			remaining_interval -= 1

			# I need to display in both cases because calling the function will delay the displaying otherwise,
			# and I can't know the label in advance unless I check twice which is ugly
			if remaining_interval <= 0:
				self.refresh_labels[scheduled_task] = f"Refreshing {scheduled_task.label}"
				self._display(event)
				scheduled_task.function()
				remaining_interval = scheduled_task.interval
			else:
				self.refresh_labels[scheduled_task] = f"Refreshing {scheduled_task.label} in {remaining_interval}s"
				self._display(event)

			#// TODO: Remove once done testing
			# if iters == 7:
			# 	break

	def stop(self):
		self.running = False

	def _display(self, lock: Event):
		# print("\r", end="")

		# id = choice([*range(1000)])
		name = current_thread().name

		# Wait for the lock to be released
		logger.debug(f"{name}: Waiting for lock to be released")
		logger.info(f"{lock.is_set() = }")

		while lock.is_set():
			time.sleep(.05)

		logger.debug(f"{name}: Acquiring the lock")
		# Acquire the lock
		lock.set()


		label = sep(self.refresh_labels.values(), " | ")

		print("\r" + label, end=" " * max(self.len_last_label - len(label), 0))
		logger.debug(f"{name}: \n\t{self.len_last_label = }\n\t {len(label) = }")
		self.len_last_label = len(label)
		logger.debug(f"Set {self.len_last_label = }")

		logger.debug(f"{name}: Releasing the lock")
		# Release the lock
		lock.clear()




from functools import partial
from typing import Optional, Union, Literal, _SpecialForm,\
	_GenericAlias, get_args, get_origin, Any, _SpecialGenericAlias
from collections.abc import Iterable, Collection, Sequence, Mapping
from types import GenericAlias, UnionType

from abc import ABCMeta
from inspect import _empty as empty, Signature
from functools import reduce, partial
from copy import deepcopy

from tools import flatten, reprs, count
from tools.deprecated import settify
import inspect

from tools.logger import Logger


logger = Logger()

# logger.setLevel("ERROR")
logger.info("Importing tools.overloading")

class _Left: """Internal class. Represents the left function for annotations comparison."""; pass
class _Right: """Internal class. Represents the right function for annotations comparison."""; pass

class SubRepr(list):
	# class _Left: """Internal class. Represents the left function for annotations comparison."""
	# class _Right: """Internal class. Represents the right function for annotations comparison."""

	def __init__(self, sub_repr: type | GenericAlias) -> None:
		# self.representation = self._get_repr(type_hint)
		# self.sub_repr = sub_repr
		super().__init__(sub_repr)

	@staticmethod
	def _order_by_specificity(sub_repr: list[type]) -> tuple[list[type], dict[type, int]]:
		priority_list: list[list[type]] = [
			[], # Literal type
			[], # object type
			[], # Regular types
			[], # Abstract base classes (ABCs)
			[], # Optional types (Union[type | None] is more specific than Union of 2 arbitrary types)
			[], # Union types
			[], # General types (Any and inspect._empty)
		]

		priority_dict: dict[type, int] = {}

		for item in sub_repr:
			if item is Literal:
				idx = 0
				priority_dict[item] = 0

			elif item is object:
				idx = 1
				priority_dict[item] = 1

			# inspect.isabstract doesn't support typing ABCs
			elif isinstance(item, ABCMeta) or isinstance(item, _SpecialGenericAlias):
				idx = 3
				priority_dict[item] = 3

			elif item is Union or item is UnionType:
				idx = 4
				item = Union
				priority_dict[item] = 4

			elif item is Any or item is empty:
				idx = 5
				priority_dict[item] = 5

			elif isinstance(item, type):
				idx = 2
				priority_dict[item] = 2

			elif isinstance(type(item), type):
				idx = None

			else:
				raise TypeError(f"Item of type: {type(item)} ({item}) is not valid")


			if idx is not None:
				priority_list[idx].append(item)

		return flatten(priority_list), priority_dict

	@staticmethod
	def get_more_specific(left_sub_repr: list[type], right_sub_repr: list[type]) -> _Left | _Right | None:

		# Weed out arguments with same type hints on both sides
		left_set = set(left_sub_repr)
		right_set = set(right_sub_repr)

		left_sub_repr = [*left_set.difference(right_set)]
		right_sub_repr = [*right_set.difference(left_set)]

		if not left_sub_repr and not right_sub_repr:
			return None

		if not left_sub_repr:
			return _Right

		if not right_sub_repr:
			return _Left

		left_list, left_dict = SubRepr._order_by_specificity(left_sub_repr)
		right_list, right_dict = SubRepr._order_by_specificity(right_sub_repr)


		for left_item, right_item in zip(left_list, right_list):
			if left_dict[left_item] == right_dict[right_item]:
				#? Idk how useful that is but anyway
				try: # __eq__ not defined error handling.
					if left_item == right_item:
						continue
				except AttributeError:
					if left_item is right_item:
						continue


				if issubclass(left_item, right_item):
					return _Left

				elif issubclass(right_item, left_item):
					return _Right

				else:
					# 2 types that have nothing to do with each other.
					# This happens when using Literals union other type
					return None
					# raise TypeError(f"Weird thing happened: {left_item}, {right_item}")

			elif left_dict[left_item] < right_dict[right_item]:
				return _Left
			elif left_dict[left_item] > right_dict[right_item]:
				return _Right
			else:
				logger.error(f"{left_item = } - {right_item = }")
				raise TypeError(f"What the fuck {left_item}, {right_item}")

		if len(left_list) > len(right_list):
			return _Left
		elif len(left_list) < len(right_list):
			return _Right
		else:
			return None

	# @staticmethod
	# def _compare_type_hints(left_type: type|GenericAlias, right_type: type|GenericAlias):
	# 	"""Compares 2 type-hints and returns which is more specific, that is, which one includes the other,
	# 	if at all. Returns the arbitrary representation of the more specific one, or None if not comparable."""
	# 	left_type_repr = get_repr(left_type)
	# 	right_type_repr = get_repr(right_type)

	# 	result = {
	# 		left_type_repr: left_type,
	# 		right_type_repr: right_type
	# 	}

	# 	return result.get(compare_repr(left_type_repr, right_type_repr), None)

	# def compare(self, other):
	# 	"""Compares 2 SubRepr types and returns the one that is most specific."""
	# 	if not isinstance(other, SubRepr):
	# 		raise TypeError(f"Expected {type(self)} type, got {type(other)}")

	# 	results = {
	# 		_Right: self,
	# 		_Left: other
	# 	}

	# 	return results.get(self._check_specificity(self.sub_repr, other.sub_repr), None)

	# @staticmethod
	# def get_more_specific(left, right):
	# 	# error = None

	# 	# if not isinstance(left, SubRepr):
	# 	# 	error = type(left)

	# 	# if not isinstance(right, SubRepr):
	# 	# 	error = type(right)


	# 	# if error is not None:
	# 	# 	raise TypeError(f"Expected {SubRepr.__name__} type, got {type(error)}")

	# 	left, right = SubRepr(left), SubRepr(right)

	# 	# results = {
	# 	# 	_Right: right,
	# 	# 	_Left: left
	# 	# }


	# 	# return results.get(SubRepr._check_specificity(left, right), None)
	# 	return SubRepr._check_specificity(left, right)

class Repr(list):
	"""Generates an arbitrary, comparable representation of a type-hint"""
	def __init__(self, type_hint) -> None:
		self.type_hint = type_hint
		self.repr = self._get_repr(type_hint)
		super().__init__(self.repr)

	def _get_repr(self, typ: type | GenericAlias) -> list[list]:
		"""Get a list representation of a type-hint. The further the list goes the deeper the nested type-hint.
		Very useful to compare type-hints. Example:
		list[str] -> [[list], [str]]
		list[list[str]] -> [[list], [list], [str]]
		dict[Union[str, int], list[Literal["literal1", "literal2"]]] ->
		[[<class 'dict'>], [typing.Union, <class 'list'>], [<class 'str'>, <class 'int'>, typing.Literal],
		['literal1', 'literal2']]
		"""

		result = []

		def is_generic_alias(typ):
			return isinstance(typ, GenericAlias) or issubclass(type(typ), (_SpecialForm, UnionType, _GenericAlias))
			# return typ is Union or typ is UnionType or typ is Literal

		def get_origins(args):
			lst = []
			for arg in args:
				# Before I was checking whether arg is a type, but since I do the same thing if it's not
				# a generic alias anyway just remove these 2 lines of code right.
				if is_generic_alias(arg):
					lst.append(get_origin(arg))
				else:
					lst.append(arg)
			return lst


		def wrapper(args):
			result.append(get_origins(args))
			args = flatten([get_args(arg) for arg in args if is_generic_alias(arg)])
			while () in args:
				args.remove(())
			if args:
				wrapper(args)

		wrapper([typ, ])

		return result

	@staticmethod
	def get_more_specific(left, right, *, return_annotation: bool=False)\
	-> (_Left | _Right | None) | (type | GenericAlias):
		"""Accepts 2 objects of type Repr and returns either _Left | _Right | None depending on which is more specific.
		Order of argument always matters.
		The 'return_annotation' argument allows the return type to be the corresponding type-hint instead of a 'side' object.
		"""

		errmsg = None

		if not isinstance(left, Repr):
			errmsg = type(left)

		if not isinstance(right, Repr):
			errmsg = type(right)

		if errmsg is not None:
			raise TypeError(f"Expected {Repr} type, got {errmsg}")

		for sub_repr1, sub_repr2 in zip(left.repr, right.repr):

			# for t1, t2 in zip(sub_repr1, sub_repr2):
			result = SubRepr.get_more_specific(sub_repr1, sub_repr2)
			if return_annotation:
				result = {_Left: left.type_hint, _Right: right.type_hint}.get(result, None)

			# Careful here return statement is in loop so it makes sense to check for None
			if result is not None:
				return result

		return None

		# return {_Left: left, _Right: right}.get(Repr.compare(left, right), None)

from types import FunctionType


# Metaclass for Singleton. Dot notation works
class SingletonMeta(type):
	"""Class decorator that allows only 1 instance. In case of multiple instanciation, returns
	the same instance."""

	_instances = {}

	def __call__(cls, *args, **kwargs):
		# __call__ as a class method has priority over the __init__ method of the class because
		# it was defined in the metaclass.
		if cls not in cls._instances:
			cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
		return cls._instances[cls]

class Overload(metaclass=SingletonMeta):
	"""Implementation of overload decorator for normal functions. Use the 'overload' function
	as a decorator to overload your functions. Alternatively you can use this class for its
	useful static methods."""

	def __init__(self) -> None:
		self._registry: dict[str, list] = {}

	def get_type_hint(self, obj: Any, *, precision: int=2) -> type|GenericAlias:
		"""Returns a type-hint in the form of a simple type or a GenericAlias with a given
		precision."""
		def union(iterable: Iterable) -> type | GenericAlias:
			string = ""

			# This entire block of code is shit tbh
			for t in iterable:
				if isinstance(t, GenericAlias):
					# Weak implementation but idgaf
					string += f"{t}|".replace("'", "").replace("<class ", "").replace(">", "")
					continue

				try:
					string += fr"{t.__name__}|"
				except AttributeError:
					string += fr"{type(t).__name__}|"

			return eval(f"({string[:-1]})")

		# Check whether obj is a str because str are Collections.
		if (isinstance(obj, Collection) and len(obj) == 0) or not isinstance(obj, Collection) or isinstance(obj, str):
			return type(obj)

		if isinstance(obj, Mapping):
			# keys = settify([self.get_type_hint(t) for t in obj.keys()])
			keys = [self.get_type_hint(t) for t in obj.keys()]
			values = settify([self.get_type_hint(t) for t in obj.values()])

			if len(keys) == 1:
				keys = keys[0]
			elif len(keys) <= precision:
				keys = union(keys)
			else:
				keys = object

			if len(values) == 1:
				values = values[0]
			elif len(values) <= precision:
				values = union(values)
			else:
				values = object

			return GenericAlias(type(obj), slice(keys.__name__, values.__name__))

		if isinstance(obj, Sequence):
			items = settify([self.get_type_hint(t) for t in obj])
			origins: list = settify([get_origin(x) for x in items])
			while None in origins:

				origins.remove(None)

			if len(items) == 1:
				items = items[0]
			elif len(items) <= precision:
				items = union(items)


			elif len(origins) == 1:
				items = origins[0]
			elif len(origins) <= precision:
				items = union(origins)

			else:
				items = object

			if items == ():
				items = object
			return GenericAlias(type(obj), items)

		return NotImplemented

	def check_type_hint(self, obj: Any, type_hint: type|GenericAlias) -> int | bool:
		"""Checks whether 'obj' fits 'type-hint'"""

		origin = get_origin(type_hint)
		args = get_args(type_hint)

		if origin is None:
			if type_hint is Any:
				return True

			return isinstance(obj, type_hint)

		if origin is Literal:
			return obj in args

		if origin is UnionType or origin is Union:
			return count(partial(self.check_type_hint, obj), [arg for arg in args])

		if origin is Optional:
			return obj is None or self.check_type_hint(obj, args[0])

		if not isinstance(obj, origin):
			return False

		if issubclass(origin, Sequence):
			return all(self.check_type_hint(item, args[0]) for item in obj)

		if issubclass(origin, Mapping):
			# if not isinstance(args, slice):
			logger.debug(f"{args = }, {type(args)}, {obj}")

			# args: slice = args[0]
			args = slice(*args, 0)


			try:
				key_type, value_type = args.start, args.stop
			except AttributeError as e:
				logger.error(e)
			return (
				isinstance(obj, dict) and
				all(self.check_type_hint(key, key_type) and self.check_type_hint(value, value_type) for key, value in obj.items())
			)

		return isinstance(obj, origin)

	# ! Deprecated, do not use
	@staticmethod
	def _validate_arguments(*args, config=None, **kwargs) -> bool:
		"""Deprecated, do not use. Replaced by inspect.bind"""

		arguments, defaults = config
		args_num, args_kwargs, kw_args, var_args_exist, var_kwargs_exist = arguments

		args_num: int
		args_kwargs: list[str]
		kw_args: list[str]
		var_args_exist: bool
		var_kwargs_exist: bool

		args = list(args)

		for _ in range(args_num):
			try:
				args.pop(0)
			except IndexError:
				return False

		for kw in kw_args:
			try:
				del kwargs[kw]
			except KeyError:
				if defaults[kw] is empty:
					return False
			# del defaults[kw]

		args_kwargs = args_kwargs[::-1]
		for arg_kwarg in args_kwargs.copy():
			try:
				del kwargs[arg_kwarg]
				args_kwargs.remove(arg_kwarg)
			except KeyError:
				pass
		args_kwargs = args_kwargs[::-1]
		for arg_kwarg in args_kwargs.copy():
			try:
				args.pop(0)
				args_kwargs.pop(0)
			except IndexError:
				if defaults[arg_kwarg] is empty:
					return False

		if args_kwargs and any(defaults[key] is empty for key in args_kwargs):
			return False

		if (args and not var_args_exist) or (kwargs and not var_kwargs_exist):
			return False

		return True

	@staticmethod
	def validate_arguments(*args, sig: Signature, **kwargs) -> bool:
		"""Takes in arguments and a configuration of arguments and defaults and
		returns True if the passed-in arguments are conform to the configuration.
		Doesn't take type-hints into account"""

		logger.debug(f"Validating arguments, {sig = }, {args = }, {kwargs = }. Assume qualifies unless stated otherwise")
		try:
			return bool(sig.bind(*args, **kwargs))
		except TypeError:
			logger.debug("Validate_arguments returned False, function doesn't qualify")
			return False

	def validate_typehints(self, *args, config: tuple, **kwargs) -> int | Literal[False]:
		"""Takes in arguments and a configuration of type-hints and defaults and
		returns True if the passed-in arguments' type-hints are conform to the configuration.
		Assumes to a certain extent that the number of arguments are conform to the configuration."""
		logger.debug(f"Validating type-hints, {config = }, {args = }, {kwargs = }")

		type_hints, _ = config
		config_arg_types, config_arg_kwarg_types, config_kwarg_types, var_args, var_kwargs = type_hints

		config_arg_types: list[type|GenericAlias]
		config_arg_kwarg_types: dict[str: type|GenericAlias]
		config_kwarg_types: dict[str: type|GenericAlias]
		var_args: type|GenericAlias
		var_kwargs: type|GenericAlias

		args = list(args)

		score = 0
		result = 0

		for type_hint in config_arg_types.copy():
			try:
				result = self.check_type_hint(args.pop(0), type_hint)
				if not result and type_hint is not empty:
					logger.debug(f"not {result = } and {type_hint = } is not empty -> return False")
					return False
			except IndexError:
				logger.debug("validate_typehints -> config_arg_types -> IndexError -> return False")
				return False

			score += result
			config_arg_types.remove(type_hint)

		for keyword in kwargs.copy():
			try:
				result = self.check_type_hint(kwargs[keyword], config_kwarg_types[keyword])
				if not result and \
				config_kwarg_types[keyword] is not empty:
					logger.debug(f"not {result = } and \
					{config_kwarg_types[keyword] = } is not empty -> return False")

					return False
					# try:
					# 	if defaults[keyword] is empty:
					# 		return False
					# except KeyError:
					# 	return False
				del kwargs[keyword], config_kwarg_types[keyword]
			except KeyError: # Error handling due to variable kwargs
				pass
			score += result

		# Check off kwargs from config args/kwargs
		for arg_kwarg in reversed(kwargs.copy()):
			if arg_kwarg in config_arg_kwarg_types.copy():
				result = self.check_type_hint(kwargs[arg_kwarg], config_arg_kwarg_types[arg_kwarg])
				if not result and \
				config_arg_kwarg_types[arg_kwarg] is not empty:
					logger.debug(f"not {result = } and \
					{config_arg_kwarg_types[keyword] = } is not empty -> return False")
					return False
				# try:
				# 	if defaults[arg_kwarg] is empty:
				# 		return False
				# except KeyError:
				# 	return False
				del kwargs[arg_kwarg], config_arg_kwarg_types[arg_kwarg]
				score += result

		if empty not in var_args and Any not in var_args:
			for var_arg in var_args:
				result = self.check_type_hint(tuple(args), var_arg)

				if not result:
					logger.debug(f"var_args -> not {result = } -> False")
					return False

				score += result

		if empty not in var_kwargs and not any(Any in get_args(var_kw) for var_kw in var_kwargs):
			for var_kwarg in var_kwargs:
				result = self.check_type_hint(kwargs, var_kwarg)

				if not result:
					logger.debug(f"var_kwargs -> not {result = } -> False")
					return False

				score += result

		for key, value in config_arg_kwarg_types.copy().items():
			try:
				current_type = args.pop(0)
			except IndexError:
				break

			result = self.check_type_hint(current_type, value)

			if not result and value is not empty:
				logger.debug(f"config_arg_kwarg_types -> not {result = } and {value = } is not empty -> False")
				return False

			del config_arg_kwarg_types[key]
			score += result

		if (len(args) > len(config_arg_types) + len(config_arg_kwarg_types) and not var_args) or \
			(len(kwargs) > len(config_kwarg_types) + len(config_arg_kwarg_types) and not var_kwargs):
			logger.debug("return False")
			return False

		return score + 1

	@staticmethod
	def arg_tie_breaker(config=None) -> int | float:
		"""Returns a score for a given configuration that is used as a tie-breaker between
		two or more functions that would get past the validate_arguments and the
		validate_typehints functions."""

		arguments, defaults = config
		args, args_kwargs, kwargs, var_args, var_kwargs = arguments

		defaults: dict
		args_kwargs: list
		kwargs: list
		len_def = len(defaults)

		for default in defaults.copy():
			if defaults[default] is empty:
				del defaults[default]
				continue
			try:
				kwargs.remove(default)
			except ValueError:
				args_kwargs.remove(default)
			del defaults[default]



		defaults = len(defaults)
		args: int
		args_kwargs: dict[str: type] = len(args_kwargs)
		kwargs: dict[str: type] = len(kwargs)


		if not any([args, args_kwargs, kwargs, var_args, var_kwargs]):
			return 0.5

		# print(reprs(args))
		# print(reprs(kwargs))
		# print(reprs(args_kwargs))
		# print(reprs(var_args))
		# print(reprs(var_kwargs))
		# print(reprs(defaults))
		# print(f"score: {2 * (args + kwargs) + args_kwargs - (var_args + var_kwargs + 0.5 * len_def)}")
		# print()


		return 2 * (args + kwargs) + args_kwargs - (var_args + var_kwargs + 0.5 * len_def)
		# return 2 * (args + kwargs) + args_kwargs - (var_args + var_kwargs)

	def __call__(self, func: FunctionType, *, static_typing: bool=True, dummy: bool=False) -> FunctionType:
		"""
		Overload decorator that is type_hint sensitive

		Problems:
		- The decorator pushes a static typing-like approach to a dynamically typed language.
		- There aren't any clear variable argument type-checking
		- No testing has been done on methods. Use typing.overload for them
		"""
		name = func.__qualname__

		if not dummy:
			arguments: list[int | list | bool] = [0, [], [], False, False]
			type_hints: list[list | dict] = [[], {}, {}, [], []]
			defaults: dict[str, empty | type] = {}

			sig = inspect.signature(func)

			# Gathers the param name, its type-hints and whether it is a default
			for param, values in sig.parameters.items():
				type_hint = values.annotation
				default = values.default
				kind = str(values.kind)

				if kind == "POSITIONAL_ONLY":
					arguments[0] += 1
					type_hints[0].append(type_hint)

				elif kind == "POSITIONAL_OR_KEYWORD":
					arguments[1].append(param)
					type_hints[1][param] = type_hint
					defaults[param] = default

				elif kind in "KEYWORD_ONLY":
					arguments[2].append(param)
					type_hints[2][param] = type_hint
					defaults[param] = default

				elif kind == "VAR_POSITIONAL":
					arguments[3] = True
					type_hints[3].append(type_hint)

				elif kind == "VAR_KEYWORD":
					arguments[4] = True
					type_hints[4].append(type_hint)

			signature = (tuple(arguments), tuple(type_hints), defaults)

			# Create the first overloaded function for a name, otherwise add to it
			if name not in self._registry:
				self._registry[name] = [(func, signature, sig), ]
			else:
				self._registry[name].append((func, signature, sig))

		def inner(*args, **kwargs):
			"""inner"""
			items = self._registry[name]
			argument_scores = {}
			literals: dict[str, dict[str, tuple]] = {}
			literal_tie_breaker = {}

			for func, signature, sig in items:
				arguments, type_hints, defaults = deepcopy(signature)
				# sig = inspect.signature(func)

				# Using "deepcopy" because of an issue that consisted of typehints changing midway through runtime
				# and triggering unjustified errors. Apparently tuple is mutable and I change it in the validate_typehints function
				# (lmao)
				# Now I'm using deepcopy everywhere because of the issue happening to every single one.
				#? Just so you know you piece of shit, you had mutable types in your bitch ass tuple which you unpacked in
				#? this function, that's why you got a bug and why it worked when using deepcopy


				if self.validate_arguments(*args, sig=sig, **kwargs):
					result = self.validate_typehints(*args, config=(type_hints, defaults), **kwargs)

					logger.debug(f"validate_typehints returned {result} for {sig}")

					if static_typing and not result:
						continue

					literal_tie_breaker[func] = result
					argument_scores[func] = self.arg_tie_breaker([arguments, defaults])

			# bring this line back down if problems
				del func

			logger.debug(f"{argument_scores = }")


			if not argument_scores:
				string = ""

				for arg in args:
					string += repr(arg) + ", "

				for key, value in kwargs.items():
					string += f"{key}={repr(value)}, "

				string = string.removesuffix(", ")
				error_message = f"No saved signatures matches function call {name}({string})."
					# " \n    Make sure keyword names and type-hints are respected as they are a common source of error"
				# error_message = f"{name}({string}) "\
				# 	"doesn't have any signatures that fits the passed in arguments (or lack thereof) and/or their typehints"
				raise TypeError(error_message)

			# Getting all functions with highest scores (often only 1)

			max_value = max(argument_scores.values())
			funcs = [key for key, value in reversed(argument_scores.items()) if value == max_value]

			literal_tie_breaker = {func: score for func, score in literal_tie_breaker.items() if func in funcs}

			max_value = max(literal_tie_breaker.values())
			funcs = [key for key, value in reversed(literal_tie_breaker.items()) if value == max_value]

			# I don't need to make typ-hints comparisons to get the most specific one
			# if there's only 1 function left already.
			logger.debug(f"{len(funcs)} function(s) qualify")
			if len(funcs) == 1:
				logger.debug("Returning only function")
				return funcs[0](*args, **kwargs)

			# Maps functions with the arbitrary representation of all their type-hints if func in funcs
			# This is to be able to compare type-hint specificity
			# Consider the following:
			# @overload
			# def func1(arg: list):
			# 	pass
			# @overload
			# def func1(arg: Sequence):
			# 	pass
			#
			# func1([1, 2, 3]) # I want to call the first function
			# func1((1, 2, 3)) # I want to call the second function
			#
			# On the first function call, I pass in a list, which matches both list and Sequence.
			# But if I annotated my function like that maybe I want a special case for list right?
			# The following code will check which of the 2 is more specific, as in, which includes the other.
			# If I don't pass in a list, then the corresponding function will be removed out of the equation, so
			# the next best (and sole) contender is the second function


			items = {func: [Repr(type_hint) for type_hint in flatten(signature[1:2])]
					for func, signature, sig in items if func in funcs}


			#? In construction
			literals = {func: arg for func, arg in literals.items() if func in funcs}
			#?

			def reduce_funcs(left_func: FunctionType, right_func: FunctionType) -> FunctionType:
				"""Function that decides which of the left or right functions to keep"""
				left_reprs = items[left_func]
				right_reprs = items[right_func]

				for left_repr, right_repr in zip(left_reprs, right_reprs):
					result = Repr.get_more_specific(left_repr, right_repr)

					if result is _Left:
						return left_func
					elif result is _Right:
						return right_func
					else:
						continue
				# Return the youngest one by default
				return right_func
				# return {left_func: right_func}

			func = reduce(reduce_funcs, reversed(funcs))

			return func(*args, **kwargs)

		return inner

def overload(func: FunctionType, *, static_typing: bool=True, dummy: bool=False) -> FunctionType:
	"""Overload decorator that is type-hint sensitive

		Problems:
		- The decorator pushes a static typing-like approach to a dynamically typed language.
		- There aren't any clear variable argument type-checking
		- No testing has been done on methods. Use typing.overload for them"""

	return Overload()(func, static_typing=static_typing, dummy=dummy)


def is_method(func: FunctionType) -> bool:
	"""Weak implementation to check whether the callable is a method. It checks the __qualname__ of the function and
	returns True if it contains a '.'. This makes sure that staticly passed in methods such as Class.method are still
	counted as one, unlike using __self__ or inspect.ismethod.
	"""

	if "." in func.__qualname__:
		return True
	return False

class MethodCallError(Exception): pass

def override(func: FunctionType) -> FunctionType:
	"""Mark methods with this decorator to mark them as overridable and prevent them from being accidentally called."""
	func.__override__ = True


	def override_method(*args, **kwargs):
		raise MethodCallError("Method marked with @override cannot be called.")

	override_method.__override__ = True

	return override_method

def is_override(func: FunctionType):
	return getattr(func, "__override__", False)

# def overload_repr() -> FunctionType:
# 	def wrapper(func: FunctionType, *, static_typing: bool=True) -> FunctionType:
# 			return overload(func, static_typing=static_typing, repr=True)
# 	return wrapper

