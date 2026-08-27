'''
Regression test for checkpoint key prefix handling in model_factory.

Guards against reintroducing str.strip('model.'), which removes CHARACTERS in
the set {m,o,d,e,l,.} rather than the prefix, silently truncating any key that
ends in one of them.

Runs without torch or a checkpoint download:

    python -m pytest tests/test_checkpoint_keys.py
    python tests/test_checkpoint_keys.py          # no pytest required
'''

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _strip_module_prefix(key, prefix='model.'):
	'''Mirror of the helper in model_factory (kept import-light for CI).'''
	return key.removeprefix(prefix)


# (checkpoint key, expected result) -- the first two are the parameters that
# actually failed to load from the released cmr_c0.1 checkpoint.
CASES = [
	('model.video_encoder.cls_positional_encoding.pos_embed_spatial',
	 'video_encoder.cls_positional_encoding.pos_embed_spatial'),
	('model.video_encoder.cls_positional_encoding.pos_embed_temporal',
	 'video_encoder.cls_positional_encoding.pos_embed_temporal'),
	('model.video_encoder.cls_token', 'video_encoder.cls_token'),
	('model.video_encoder.patch_embed.proj.weight', 'video_encoder.patch_embed.proj.weight'),
	('model.video_encoder.blocks.0.mlp.fc1.bias', 'video_encoder.blocks.0.mlp.fc1.bias'),
	('model.video_encoder.norm.weight', 'video_encoder.norm.weight'),
	('model.projection.0.weight', 'projection.0.weight'),
	# no prefix -> unchanged
	('video_encoder.norm.weight', 'video_encoder.norm.weight'),
	# only the leading prefix is removed
	('model.model.video_encoder.cls_token', 'model.video_encoder.cls_token'),
]


def test_prefix_removal():
	for key, expected in CASES:
		got = _strip_module_prefix(key)
		assert got == expected, f'{key!r} -> {got!r}, expected {expected!r}'


def test_differs_from_strip_on_trailing_chars():
	'''The two parameters that regressed must not survive a .strip() round-trip.'''
	for key in (
		'model.video_encoder.cls_positional_encoding.pos_embed_spatial',
		'model.video_encoder.cls_positional_encoding.pos_embed_temporal',
	):
		assert key.strip('model.') != _strip_module_prefix(key), (
			f'{key!r}: str.strip happens to agree -- test case no longer '
			f'exercises the bug'
		)


def test_source_has_no_strip_prefix_call():
	'''Fail if str.strip('model.') reappears in model_factory.py.'''
	path = os.path.join(os.path.dirname(__file__), '..', 'model_factory.py')
	with open(path) as fh:
		src = fh.read()
	hits = re.findall(r"\.strip\(\s*['\"]model\.['\"]\s*\)", src)
	assert not hits, f'found {len(hits)} occurrence(s) of .strip("model.")'


if __name__ == '__main__':
	failures = 0
	for name, fn in sorted(globals().items()):
		if name.startswith('test_') and callable(fn):
			try:
				fn()
				print(f'PASS {name}')
			except AssertionError as exc:
				failures += 1
				print(f'FAIL {name}: {exc}')
	sys.exit(1 if failures else 0)
