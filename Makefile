PY_SOURCE_FILES=data_process/ training/ inference/ utils/ examples/ #this can be modified to include more files

install: package
	pip install -e .[dev]

test:
	pytest tests -vv

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name '*.pyc' -type f -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

package: clean
	python3 setup.py sdist bdist_wheel

format:
	autoflake --in-place --remove-all-unused-imports --recursive ${PY_SOURCE_FILES}
	isort ${PY_SOURCE_FILES}
	black ${PY_SOURCE_FILES}

lint:
	isort --check --diff ${PY_SOURCE_FILES}
	black --check --diff ${PY_SOURCE_FILES}
	flake8 ${PY_SOURCE_FILES} --count --show-source --statistics --max-line-length 120
	revive -config revive.toml -formatter friendly .

