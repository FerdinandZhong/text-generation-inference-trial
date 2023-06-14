PY_SOURCE_FILES=scripts ray_scripts #this can be modified to include more files

install: 
	pip install "pytest>=6"
	pip install "flake8>=3.8"
	pip install "black>=20.8b1"
	pip install "isort>=5.6"
	pip install "autoflake>=1.4"

test:
	pytest tests -vv -s

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name '*.pyc' -type f -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

package: clean
	python setup.py sdist bdist_wheel

format:
	autoflake --in-place --remove-all-unused-imports --recursive ${PY_SOURCE_FILES}
	isort ${PY_SOURCE_FILES}
	black ${PY_SOURCE_FILES}

lint:
	isort --check --diff ${PY_SOURCE_FILES}
	black --check --diff ${PY_SOURCE_FILES}
	flake8 ${PY_SOURCE_FILES} --count --show-source --statistics --max-line-length 120

