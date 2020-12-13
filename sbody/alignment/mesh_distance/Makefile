all:
	python setup.py build_ext --inplace

clean:tidy
	rm -f *.so

tidy:
	rm -rf build

test:
	python -m unittest discover
