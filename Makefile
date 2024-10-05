run:
	python -B -m src

train:
	python -B -m src train


test:
	python -B -m src test

clean:
	rm -rf build/*
