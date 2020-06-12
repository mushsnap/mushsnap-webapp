


IMAGE_NAME = mushsnap-webapp

build:
	docker build -t ${IMAGE_NAME} .

run: build
	docker run --rm -it -p 5000:5000 ${IMAGE_NAME}

build-run: build run
	