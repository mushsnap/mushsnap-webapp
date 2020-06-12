
include .envrc_vars

IMAGE_NAME = mushsnap-webapp

build:
	docker build -t ${IMAGE_NAME} . 

run: build
	docker run --rm -e FILE_URL=${FILE_URL} -it -p 5000:5000 ${IMAGE_NAME}

build-run: build run
	



docker-build-push:
	# Create the image and push it to the container reqistry.
	heroku container:push web

docker-release:
	heroku container:release web

open-app:
	heroku open