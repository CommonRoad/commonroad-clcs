## CI Docker image

To accelerate the CI pipeline we use a Docker image where the build dependencies are pre-installed. 
If we need any new dependencies, we need to update the Docker image that is used for CI.

1. Update the `Dockerfile` with the new dependencies
2. Login to the Gitlab Docker registry
    ```bash
    docker login gitlab.lrz.de:5005
    ```
3. Build the Docker image locally
    ```bash
    docker build -t gitlab.lrz.de:5005/cps/commonroad/commonroad-clcs/deps:<TAG> .
    ```
    > **Important:** Use the same <TAG> as the one in the `.gitlab-ci.yml` file.
4. Push the image to the GitLab container registry
    ```bash
    docker push gitlab.lrz.de:5005/cps/commonroad/commonroad-reachable-set/deps:<TAG>
    ```
