MNIST Airflow Demo
==================

Installation
------------

- Install Airflow using [these instructions][installation]
  - The `docker-compose.yaml` file is included here for convenience
- Run this command due to https://github.com/apache/airflow/issues/12985

      docker exec mnist-airflow_airflow-worker_1 pip install tensorflow==1.14.0 h5py==2.10.0`

- Go to http://localhost:8080/home and log in with default credentials of
  `airflow`/`airflow`

[installation]: https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html

Usage
-----

To run this example, run the `mnist` DAG that was automatically created in the
dashboard. You can select the "Trigger DAG" option to run with defaults, or you
can select the "Trigger DAG w/ config" option to override default options.

If you want to override the default options, add a config that looks like this:

```json
{
    "conf": {
        "train_images": "<URL>"
        "train_labels": "<URL>"
        "test_images": "<URL>"
        "test_labels": "<URL>"
    }
}
```

Where any of `train_images`, `train_labels`, `test_images`, or `test_labels` can
be included or excluded as necessary. They should point to URLs that are
accessible to the Airflow executor task, and are copies of the [MNIST dataset
found here][mnist].

[mnist]: http://yann.lecun.com/exdb/mnist/

