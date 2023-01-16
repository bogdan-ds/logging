Installation
------------

Install poetry:

```bash
pip install poetry
```

Navigate to the project directory

```bash
poetry install
```

Testing
-------

Tests are run using pytest via poetry:

```bash
poetry run pytest
```

TODO
----

- Migrate IWPOD_NET from tensorflow to pytorch
- Clean up IWPOD_NET code
- Create DB for detections
- Write time into video when detection appears and ends
- Investigate TensorRT for more optimal inference
- Google Drive client
- Check for new items based on DB