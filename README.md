# kfserving-keras-transformer
A transforming for KFServing that users keras_image_helper


## Running locally

Running the service

```bash
export MODEL_INPUT_SIZE="299,299"
export KERAS_MODEL_NAME="xception"
export MODEL_LABELS="dress,hat,longsleeve,outwear,pants,shirt,shoes,shorts,skirt,t-shirt"

python image_transformer.py \
    --predictor_host="clothing-model.default.kubeflow.mlbookcamp.com" \
    --model_name="clothing-model"
```

Testing it

```python
import requests

data = {
    "instances": [
        {"url": "http://bit.ly/mlbookcamp-pants"},
    ]
}

url = 'http://localhost:8080/v1/models/clothing-model:predict'
result = requests.post(url, json=data).json()

print(result)
```

Or run `python test.py`

## Running with Docker

Build it:

```bash
LOCAL_TAG="kfserving-keras-transformer:0.0.1"
docker build -t ${LOCAL_TAG} .
```

Running it:

```bash
docker run -it \
    -p 8080:8080 \
    -e MODEL_INPUT_SIZE="299,299" \
    -e KERAS_MODEL_NAME="xception" \
    -e MODEL_LABELS="dress,hat,longsleeve,outwear,pants,shirt,shoes,shorts,skirt,t-shirt" \
    ${LOCAL_TAG} \
    --predictor_host="clothing-model.default.kubeflow.mlbookcamp.com" \
    --model_name="clothing-model"
```

Testing:

```bash
python test.py
```

Publishing:

```bash
REMOTE_TAG="agrigorev/${LOCAL_TAG}"
docker tag ${LOCAL_TAG} ${REMOTE_TAG}
docker push ${REMOTE_TAG}
```

## Using it with KFServing

```yaml
apiVersion: "serving.kubeflow.org/v1alpha2"
kind: "InferenceService"
metadata:
  name: "clothing-model"
spec:
  default:
    predictor:
      serviceAccountName: sa
      tensorflow:
        storageUri: "s3://mlbookcamp-models/clothing-model"
    transformer:
      custom:
        container:
          image: "agrigorev/kfserving-keras-transformer:0.0.1"
          name: user-container
          env:
            - name: MODEL_INPUT_SIZE
              value: "299,299"
            - name: KERAS_MODEL_NAME
              value: "xception"
            - name: MODEL_LABELS
              value: "dress,hat,longsleeve,outwear,pants,shirt,shoes,shorts,skirt,t-shirt"
```

Testing it:

```python
import requests

data = {
    "instances": [
        {"url": "http://bit.ly/mlbookcamp-pants"},
    ]
}

url = 'https://clothing-model.default.kubeflow.mlbookcamp.com/v1/models/clothing-model:predict'
result = requests.post(url, json=data).json()

print(result)
```