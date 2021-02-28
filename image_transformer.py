import os
import argparse
import kfserving

from keras_image_helper import create_preprocessor


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name, predictor_host, preprocessor, labels):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.preprocessor = preprocessor
        self.labels = labels

    def image_transform(self, instance):
        url = instance['url']
        X = self.preprocessor.from_url(url)
        return X[0].tolist()

    def preprocess(self, inputs):
        instances = [self.image_transform(instance) for instance in inputs['instances']]
        return {'instances': instances}

    def postprocess(self, outputs):
        results = []

        raw = outputs['predictions']

        for row in raw:
            result = {c: p for c, p in zip(self.labels, row)}
            results.append(result)

        return {'predictions': results}


def configure_arg_parser():
    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_name',
                        help='The name that the model is served under.')
    parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)
    return parser


def main():
    parser = configure_arg_parser()
    args, _ = parser.parse_known_args()
    
    size = os.environ['MODEL_INPUT_SIZE']
    size_h, size_w = size.split(',')
    size_h = int(size_h)
    size_w = int(size_w)

    keras_model = os.environ['KERAS_MODEL_NAME']
    labels = os.environ['MODEL_LABELS'].split(',')

    preprocessor = create_preprocessor(keras_model, target_size=(size_w, size_w))

    transformer = ImageTransformer(
        args.model_name,
        predictor_host=args.predictor_host,
        preprocessor=preprocessor,
        labels=labels
    )

    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])


if __name__ == "__main__":
    main()
