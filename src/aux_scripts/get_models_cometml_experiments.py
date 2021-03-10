import comet_ml
import numpy as np
from tensorflow.keras.models import load_model
import gc
import os
import argparse
import src.constants as cons
import joblib
from src.models import metrics


def main(models_path, workspace, project_name):

    models = set(os.listdir(models_path))
    print()

    comet_api = comet_ml.api.API()
    experiments = comet_api.get(workspace=workspace, project_name=project_name)

    print(experiments)






    joblib.dump(experiment, os.path.join(model_dir, cons.EXPERIMENT_ULR_FILENAME))

    #prediction = model.predict(data)



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("models_path", default="models/models/", help="Models path")
    argparser.add_argument("workspace", default="pabloduque0", help="CometML workspace")
    argparser.add_argument("project_name", default="wmh", help="CometML project name")

    args = argparser.parse_args()

    main(os.path.abspath(args.models_path), args.workspace, args.project_name)