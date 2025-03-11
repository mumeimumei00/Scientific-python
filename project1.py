# import mlcroissant as mlc
# import pandas as pd
#
# # Fetch the Croissant JSON-LD
# croissant_dataset = mlc.Dataset('www.kaggle.com/datasets/kwadwoofosu/predict-test-scores-of-students/croissant/download')
#
# # Check what record sets are in the dataset
# record_sets = croissant_dataset.metadata.record_sets
# print(record_sets)
#
# # Fetch the records and put them in a DataFrame
# record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
# record_set_df.head()

import kagglehub

# Download latest version
path = kagglehub.dataset_download("kwadwoofosu/predict-test-scores-of-students")

print("Path to dataset files:", path)
