import pickle
from birdset_preparation import load_birdset_data
import json
import numpy as np
CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
def check_labels_in_perch(labels, perch_labels):
    for label in labels:
        if label not in perch_labels:
            print(label)
            return False
    return True
subset = 'SSW'
print(subset)
birdset_test = load_birdset_data(subset)
print(birdset_test.column_names)
birdset_label_mapping = birdset_test.features['ebird_code']._int2str
#print(birdset_test['lat'][:10])
num_labels = np.sum([1 if label != None else 0 for label in birdset_test['ebird_code']])
print(num_labels)
total_labels = np.sum([len(label) for label in birdset_test['ebird_code_multilabel']])
print(total_labels)

num_epochs = 1
print(num_epochs, "epochs")
num_experts = 4
prediction_save_path = "constrained_geospatial_mixup_balanced_mixture_of_" + str(num_experts) + "_experts_early_fusion_classifier_lr_0006_epoch_" + str(num_epochs + 1) + "_" + subset + ".pth" 
#prediction_save_path = "mixture_of_experts_early_fusion_lr_0001_" + subset +" .pkl"
predictions_path = "/home/svu/e1583377/ST-Geo-Perch/predictions/" + prediction_save_path
with open(predictions_path, 'rb') as f:
    predictions = pickle.load(f)

print(predictions['perch'][:10])
print(predictions['geo_aware'][:10])
with open(CLASS_LABELS_FILEPATH) as f:
    perch_label_mapping = json.load(f)
#perch_label_set = set(perch_label_mapping.values())
#ground_truth = [[label_mapping[str(label)] for label in multilabel] for multilabel in ground_truth]
ground_truth = [[birdset_label_mapping[label] if label != None else None for label in multilabel] for multilabel in birdset_test['ebird_code_multilabel']]
print(ground_truth[:10])
# compute accuracies
num_bird_audio = 0
num_perch_correct = 0
num_geo_perch_correct = 0
for labels, perch_prediction, geo_perch_prediction in zip(ground_truth, predictions['perch'], predictions['geo_aware']):
    # only consider clip in evaluation if a single species is labeled
    # uncomment to only consider samples that are known to perch
    # this seems to effect very few samples
    if labels != []: #and check_labels_in_perch(labels, perch_label_set):
        num_bird_audio += 1
        if perch_prediction in labels:
            num_perch_correct += 1
        if geo_perch_prediction in labels:
            num_geo_perch_correct += 1
print("Number of samples:", num_bird_audio)
print("Number correctly predicted by Perch", num_perch_correct, "Accuracy:", num_perch_correct / num_bird_audio)
print("Number correctly predicted by GeoPerch", num_geo_perch_correct, "Accuracy:", num_geo_perch_correct / num_bird_audio)