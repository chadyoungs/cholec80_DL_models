img_w = 224
img_h = 224

dataset_params = {
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 4,
    'pin_memory': True
}

net_params = {
    'out_dim_tool': 7,
    'out_dim_phase': 7,
    'drop_prob': 0.3,
    'bn_momentum': 0.01
}

# labels
phase_labels = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
tool_labels_without_NoTool = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
tool_labels_with_NoTool = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag", "NoTool"]

learning_rate_feature = 1e-3
learning_rate_classifier = 1e-2
momentum = 9e-1

epoches = 100
log_interval = 5 # the interval of printing
save_interval = 10 # the interval of saving model
