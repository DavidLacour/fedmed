"""fedmed: a proof-of-concept federated learning platform for medical imaging.

"Data stays, AI collaborates": every hospital keeps its images on its own node.
Only model updates, which may be protected by DP-SGD and secure aggregation, leave the site.
"""

__version__ = "0.1.0"

CLASS_NAMES = [
    "actinic keratoses",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SHAPE = (28, 28, 3)
