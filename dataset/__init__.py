from .instance.aircraft import *
from .instance.car import *
from .instance.cub import *
from .instance.dog import *
from .instance.flower import *
from .instance.food import *
from .instance.pascal import *
from .instance.pet import *
from .instance.waterbird import *
from .instance.bear import *

DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugDataset,
    "flower": Flowers102Dataset,
    "car": CarHugDataset,
    "pet": PetHugDataset,
    "aircraft": AircraftHugDataset,
    "food": FoodHugDataset,
    "pascal": PascalDataset,
    "dog": StanfordDogDataset,
    "bear": BearHugDataset,
}
IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDataset,
    "flower": FlowersImbalanceDataset,
     "bear": BearHugDataset,
}
T2I_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugDatasetForT2I,
    "flower": FlowersDatasetForT2I,
    "bear": BearHugDataset,
}
T2I_IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDatasetForT2I,
    "flower": FlowersImbalanceDatasetForT2I,
    "bear": BearHugDataset,
}
