from abc import ABC
from pydantic import BaseModel


item_feature_name = "movieId"
user_feature_name = "userId"


class FeatureConfig(BaseModel, ABC):
    pass

class UserConfig(BaseModel, ABC):
    features: FeatureConfig
    emb_dim: int

class ItemConfig(BaseModel, ABC):
    features: FeatureConfig
    emb_dim: int

class ModelConfig(BaseModel, ABC):
    user: UserConfig
    item: ItemConfig
    emb_dim: int
    tasks: list[tuple[str, int]]

class DataConfig(BaseModel, ABC):
    pass

class PipelineConfig(BaseModel, ABC):
    model_config: ModelConfig
    data_config: DataConfig
    
    @staticmethod
    def from_json(json_config_file):
        pass