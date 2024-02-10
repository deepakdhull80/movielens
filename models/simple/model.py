import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import item_feature_name, user_feature_name

class ItemTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = config['item']['features']
        if item_feature_name in self.features:
            self.item_module = nn.Embedding(config['item']['n_items'], config['item']['emb'])
        else:
            self.item_module = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(config['item']['emb'], config['item']['emb']),
            nn.GELU(),
            nn.LayerNorm(config['item']['emb'])
        )
        
    def forward(self, batch):
        item_emb = self.item_module(batch[item_feature_name].long())
        return self.mlp(item_emb)


class UserTower(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.features = config['user']['features']
        if user_feature_name in self.features:
            self.user_module = nn.Embedding(config['user']['n_users'], config['user']['emb'])
        else:
            self.item_module = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(config['user']['emb'], config['user']['emb']),
            nn.GELU(),
            nn.LayerNorm(config['user']['emb'])
        )
        
    def forward(self, batch):
        user_emb = self.user_module(batch[user_feature_name].long())
        return self.mlp(user_emb)

class CombinerTower(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.item_tower = ItemTower(config)
        self.user_tower = UserTower(config)
        user_dim, item_dim = config['user']['emb'], config['item']['emb']
        self.mlp = nn.Sequential(
            nn.Linear(user_dim+item_dim, config['emb']),
            nn.GELU(),
            nn.LayerNorm(config['emb'])
        )
        self.tasks = nn.ModuleDict({
            task_name: nn.Linear(config['emb'], out) for task_name, out in config['tasks']
        })
        
    def forward(self, batch):
        item_emb = self.item_tower(batch)
        user_emb = self.user_tower(batch)
        x = torch.concat([user_emb, item_emb], axis=1)
        x = self.mlp(x)
        res = {}
        for task_name, mod in self.tasks.items():
            res[task_name] = mod(x)
        return res


if __name__ == '__main__':
    no_users = int(2e12)
    no_items = int(2e10)
    config = {
        'user':{
            'features':[
                'userId'
            ],
            'n_users':no_users + 2**10,
            'emb': 64
        },
        'item':{
            'features':[
                'movieId'
            ],
            'n_items':no_items + 2**10,
            'emb': 64
        },
        'emb':64,
        'tasks':[
            ['rating', 1]
        ]
    }
    
    