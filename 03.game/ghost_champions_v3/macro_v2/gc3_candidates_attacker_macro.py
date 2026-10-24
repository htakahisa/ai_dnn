"""Validated user map candidates; static planning targets, not movement masks."""
import hashlib
import numpy as np
from ..gc3_observation import ARENA_SHAPE,CELLS
from ..gc3_spatial_policy import parse_map
from . import gc3_map_attacker_destinations as settings


class MacroCandidates:
    def __init__(self,text,arena,plant_coordinates=()):
        arena=np.asarray(arena)
        markers=parse_map(text)
        if arena.shape!=ARENA_SHAPE or markers.shape!=arena.shape or not np.array_equal(markers==1,arena==1):
            raise ValueError("Macro candidate map walls/dimensions do not match the arena")
        self.destination_mask=(markers==5).reshape(CELLS)
        if not self.destination_mask.any():
            raise ValueError("Set destination candidates (5) in macro_v2/gc3_map_attacker_destinations.py first")
        self.plant_mask=np.zeros(CELLS,dtype=np.bool_)
        if plant_coordinates:
            for coordinate in plant_coordinates:
                point=tuple(coordinate)
                if len(point)!=2 or any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) for v in point):
                    raise ValueError("Plant coordinates must be integer (row, column) pairs")
                if not (0<=point[0]<arena.shape[0] and 0<=point[1]<arena.shape[1]) or arena[point]!=2:
                    raise ValueError("Planned plant coordinate must be a real plantable cell")
                self.plant_mask[np.ravel_multi_index(point,arena.shape)]=True
        else:
            self.plant_mask=(arena==2).reshape(CELLS)
        if not self.plant_mask.any():raise ValueError("No plant candidates")
        self.wall_mask=arena==1
        self.signature=hashlib.sha256(self.wall_mask.tobytes()+self.destination_mask.tobytes()+self.plant_mask.tobytes()).hexdigest()
        for value in (self.destination_mask,self.plant_mask,self.wall_mask):value.setflags(write=False)

    @classmethod
    def configured(cls,arena):
        return cls(settings.MACRO_DESTINATION_MAP_STR,arena,settings.PLANT_CANDIDATE_COORDINATES)

    def validate_arena(self,arena):
        arena=np.asarray(arena)
        if arena.shape!=ARENA_SHAPE or not np.array_equal(arena==1,self.wall_mask) or not np.all((arena==2).reshape(-1)[self.plant_mask]):
            raise ValueError("Macro candidates do not match this arena")

    @staticmethod
    def coordinate(index):
        return tuple(map(int,np.unravel_index(int(index),ARENA_SHAPE)))

    def summary(self):
        return dict(destinations=int(self.destination_mask.sum()),plants=int(self.plant_mask.sum()),signature=self.signature)
