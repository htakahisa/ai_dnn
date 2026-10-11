"""Scoped, versioned public-information hooks; no training loop or global patches."""
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import hashlib
import re
import sys
import copy
import math
import numbers
import numpy as np

EXTENSION_DIRECTORY=Path(__file__).resolve().parent
EXTENSION_API_VERSION=1


@dataclass(frozen=True)
class ExtensionContext:
    role: str
    opponent: str | None
    site: str | None
    snapshot: object
    ally: object = None
    goal: object = None
    action: object = None


def _module(name,expected_hash=None):
    if not isinstance(name,str) or not re.fullmatch(r'tv3_custom_[A-Za-z0-9_]+',name):
        raise ValueError('Custom module must be a local tv3_custom_*.py module name')
    path=EXTENSION_DIRECTORY/f'{name}.py'
    payload=path.read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    if expected_hash is not None and digest!=expected_hash:
        raise ValueError(f'Custom extension source changed: {name}')
    # Execute exactly the bytes whose hash is stored, even with stale import/pyc caches.
    key=f'touyama_v3._extension_{name}_{digest}'
    if key not in sys.modules:
        module=ModuleType(key)
        module.__file__=str(path)
        module.__package__='touyama_v3'
        sys.modules[key]=module
        try:
            exec(compile(payload,str(path),'exec'),module.__dict__)
        except BaseException:
            sys.modules.pop(key,None)
            raise
    module=sys.modules[key]
    spec=module.EXTENSION_SPEC
    required={'id','revision','role','opponent','site'}
    if not isinstance(spec,dict) or set(spec)!=required or not all(isinstance(spec[k],str) and spec[k] for k in required-{'revision'}):
        raise ValueError(f'Invalid extension specification: {name}')
    if type(spec['revision']) is not int or spec['revision']<1 or spec['site'] not in ('*','L','R'):
        raise ValueError(f'Invalid extension revision/site: {name}')
    for hook in ('adjust_goal','adjust_inputs','adjust_reward'):
        if hasattr(module,hook) and not callable(getattr(module,hook)):
            raise ValueError(f'Invalid extension hook: {name}.{hook}')
    return module,dict(module=name,source_sha256=digest,**spec)


class ExtensionPipeline:
    def __init__(self,role,opponent,entries=()):
        self.role,self.opponent,self.entries=role,opponent,tuple(entries)

    @property
    def contract(self):
        if not self.entries:
            return None  # Original/common checkpoints keep their exact schema.
        return dict(api_version=EXTENSION_API_VERSION,role=self.role,opponent=self.opponent,
                    extensions=[copy.deepcopy(spec) for _,spec in self.entries])

    def _matching(self,context):
        if context.role!=self.role or context.opponent!=self.opponent:
            return
        for module,spec in self.entries:
            if spec['site']=='*' or spec['site']==context.site:
                yield module

    def adjust_goal(self,context,goal):
        for module in self._matching(context):
            hook=getattr(module,'adjust_goal',None)
            if hook is not None:
                goal=hook(context,goal)
                grid=context.snapshot.grid
                if (not isinstance(goal,(tuple,list)) or len(goal)!=2
                        or not all(isinstance(v,numbers.Integral) for v in goal)
                        or not 0<=goal[0]<len(grid) or not 0<=goal[1]<len(grid[0])
                        or grid[goal[0]][goal[1]]==1):
                    raise ValueError('Custom goal must be an in-bounds walkable cell')
                goal=tuple(map(int,goal))
        return goal

    def adjust_inputs(self,context,inputs):
        for module in self._matching(context):
            hook=getattr(module,'adjust_inputs',None)
            if hook is not None:
                shape=inputs.observation.shape
                dtype=inputs.observation.dtype
                mask=inputs.mask.copy()
                actions=inputs.actions
                result=hook(context,inputs)
                if result is not None:
                    inputs=result
                if (not isinstance(inputs.observation,np.ndarray) or inputs.observation.shape!=shape
                        or inputs.observation.dtype!=dtype or not np.isfinite(inputs.observation).all()
                        or inputs.mask.shape!=mask.shape or inputs.mask.dtype!=np.dtype(bool)
                        or len(inputs.actions)!=len(actions) or not inputs.mask.any()
                        or np.any((inputs.mask & ~mask)[:40]) or np.any((inputs.mask & ~mask)[56:])):
                    raise ValueError('Custom inputs must preserve dimensions/actions and cannot enable blocked actions')
                for index,(before,after) in enumerate(zip(actions,inputs.actions)):
                    if before!=after or inputs.mask[index] and not mask[index]:
                        if (not 40<=index<56 or getattr(before,'kind',None)!=getattr(after,'kind',None)
                                or before==after):
                            raise ValueError('Only ability target candidates may be replaced')
                        if inputs.mask[index]:
                            from frc_v1.actions import build_masks,validate_action
                            validate_action(context.snapshot,build_masks(context.snapshot),context.ally.slot,after)
                if (not isinstance(inputs.teacher,numbers.Integral)
                        or not 0<=inputs.teacher<len(inputs.actions)):
                    raise ValueError('Custom teacher index is out of range')
        return inputs

    def adjust_reward(self,context,reward):
        for module in self._matching(context):
            hook=getattr(module,'adjust_reward',None)
            if hook is not None:
                reward=hook(context,float(reward))
                if not isinstance(reward,numbers.Real) or not math.isfinite(reward):
                    raise ValueError('Custom reward must be a finite real number')
        return float(reward)


def build_extensions(module_names,*,role,opponent):
    if len(module_names)!=len(set(module_names)):
        raise ValueError('Duplicate custom module')
    entries=[]
    for name in module_names:
        module,spec=_module(name)
        if spec['role'] in ('*',role) and spec['opponent'] in ('*',opponent):
            entries.append((module,spec))
    if len({spec['id'] for _,spec in entries})!=len(entries):
        raise ValueError('Duplicate extension identifier')
    return ExtensionPipeline(role,opponent,entries)


def load_extensions(contract,*,role,opponent):
    if contract is None:
        return ExtensionPipeline(role,opponent)
    if (not isinstance(contract,dict) or set(contract)!= {'api_version','role','opponent','extensions'}
            or contract['api_version']!=EXTENSION_API_VERSION
            or contract['role']!=role or contract['opponent']!=opponent or not contract['extensions']):
        raise ValueError('Custom extension role/opponent/API mismatch')
    entries=[]
    for expected in contract['extensions']:
        module,actual=_module(expected['module'],expected['source_sha256'])
        if actual!=expected or actual['role'] not in ('*',role) or actual['opponent'] not in ('*',opponent):
            raise ValueError('Custom extension scope/revision mismatch')
        entries.append((module,actual))
    pipeline=ExtensionPipeline(role,opponent,entries)
    if len({spec['id'] for _,spec in entries})!=len(entries) or pipeline.contract!=contract:
        raise ValueError('Invalid custom extension contract')
    return pipeline


def extension_weight_scope(contract):
    if contract is None:
        return None
    sites={row['site'] for row in contract['extensions']}
    if len(sites)!=1:
        raise ValueError('One training profile must have one weight scope: L, R, or *')
    return next(iter(sites))
