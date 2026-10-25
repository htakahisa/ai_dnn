"""Optional initial imitation of the simple logic, never used to override PPO actions."""
import numpy as np
import torch
from ..gc3_io import replace_file
from ..preplant.gc3_observation_attacker_history import MatchHistory
from .gc3_policy_attacker_macro import SCHEMA_HASH
from .gc3_directives_attacker_macro import INSTRUCTIONS
from .gc3_curriculum_attacker_directives import directive_policy


class MacroTeacher:
    def __init__(self,student,config,seed):
        self.student=student;self.logic=directive_policy(dict(config,MACRO_DIRECTIVE_MODE='simple'),seed)

    def select_macro(self,char,state,history,current_plan,candidates):
        plan,_=self.logic.select_macro(char,state,history,current_plan,candidates)
        _,row=self.student.select_macro(char,state,history,current_plan,candidates)
        index=lambda p:int(np.ravel_multi_index(p,candidates.wall_mask.shape))
        row['selections']=dict(destination=[index(p) if p is not None else 0 for p in plan['destinations']],
            instruction=[INSTRUCTIONS.index(v) if v is not None else 0 for v in plan['instructions']],plant=index(plan['plant']))
        with torch.no_grad():
            logp,_,values=self.student.evaluate_records([row])
        row.update(logp=float(logp[0]),value=float(values[0]))
        return plan,row


def save_teacher(path,rows,history,candidates):
    history_path=path.with_suffix('.npz');history.save(history_path)
    packed=[]
    for row in rows:
        packed.append(dict(obs=torch.from_numpy(row['obs']),active=torch.from_numpy(row['active']),
            masks={k:torch.from_numpy(v) for k,v in row['masks'].items()},selections=row['selections'],
            current_plan=row['current_plan'],history_index=row['history_index'],roster=row['roster'],phase='macro'))
    temp=path.with_suffix('.tmp')
    torch.save(dict(schema_hash=SCHEMA_HASH,candidate_signature=candidates.signature,history=history_path.name,rows=packed),temp)
    replace_file(temp,path)


def load_teacher(root,candidates):
    rows=[]
    for path in sorted(root.glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['schema_hash']!=SCHEMA_HASH or saved['candidate_signature']!=candidates.signature:
            raise ValueError('Macro teacher schema/candidate mismatch')
        history=MatchHistory.load(root/saved['history'])
        for packed in saved['rows']:
            row=dict(packed,obs=packed['obs'].numpy(),active=packed['active'].numpy(),
                masks={k:v.numpy() for k,v in packed['masks'].items()},history=history)
            rows.append(row)
    return rows


def optimize_bc(policy,optimizer,rows,config):
    policy.train();total=count=0
    for offset in np.random.permutation(list(range(0,len(rows),config['BATCH_SIZE']))):
        batch=rows[offset:offset+config['BATCH_SIZE']]
        logp,_,_=policy.evaluate_records(batch);loss=-logp.mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite macro BC loss')
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        total+=float(loss.detach())*len(batch);count+=len(batch)
    policy.eval()
    return total/count
