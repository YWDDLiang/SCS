"""Raw subsequence identities; suffix-equivalent entries remain distinct."""
import numpy as np
import torch
from src.data.selection.subsequence_candidate import SubsequenceCandidateStore


class RawSubsequenceCandidateStore(SubsequenceCandidateStore):
    def validate(self):
        if self.metadata.get('pool_kind') != 'raw_subsequence_no_dedup':
            raise ValueError('Expected explicitly identified raw no-dedup store')
        if self.sid_hierarchy != 4 or self.sequence_length != 120:
            raise ValueError('Raw experiment preserves four SID levels and length120')
        offsets=self.offsets.cpu().numpy()
        if offsets.ndim!=1 or len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=self.tokens.numel():
            raise ValueError('Invalid raw-store sequence offsets')
        lengths=np.diff(offsets)
        if np.any(lengths<0) or np.any(lengths%self.sid_hierarchy):
            raise ValueError('Raw-store rows must contain complete items')
        if self.user_ids.numel()!=self.num_rows or self.user_ids.unique().numel()!=self.num_rows:
            raise ValueError('Raw-store users must be unique')
        triples=self.candidate_triples.cpu().numpy()
        if triples.ndim!=2 or triples.shape[1]!=3 or not len(triples):
            raise ValueError('Raw candidate triples must be nonempty Nx3')
        rows,starts,targets=triples.T
        if np.any(rows<0) or np.any(rows>=self.num_rows) or np.any(starts<0) or np.any(starts>=targets):
            raise ValueError('Invalid raw candidate identity')
        if np.any(targets>=lengths[rows]//self.sid_hierarchy):
            raise ValueError('Raw candidate target exceeds train sequence')
        # Preparation enumerates user -> target -> raw start, making raw identity
        # unique and auditable without deleting suffix-equivalent model inputs.
        previous,current=triples[:-1],triples[1:]
        ordered=(current[:,0]>previous[:,0]) | ((current[:,0]==previous[:,0]) &
            ((current[:,2]>previous[:,2]) | ((current[:,2]==previous[:,2]) & (current[:,1]>previous[:,1]))))
        if not np.all(ordered):raise ValueError('Raw triples must be strictly ordered by user,target,start')
        expected=int(np.sum((lengths//self.sid_hierarchy)*(lengths//self.sid_hierarchy-1)//2))
        if len(triples)!=expected:raise ValueError('Raw pool must enumerate all start/target pairs')

    def length_histogram(self):
        lengths=(self.candidate_triples[:,2]-self.candidate_triples[:,1]+1).clamp(max=self.max_items).long()
        counts=torch.bincount(lengths,minlength=self.max_items+1)
        return {i:int(counts[i]) for i in range(2,self.max_items+1) if int(counts[i])}
