"""Author-style GORACS SeqRec prompts on the unchanged existing TRAIN query."""
import json
from pathlib import Path

import numpy as np
import torch


def prompt_for_titles(history, target, dataset):
    category = {"toys": "toys and games", "beauty": "beauty products"}[dataset]
    instruction = (f"Given a list of {category} the user has purchased before, "
                   f"please recommend the name of a new product that the user "
                   f"will like to purchase in the following time.")
    inputs = ("The user has purchased the following " + category + " before:"
              + ", ".join('"' + name + '"' for name in history) + "\n ")
    # Same SeqRec input+output structure as the author's embedding notebook.
    return (f"Below is an instruction that describes a task, paired with an input "
            f"that provides further context. Write a response that appropriately completes the request.  \n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{inputs}\n\n"
            f'### Response:\n"{target}"\n')


class OriginalQueryText:
    def __init__(self, root, dataset):
        self.dataset = dataset
        root = Path(root)
        cache = root / "preparation" / dataset / "query_cache"
        self.items = np.load(cache / "train_items.npy", mmap_mode="r")
        self.offsets = np.load(cache / "train_offsets.npy", mmap_mode="r")
        users = np.load(cache / "train_user_ids.npy")
        self.order = np.argsort(users)
        self.users = users[self.order]
        self.titles = json.loads((root / "preparation" / dataset /
                                 "text_assets/item_titles.json").read_text())
        meta = json.loads((cache / "manifest.json").read_text())
        assert len(self.titles) == meta["item_count"]

    def render(self, triples):
        texts = []
        for uid, start, target in np.asarray(triples, dtype=np.int64):
            found = int(np.searchsorted(self.users, uid))
            if found >= len(self.users) or self.users[found] != uid:
                raise ValueError("Unknown original user identity")
            row = int(self.order[found])
            start = max(int(start), int(target)-29)
            base = int(self.offsets[row])
            history = self.items[base+start:base+int(target)]
            label = int(self.items[base+int(target)])
            texts.append(prompt_for_titles([self.titles[int(i)-1] for i in history],
                                           self.titles[label-1], self.dataset))
        return texts


class FrozenRobertaMeanEncoder:
    def __init__(self, path, device):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, add_pooling_layer=False,
            attn_implementation="eager").to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.max_length = min(512, self.tokenizer.model_max_length)

    @torch.no_grad()
    def encode(self, texts, return_metadata=False):
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt",
                               return_offsets_mapping=return_metadata)
        offsets = batch.pop("offset_mapping", None)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        # SentenceTransformer's default mean pooling for the linked plain
        # RoBERTa model, without an extra L2 normalization layer.
        result = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite frozen text embeddings")
        if return_metadata:
            last_character = offsets[:, :, 1].max(1).values.numpy()
            fully_encoded = last_character >= np.array([len(t.rstrip()) for t in texts])
            return result, dict(target_fully_encoded=fully_encoded,
                                nonpadding_tokens=batch["attention_mask"].sum(1).cpu().numpy())
        return result
