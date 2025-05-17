import torch, clip

TEMPLATES = {
    "inat2021_mini": [
        "a photo of a {} in the wild.",
        "an image of the {} in its natural habitat.",
        "a wildlife photograph of a {}.",
        "a close-up photo of the {}.",
        "picture of a {} in nature."
    ],
    "pathmnist": [
        "a microscopic image of {} tissue.",
        "a histopathology slide showing {}."
    ],
    # other MNIST datasets go here, i just used chatgpt to generate these templates
}

class PromptSoup(torch.nn.Module):
    """Descriptor-Soup head with an optional learnable delta per class."""
    def __init__(self, classnames, dataset, device, learn_delta=False):
        super().__init__()
        self.device  = device
        self.model,_ = clip.load("ViT-B/32", device=device)
        self.model.eval()

        temps  = TEMPLATES.get(dataset, ["a photo of a {}."])
        with torch.no_grad():
            base = []
            for name in classnames:
                sent_emb = []
                for t in temps:
                    txt = clip.tokenize(t.format(name)).to(device)
                    emb = self.model.encode_text(txt).float()
                    sent_emb.append(emb / emb.norm(dim=-1, keepdim=True))
                base.append(torch.mean(torch.stack(sent_emb), 0))
        base = torch.cat(base, 0)                # [C,512]

        if learn_delta:
            self.delta = torch.nn.Parameter(torch.zeros_like(base))
        else:
            self.register_buffer("delta", torch.zeros_like(base))

        self.register_buffer("base", base)       # frozen soup

    def forward(self):
        w = self.base + self.delta
        return w / w.norm(dim=1, keepdim=True)   # [C,512]
