import argparse, torch, pandas as pd, tqdm
from prompt_soup import PromptSoup
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class CSVImages(Dataset):
    def __init__(self, csv_file, preprocess):
        df = pd.read_csv(csv_file); self.paths=df.img_path; self.labels=df.label
        self.tfm = preprocess
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img), int(self.labels[idx])

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ----- build soup with learnable delta -----
    classes = [l.strip() for l in open(cfg.classnames).readlines()]
    soup = PromptSoup(classes, cfg.dataset, device, learn_delta=True).to(device)
    clip_model = soup.model
    tfm = clip_model.preprocess
    loader = DataLoader(CSVImages(cfg.train_csv, tfm),
                        batch_size=64, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam([soup.delta], lr=2e-3)
    lam = 0.1
    base = soup.base.clone().detach()

    for ep in range(5):
        for imgs, lbl in tqdm.tqdm(loader, desc=f"ep{ep+1}"):
            imgs, lbl = imgs.to(device), lbl.to(device)
            with torch.no_grad():
                f = clip_model.encode_image(imgs)
                f = f / f.norm(dim=1, keepdim=True)
            logits = f @ soup().T * clip_model.logit_scale.exp()
            loss  = torch.nn.functional.cross_entropy(logits, lbl)
            loss += lam * (soup.delta**2).mean()      # L2 reg
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(soup.state_dict(), cfg.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--classnames", required=True)   # text file, one class per line
    p.add_argument("--out", default="trained_prompt.pt")
    main(p.parse_args())