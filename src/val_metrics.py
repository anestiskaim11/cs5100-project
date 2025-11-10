
import csv, numpy as np, cv2, os, torch, pandas as pd
from config import *
from gan import GeneratorUNet
from dataset import read_gray

def vessel_density(prob, mask, thr=0.5):
    binmap = (prob>=thr).astype(np.uint8)
    m = (mask>0.5).astype(np.uint8)
    vd = binmap[m>0].mean() if m.sum()>0 else 0.0
    return float(vd)

def faz_area_from_prob(prob):
    # crude: invert prob, largest central low‑prob region ~ FAZ; placeholder metric for demo
    inv = (prob<0.4).astype(np.uint8)*255
    cnts,_ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    areas = [cv2.contourArea(c) for c in cnts]
    return float(max(areas))

@torch.no_grad()
def export_val_metrics_csv(out_csv):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G  = GeneratorUNet(in_ch=4).to(device)


    dfv = pd.read_csv(f"{RUN_DIR}/index_val.csv")
    ck = torch.load(f"{RUN_DIR}/checkpoints/best.pt", map_location=device)
    G.load_state_dict(ck["G"], strict=False)
    shadow = ck.get("ema", None)
    if shadow is not None:
        for k, v in shadow.items():
            if k in G.state_dict() and G.state_dict()[k].dtype.is_floating_point:
                G.state_dict()[k].copy_(v)
    G.eval()

    rows = []
    for _, r in dfv.iterrows():
        # load tensors
        f_rgb = cv2.cvtColor(cv2.imread(r.fundus), cv2.COLOR_BGR2RGB)
        f_rgb = cv2.resize(f_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        f = f_rgb.copy(); mean=f.mean((0,1),keepdims=True); std=f.std((0,1),keepdims=True)+1e-6
        f = (f-mean)/std
        f = torch.from_numpy(f.transpose(2,0,1))[None].float().to(device)
        gray = cv2.cvtColor((f_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        m = (gray>0.02).astype(np.float32); m = cv2.erode(m, np.ones((5,5), np.uint8),1)
        m = torch.from_numpy(m)[None,None].float().to(device)

        y = read_gray(r.octa); y = torch.from_numpy(y)[None,None].float().to(device)*2-1
        pgt = read_gray(r.prob); pgt = torch.from_numpy(pgt)[None,None].float().to(device)

        y_hat, p_hat, _ = G(f, m)
        yh = (y_hat[0,0].detach().cpu().numpy()+1)/2
        ph = p_hat[0,0].detach().cpu().numpy()
        mask = m[0,0].cpu().numpy()

        vd = vessel_density(ph, mask, thr=0.5)
        faz = faz_area_from_prob(ph) if r.location=='macula' else None
        rows.append(dict(subject_id=r.subject_id, location=r.location, vessel_density=vd, faz_area=faz))
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print("Saved metrics to", out_csv)

if __name__ == "__main__":
    export_val_metrics_csv(f"{RUN_DIR}/reports/val_metrics.csv")
