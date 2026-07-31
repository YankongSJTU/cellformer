#!/usr/bin/env python3
"""
Full-patch Grad-CAM WSI classification heatmap.
Processes ALL patches in each WSI (no subsampling).
Uses pre-computed CPS features for TCGA, fresh pipeline for CPTAC.
Better colormap with percentile-based normalization.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(4)
import cv2, numpy as np
import torch.nn as nn, torch.nn.functional as F
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import openslide, random, gc, shutil, subprocess
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEAT_DIR   = os.path.join(PROJECT_ROOT, 'features_cpsformer')
TCGA_DIR   = '/data1/TumorGroup/DATA/public_database/TCGA/slide'
CPTAC_DIR  = '/data1/TumorGroup/DATA/public_database/slide/CPTAC'
CPS_CKPT   = os.path.join(PROJECT_ROOT, 'checkpoints_supcon', 'best_model.pth')
CELL_CKPT  = os.path.join(PROJECT_ROOT, 'checkpoints_cell', 'model.pth')
OUT_DIR    = os.path.join(PROJECT_ROOT, 'figures_gradcam_v2')
COHORTS    = ['BLCA','BRCA','CESC','COAD','DLBC','ESCA','GBM','HNSC','KIRC','KIRP',
              'LGG','LIHC','LUAD','LUSC','OV','PAAD','PRAD','READ','STAD','THCA','THYM','UCEC']


class AttentionMIL(nn.Module):
    def __init__(self, feat_dim=1024, n_classes=22, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.attention_V = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x, return_attention=False):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, D = x.shape
        h = self.fc1(x)
        a = self.attention_weights(self.attention_V(h) * self.attention_U(h))
        a = F.softmax(a, dim=1)
        self.last_attention = a.detach()
        z = (x * a).sum(dim=1)
        logits = self.classifier(z)
        if return_attention:
            return logits, a.squeeze(-1)
        return logits


class MultiTaskModel(nn.Module):
    def __init__(self, feat_dim=1024, n_classes=22):
        super().__init__()
        self.mil = AttentionMIL(feat_dim, n_classes)
        self.direct = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_classes))
        self.surv_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x, mode='auto', return_attention=False):
        if x.dim() == 2 and mode != 'mil':
            logits = self.direct(x)
            if return_attention:
                return logits, None, self.surv_head(x)
            return logits
        else:
            out = self.mil(x, return_attention=return_attention)
            if return_attention:
                logits, attn = out
                feat = (x * attn.unsqueeze(-1)).sum(1) if x.dim() == 3 else x
                return logits, attn, self.surv_head(feat)
            return out


def load_all_features():
    all_data = []
    for cohort in tqdm(COHORTS, desc="Loading features"):
        for suf in ['', '1', '2']:
            fp = os.path.join(FEAT_DIR, f'{cohort}{suf}.cps_feature.csv')
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                df['tumor_type'] = cohort
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def train_classifier(data, device, epochs=100):
    fcols = [c for c in data.columns if c not in ('samplename','imgname','tumor_type')]
    pf = data.groupby('samplename')[fcols].mean().reset_index()
    pt = data.groupby('samplename')['tumor_type'].first().reset_index()
    pf = pd.merge(pf, pt, on='samplename')

    le = LabelEncoder()
    y = le.fit_transform(pf['tumor_type'].values)
    X = pf[fcols].values.astype(np.float32)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = MultiTaskModel(feat_dim=X.shape[1], n_classes=len(le.classes_)).to(device)
    Xt = torch.FloatTensor(X_s).to(device)
    yt = torch.LongTensor(y).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0; best_state = None

    for ep in range(epochs):
        model.train()
        loss = F.cross_entropy(model(Xt), yt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            acc = (model(Xt).argmax(1) == yt).float().mean().item()
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 25 == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.4f}")

    if best_state: model.load_state_dict(best_state)
    model.eval()
    print(f"  Best accuracy: {best_acc:.4f} ({len(le.classes_)} classes)")
    return model, scaler, le


def find_tcga_svs(cohort, patient):
    cdir = os.path.join(TCGA_DIR, cohort)
    if not os.path.isdir(cdir):
        return None
    # Direct match
    for root, dirs, files in os.walk(cdir):
        for f in files:
            if f.endswith('.svs') and patient in f:
                return os.path.join(root, f)
    return None


def process_tcga_full(data, model, scaler, le, device, cohort, patient):
    """Process ALL patches for a TCGA patient using pre-computed features."""
    fcols = [c for c in data.columns if c not in ('samplename','imgname','tumor_type')]
    cdata = data[(data['tumor_type'] == cohort) & (data['samplename'] == patient)].copy()
    if len(cdata) < 5:
        return None

    coords, feats = [], []
    for _, row in cdata.iterrows():
        iname = row['imgname']
        parts = iname.replace('.png','').replace('.jpg','').split('_')
        try:
            x, y = int(parts[-2]), int(parts[-1])
            coords.append((x, y))
            feats.append([row[c] for c in fcols])
        except:
            continue

    if len(coords) < 5:
        return None

    feats = np.array(feats, dtype=np.float32)
    fs = scaler.transform(feats)
    Xt = torch.FloatTensor(fs).to(device)

    model.eval()
    with torch.enable_grad():
        logits, attn, risk = model(Xt, mode='mil', return_attention=True)

    pred_class = logits.argmax(1).item()
    pred_label = le.classes_[pred_class]
    confidence = F.softmax(logits, dim=1).max().item()

    attention = attn.cpu().detach().numpy().flatten() if attn is not None else np.ones(len(coords))/len(coords)

    # Also get per-patch predictions
    with torch.no_grad():
        patch_logits = model.direct(torch.FloatTensor(fs).to(device))
        patch_probs = F.softmax(patch_logits, dim=1)
        patch_preds = patch_probs[:, pred_class].cpu().numpy()

    svs_path = find_tcga_svs(cohort, patient)
    print(f"  {cohort}/{patient}: {len(coords)} patches, Pred={pred_label} ({confidence:.1%}), {'correct' if pred_label==cohort else 'wrong'}")

    return {
        'svs_path': svs_path,
        'basename': f'{cohort}_{patient}',
        'pred_label': pred_label,
        'confidence': confidence,
        'risk_score': risk.item(),
        'coords': coords,
        'attention': attention,
        'patch_probs': patch_preds,
        'source': 'TCGA',
        'known_type': cohort,
        'correct': pred_label == cohort
    }


def extract_cells_cv(img_rgb, seg_mask, max_cells=150, min_cells=15):
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crds = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if 10 <= cx <= 990 and 10 <= cy <= 990:
                crds.append([cy, cx])
    if len(crds) < min_cells: return None, None, 0
    crds = np.array(crds)
    if len(crds) > max_cells:
        crds = crds[np.random.choice(len(crds), max_cells, replace=False)]
    cps, pos = [], []
    for cy, cx in crds:
        y1,y2 = max(0,cy-28), min(1000,cy+28)
        x1,x2 = max(0,cx-28), min(1000,cx+28)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.shape[0]<56 or crop.shape[1]<56:
            crop = cv2.copyMakeBorder(crop,0,56-crop.shape[0],0,56-crop.shape[1],cv2.BORDER_REFLECT)
        if crop.shape[0]!=56 or crop.shape[1]!=56: crop = cv2.resize(crop,(56,56))
        cps.append(crop); pos.append([cx,cy])
    cps = np.ascontiguousarray(np.array(cps,dtype=np.float32).transpose(0,3,1,2)/255.0)
    pos = np.ascontiguousarray(np.clip(np.array(pos,dtype=np.float32),0,999))
    return cps, pos, len(crds)


def load_cps_model(device):
    from models import MILCellModelmerge
    model = MILCellModelmerge(num_classes=24, d_model=256, output_dim=1024,
                              distilled_path=CELL_CKPT).to(device)
    ckpt = torch.load(CPS_CKPT, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    nsd = {k.replace('module.',''):v for k,v in sd.items()
           if not k.replace('module.','').startswith('cell_encoder.')}
    model.load_state_dict(nsd, strict=False)
    model.eval()
    return model


def run_nucsegstep(patches_dir, seg_work):
    """Run DeepLabV3 nuclear segmentation using nucseg_modules."""
    import nucseg_modules
    nucseg_root = os.path.dirname(nucseg_modules.__file__)
    sys.path.insert(0, nucseg_root)
    from nucseg_deeplabv3 import run_deeplabv3_seg

    # Collect image paths
    image_paths = sorted([
        os.path.join(patches_dir, f)
        for f in os.listdir(patches_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_paths:
        return

    # Run segmentation
    results = run_deeplabv3_seg(image_paths, seg_work, gpu_id=0)

    # Save masks
    masks_dir = os.path.join(seg_work, 'segment')
    os.makedirs(masks_dir, exist_ok=True)
    for name, mask in results.items():
        cv2.imwrite(os.path.join(masks_dir, name + '.png'), mask)


def process_cptac_full(svs_path, cps_model, cls_model, scaler, le, device, known_type=None):
    """Full pipeline for CPTAC WSI - process ALL patches."""
    basename = os.path.basename(svs_path).split('.')[0][:20]
    work = f'/tmp/gradcam_{basename}'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    try:
        patches_dir = os.path.join(work, 'patches')
        masks_dir = os.path.join(work, 'masks')
        os.makedirs(patches_dir); os.makedirs(masks_dir)

        slide = openslide.open_slide(svs_path)
        lv = slide.level_count - 1
        thumb = cv2.cvtColor(np.array(slide.read_region((0,0), lv, slide.level_dimensions[lv])), cv2.COLOR_RGBA2GRAY)
        _, binary = cv2.threshold(thumb, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = slide.level_downsamples[lv]
        w0, h0 = slide.level_dimensions[0]
        pinfos = []
        for cnt in contours:
            tx,ty,tw,th_ = cv2.boundingRect(cnt)
            if tw < 20 or th_ < 20: continue
            x0,y0 = int(tx*scale), int(ty*scale)
            for x in range(x0, x0+int(tw*scale), 1000):
                for y in range(y0, y0+int(th_*scale), 1000):
                    if x+1000>w0 or y+1000>h0: continue
                    p = np.array(slide.read_region((x,y), 0, (1000,1000)))
                    p = cv2.cvtColor(p, cv2.COLOR_RGBA2RGB)
                    g = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
                    if np.sum(g<200)/g.size > 0.05:
                        fn = f'patch_{x}_{y}.png'
                        cv2.imwrite(os.path.join(patches_dir, fn), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
                        pinfos.append({'x':x,'y':y,'filename':fn})
        slide.close()
        print(f"    {len(pinfos)} total patches", flush=True)
        if len(pinfos) < 10: return None

        print(f"    Cell segmentation...", flush=True)
        seg_work = os.path.join(work, 'seg')
        run_nucsegstep(patches_dir, seg_work)

        for f in os.listdir(os.path.join(seg_work, 'segment')):
            if f.endswith('.png'):
                src = os.path.join(seg_work, 'segment', f)
                dst = os.path.join(masks_dir, f)
                shutil.copy2(src, dst)

        print(f"    CPS features...", flush=True)
        features, vrows = [], []
        for idx in range(len(pinfos)):
            row = pinfos[idx]
            img = cv2.imread(os.path.join(patches_dir, row['filename']))
            mk = cv2.imread(os.path.join(masks_dir, row['filename']), 0)
            if img is None or mk is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cps, pos, n_cells = extract_cells_cv(img_rgb, mk)
            if cps is None: continue
            ct = torch.from_numpy(cps).unsqueeze(0).to(device)
            pt = torch.from_numpy(pos).unsqueeze(0).to(device)
            mt = torch.ones(1, n_cells, dtype=torch.float32).to(device)
            try:
                with torch.no_grad(): feat,_,_ = cps_model(ct,pt,mt)
                features.append(feat.cpu().numpy().flatten())
                vrows.append({'x':row['x'],'y':row['y']})
            except: pass
            del ct,pt,mt
            if idx % 50 == 0:
                torch.cuda.empty_cache(); gc.collect()
                print(f"      {idx}/{len(pinfos)} patches, {len(features)} valid", flush=True)

        if len(features) < 5: return None
        print(f"    {len(features)} valid patches", flush=True)

        feats = np.array(features)
        vdf = pd.DataFrame(vrows)
        fs = scaler.transform(feats)
        Xt = torch.FloatTensor(fs).to(device)

        cls_model.eval()
        with torch.enable_grad():
            logits, attn, risk = cls_model(Xt, mode='mil', return_attention=True)

        pred_class = logits.argmax(1).item()
        pred_label = le.classes_[pred_class]
        confidence = F.softmax(logits, dim=1).max().item()

        attention = attn.cpu().detach().numpy().flatten() if attn is not None else np.ones(len(features))/len(features)

        with torch.no_grad():
            patch_logits = cls_model.direct(torch.FloatTensor(fs).to(device))
            patch_probs = F.softmax(patch_logits, dim=1)
            patch_preds = patch_probs[:, pred_class].cpu().numpy()

        return {
            'svs_path': svs_path,
            'basename': f'CPTAC_{basename}',
            'pred_label': pred_label,
            'confidence': confidence,
            'risk_score': risk.item(),
            'coords': list(zip(vdf['x'].astype(int), vdf['y'].astype(int))),
            'attention': attention,
            'patch_probs': patch_preds,
            'source': 'CPTAC',
            'known_type': known_type,
            'correct': pred_label == known_type if known_type else None,
            '_work': work,
            '_patches_dir': patches_dir
        }
    except Exception as e:
        print(f"    Error: {e}")
        shutil.rmtree(work, ignore_errors=True)
        torch.cuda.empty_cache(); gc.collect()
        return None


def plot_full_wsi(r, output_dir):
    """Plot individual WSI with full-patch Grad-CAM heatmap using improved visualization."""
    if r['svs_path'] is None or not os.path.exists(r['svs_path']):
        print(f"    Skipping {r['basename']}: no SVS file")
        return

    slide = openslide.open_slide(r['svs_path'])
    tl = min(2, slide.level_count - 1)
    d = slide.level_dimensions[tl]
    if max(d) > 5000: tl = slide.level_count - 1
    thumb = cv2.cvtColor(np.array(slide.read_region((0,0), tl, slide.level_dimensions[tl])), cv2.COLOR_RGBA2RGB)
    w0, h0 = slide.level_dimensions[0]
    tw, th_ = slide.level_dimensions[tl]
    sx, sy = tw/w0, th_/h0
    ps = int(1000 * sx)
    slide.close()

    coords = r['coords']
    scores = r['patch_probs']

    # Percentile-based normalization for better color discrimination
    p5, p95 = np.percentile(scores, [5, 95])
    scores_norm = np.clip((scores - p5) / (p95 - p5 + 1e-8), 0, 1)

    # Build continuous heatmap
    heatmap = np.zeros(thumb.shape[:2], dtype=np.float64)
    count_map = np.zeros(thumb.shape[:2], dtype=np.float64)
    for (x, y), s in zip(coords, scores_norm):
        xt, yt = int(x * sx), int(y * sy)
        xe, ye = min(xt + ps, tw), min(yt + ps, th_)
        heatmap[yt:ye, xt:xe] += s
        count_map[yt:ye, xt:xe] += 1
    heatmap_avg = np.where(count_map > 0, heatmap / count_map, np.nan)

    # Apply Gaussian smoothing for smoother appearance
    valid_mask = count_map > 0
    heatmap_filled = np.where(valid_mask, heatmap_avg, 0)
    heatmap_smooth = cv2.GaussianBlur(heatmap_filled.astype(np.float32), (0, 0), sigmaX=max(1, ps//4))
    heatmap_smooth = np.where(valid_mask, heatmap_smooth, np.nan)

    fig = plt.figure(figsize=(20, 8), dpi=200)
    gs = GridSpec(1, 3, width_ratios=[1.1, 1.1, 0.05], wspace=0.08)

    # Panel A: Original WSI
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(thumb)
    ax1.set_title('(A) Original WSI', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Panel B: Grad-CAM heatmap overlay
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(thumb)
    cmap = plt.cm.get_cmap('RdYlBu_r')  # Red=high, Blue=low
    im = ax2.imshow(heatmap_smooth, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
    pred = r['pred_label']; true = r['known_type']
    conf = r['confidence']
    status = 'correct' if true and pred == true else ('wrong' if true else '')
    title = f'(B) Classification Heatmap: Pred={pred} ({conf:.1%})'
    if true:
        title += f' | True={true} [{status}]'
    color = 'green' if true and pred == true else ('red' if true else 'black')
    ax2.set_title(title, fontsize=12, fontweight='bold', color=color)
    ax2.axis('off')

    # Colorbar
    cax = fig.add_subplot(gs[2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Classification Probability', fontsize=10)

    tag = r['basename'].replace('/','_')
    fig.suptitle(f'{r["source"]} — {r["basename"]} ({len(coords)} patches)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_png = os.path.join(output_dir, f'gradcam_v2_{r["source"]}_{tag}.png')
    out_tif = os.path.join(output_dir, f'gradcam_v2_{r["source"]}_{tag}.tif')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_tif, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_png}")
    return out_png


def plot_combined_panel(results, output_dir):
    """Combined panel figure with all WSIs."""
    n = len(results)
    if n == 0: return
    cols = min(n, 5)
    rows = 2  # Two rows: original + heatmap

    fig = plt.figure(figsize=(cols * 4.5, rows * 5), dpi=200)

    for idx, r in enumerate(results):
        if r['svs_path'] is None or not os.path.exists(r['svs_path']):
            continue

        slide = openslide.open_slide(r['svs_path'])
        tl = min(2, slide.level_count - 1)
        d = slide.level_dimensions[tl]
        if max(d) > 5000: tl = slide.level_count - 1
        thumb = cv2.cvtColor(np.array(slide.read_region((0,0), tl, slide.level_dimensions[tl])), cv2.COLOR_RGBA2RGB)
        w0, h0 = slide.level_dimensions[0]
        tw, th_ = slide.level_dimensions[tl]
        sx, sy = tw/w0, th_/h0
        ps = int(1000 * sx)
        slide.close()

        coords = r['coords']
        scores = r['patch_probs']
        p5, p95 = np.percentile(scores, [5, 95])
        scores_norm = np.clip((scores - p5) / (p95 - p5 + 1e-8), 0, 1)

        heatmap = np.zeros(thumb.shape[:2], dtype=np.float64)
        count_map = np.zeros(thumb.shape[:2], dtype=np.float64)
        for (x, y), s in zip(coords, scores_norm):
            xt, yt = int(x * sx), int(y * sy)
            xe, ye = min(xt + ps, tw), min(yt + ps, th_)
            heatmap[yt:ye, xt:xe] += s
            count_map[yt:ye, xt:xe] += 1
        heatmap_avg = np.where(count_map > 0, heatmap / count_map, np.nan)
        valid_mask = count_map > 0
        heatmap_filled = np.where(valid_mask, heatmap_avg, 0)
        heatmap_smooth = cv2.GaussianBlur(heatmap_filled.astype(np.float32), (0,0), sigmaX=max(1, ps//4))
        heatmap_smooth = np.where(valid_mask, heatmap_smooth, np.nan)

        # Original WSI
        ax1 = fig.add_subplot(rows, cols, idx + 1)
        ax1.imshow(thumb); ax1.axis('off')
        label = f'{r["source"]}'
        if r['known_type']:
            check = 'correct' if r['correct'] else 'wrong'
            label += f' | {r["known_type"]} ({check})'
        ax1.set_title(label, fontsize=9, fontweight='bold')

        # Heatmap
        ax2 = fig.add_subplot(rows, cols, cols + idx + 1)
        ax2.imshow(thumb)
        ax2.imshow(heatmap_smooth, cmap='RdYlBu_r', alpha=0.55, vmin=0, vmax=1)
        ax2.axis('off')
        pred = r['pred_label']; conf = r['confidence']
        color = 'green' if r.get('correct') else 'red'
        ax2.set_title(f'Pred: {pred} ({conf:.0%})', fontsize=9, fontweight='bold', color=color)

    fig.suptitle('Weakly-Supervised WSI Classification (CPSformer)\nAll Patches — No Subsampling',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_png = os.path.join(output_dir, 'combined_gradcam_v2_panel.png')
    out_tif = os.path.join(output_dir, 'combined_gradcam_v2_panel.tif')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_tif, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined panel: {out_png}")


def select_tcga_patients(data, n_cohorts=5):
    """Select TCGA patients with most patches for good visualization."""
    fcols = [c for c in data.columns if c not in ('samplename','imgname','tumor_type')]
    selected_cohorts = random.sample(COHORTS, n_cohorts)
    patients = []
    for cohort in selected_cohorts:
        cdata = data[data['tumor_type'] == cohort]
        patient_counts = cdata['samplename'].value_counts()
        # Pick patient with most patches (better WSI coverage)
        if len(patient_counts) == 0: continue
        patient = patient_counts.index[0]
        n_patches = patient_counts.iloc[0]
        # Verify SVS exists
        svs = find_tcga_svs(cohort, patient)
        if svs and n_patches >= 10:
            patients.append((cohort, patient, n_patches))
            print(f"  Selected {cohort}/{patient}: {n_patches} patches")
        else:
            # Try next patient
            for p in patient_counts.index[1:5]:
                svs = find_tcga_svs(cohort, p)
                if svs and patient_counts[p] >= 10:
                    patients.append((cohort, p, patient_counts[p]))
                    print(f"  Selected {cohort}/{p}: {patient_counts[p]} patches")
                    break
    return patients


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Training classifier")
    print("="*60, flush=True)
    data = load_all_features()
    print(f"  {len(data)} patches, {data['samplename'].nunique()} patients")
    cls_model, scaler, le = train_classifier(data, device)

    # ─── TCGA: all patches ───
    print("\n" + "="*60)
    print("STEP 2: TCGA Grad-CAM (ALL patches)")
    print("="*60, flush=True)
    random.seed(42)
    patients = select_tcga_patients(data, n_cohorts=5)

    tcga_results = []
    for cohort, patient, n_patches in patients:
        r = process_tcga_full(data, cls_model, scaler, le, device, cohort, patient)
        if r:
            tcga_results.append(r)

    # ─── CPTAC: all patches ───
    print("\n" + "="*60)
    print("STEP 3: CPTAC Grad-CAM (ALL patches)")
    print("="*60, flush=True)

    cps_model = load_cps_model(device)
    cptac_map = {'BRCA':'BRCA','COAD':'COAD','LUAD':'LUAD','OV':'OV','PDA':'PAAD','UCEC':'UCEC','CCRCC':'KIRC'}

    available = []
    for cptac_type, tcga_type in cptac_map.items():
        cdir = os.path.join(CPTAC_DIR, cptac_type)
        if not os.path.isdir(cdir): continue
        svs = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith('.svs')]
        if svs:
            # Pick small-medium files
            sizes = [(f, os.path.getsize(f)) for f in svs]
            sizes.sort(key=lambda x: x[1])
            mid = len(sizes)//2
            candidates = sizes[max(0,mid-3):mid+3]
            random.shuffle(candidates)
            available.append((candidates[0][0], tcga_type))

    random.shuffle(available)
    cptac_results = []
    for svs_path, tcga_type in available[:3]:
        print(f"\n  CPTAC ({tcga_type}): {os.path.basename(svs_path)[:30]}", flush=True)
        result = process_cptac_full(svs_path, cps_model, cls_model, scaler, le, device, known_type=tcga_type)
        if result:
            cptac_results.append(result)

    # ─── Visualization ───
    print("\n" + "="*60)
    print("STEP 4: Generating figures")
    print("="*60, flush=True)

    all_results = tcga_results + cptac_results
    for r in all_results:
        try:
            plot_full_wsi(r, OUT_DIR)
        except Exception as e:
            print(f"    Error: {e}")

    plot_combined_panel(all_results, OUT_DIR)

    # Cleanup CPTAC work dirs
    for r in cptac_results:
        if '_work' in r:
            shutil.rmtree(r['_work'], ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_results)} samples, {OUT_DIR}/")
    print(f"TCGA: {len(tcga_results)}, CPTAC: {len(cptac_results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
