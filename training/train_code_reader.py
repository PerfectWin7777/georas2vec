import os as _o, sys as _s
_R = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..") if "eval" in __file__ else _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..")
_s.path[:0] = [_o.path.join(_R, "georas2vec"), _o.path.join(_R, "training", "eval"), _o.path.join(_R, "training")]
import os, random, time, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import code_reader as CR
torch.set_num_threads(3)
V = pickle.load(open("code_vocab.pkl", "rb")); G, O = V["gmna"], V["other"]
class Gen(IterableDataset):
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random((wi.id if wi else 0) * 7919 + int(time.time() * 1000) % 100000)
        while True:
            k = rng.random(); c = rng.choice(G) if k < 0.6 else (rng.choice(O) if k < 0.75 else CR.random_code(rng))
            x, w = CR.to_tensor(CR.render(c, rng)); yield x, torch.tensor([CR.IDX[ch] for ch in c]), w
def collate(b):
    xs, ys, ws = zip(*b)
    return torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys])
rng = random.Random(123); VAL = [(c, CR.render(c, rng)) for c in [rng.choice(G) for _ in range(1500)]]
Xv = torch.stack([CR.to_tensor(im)[0] for _, im in VAL]); Yv = [c for c, _ in VAL]
m = CR.CRNN(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
CK = "code_reader_ck.pt"; step = 0
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False); m.load_state_dict(ck["m"]); opt.load_state_dict(ck["o"]); step = ck["step"]
STEPS = 14000; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, last_epoch=step - 1 if step else -1)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
dl = DataLoader(Gen(), batch_size=64, num_workers=3, collate_fn=collate)
t = time.time(); best = 0
for x, y, yl in dl:
    if step >= STEPS:
        break
    m.train(); lp = m(x).log_softmax(-1).permute(1, 0, 2)
    loss = ctc(lp, y, torch.full((x.shape[0],), lp.shape[0], dtype=torch.long), yl)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); sched.step(); step += 1
    if step % 500 == 0:
        m.eval()
        with torch.no_grad():
            pr = sum([CR.decode(m(Xv[i:i + 250])) for i in range(0, len(Xv), 250)], [])
        acc = np.mean([p == c for p, c in zip(pr, Yv)])
        print(f"pas {step} perte {loss.item():.3f} exact (synthétique) {acc:.1%} {time.time() - t:.0f}s", flush=True)
        torch.save(dict(m=m.state_dict(), o=opt.state_dict(), step=step), CK)
        if acc >= best:
            best = acc; torch.save(m.state_dict(), "code_reader.pt")
print("fini", flush=True)
