
import pickle, numpy as np, pandas as pd, torch, torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cardinalities, d_model):
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.empty(n_num, d_model))
        self.num_bias   = nn.Parameter(torch.empty(n_num, d_model))
        self.cardinalities = cardinalities
        if len(cardinalities) > 0:
            offsets = np.cumsum([0] + cardinalities[:-1])
            self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
            self.cat_emb = nn.Embedding(sum(cardinalities), d_model)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
    def forward(self, x_num, x_cat):
        B = x_num.size(0)
        tokens = [self.cls.expand(B, -1, -1)]
        if self.n_num > 0:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if len(self.cardinalities) > 0:
            tokens.append(self.cat_emb(x_cat + self.cat_offsets))
        return torch.cat(tokens, dim=1)

class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities, d_model, n_layers, n_heads, ffn_mult, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cardinalities, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model * ffn_mult * 2), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x_num, x_cat):
        out = self.encoder(self.tokenizer(x_num, x_cat))
        return self.head(out[:, 0]).squeeze(-1)

def _norm_cat(s):
    return (s.astype(str).str.strip().str.lower()
             .replace({"nan": "__na__", "none": "__na__"}))

def predict_dl(df, model_path="ft_transformer_v2.pt", prep_path="dl_preprocess.pkl",
               batch_size=4096, device=None):
    # 로우데이터 df(분할 parquet 형식) -> 0~1 부정확률 배열
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = FTTransformer(len(prep["num_cols"]), prep["cardinalities"], cfg["d_model"],
                          cfg["n_layers"], cfg["n_heads"], cfg["ffn_mult"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state_dict"]); model.to(device).eval()

    x = df[prep["num_cols"]].astype(np.float64)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(prep["num_medians"]).values
    X_num = prep["qt"].transform(x).astype(np.float32)
    X_cat = np.zeros((len(df), len(prep["cat_cols"])), dtype=np.int64)
    for j, c in enumerate(prep["cat_cols"]):
        X_cat[:, j] = _norm_cat(df[c]).map(prep["cat_maps"][c]).fillna(0).astype(np.int64).values

    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            xn = torch.from_numpy(X_num[i:i+batch_size]).to(device)
            xc = torch.from_numpy(X_cat[i:i+batch_size]).to(device)
            probs.append(torch.sigmoid(model(xn, xc).float()).cpu().numpy())
    return np.concatenate(probs)
