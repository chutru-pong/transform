# pyrefly: ignore [missing-import]
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU THỰC TẾ
# ==========================================
def download_financial_data(ticker, start_date, end_date):
    print(f"Đang tải dữ liệu thực tế cho {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)
    df = df.dropna()
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    features = pd.DataFrame(index=df.index)
    features['Close'] = df[close_col]
    features['Return'] = df[close_col].pct_change()
    features['Volatility'] = df['High'] / df['Low'] - 1
    features['Volume_Log'] = np.log1p(df['Volume'])
    return features.dropna()

def create_sliding_windows(data, seq_len):
    data_array = data.values
    X, Y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i : i + seq_len])
        Y.append(data_array[i + seq_len, 1]) # Dự báo Lợi nhuận (index 1)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# ==========================================
# 2. KIẾN TRÚC CÁC MÔ HÌNH SOTA & ĐỀ XUẤT
# ==========================================

# --- A. Mô hình Đề xuất (Proposed Model) ---
class ProposedModel(nn.Module):
    def __init__(self, num_vars=4, d_model=32):
        super().__init__()
        self.embedding = nn.Linear(num_vars, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.predictor = nn.Linear(d_model, 1)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.attn(emb, emb, emb)
        return self.predictor(out[:, -1, :]).squeeze(-1)

# --- B. DLinear (Zeng et al., 2023) ---
class DLinear(nn.Module):
    def __init__(self, seq_len=60):
        super().__init__()
        self.linear_trend = nn.Linear(seq_len, 1)
        self.linear_seasonal = nn.Linear(seq_len, 1)
        self.avg = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)

    def forward(self, x):
        # Chọn biến Return (index 1) để dự báo
        x_ret = x[:, :, 1] 
        trend = self.avg(x_ret.unsqueeze(1)).squeeze(1)
        seasonal = x_ret - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

# --- C. iTransformer (Liu et al., 2024) ---
# Đảo ngược: Xem mỗi biến là 1 token, thời gian là đặc trưng
class iTransformer(nn.Module):
    def __init__(self, seq_len=60, num_vars=4, d_model=32):
        super().__init__()
        self.project = nn.Linear(seq_len, d_model) # Chiếu chiều thời gian
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.predict = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # [Batch, Vars, Seq_len]
        x = self.project(x)    # [Batch, Vars, d_model]
        x = self.encoder(x)
        x = self.predict(x)    # [Batch, Vars, 1]
        return x[:, 1, 0]      # Lấy dự báo của biến Return (index 1)

# --- D. PatchTST (Nie et al., 2023) ---
# Chia chuỗi thành các phân đoạn (patches)
class PatchTST(nn.Module):
    def __init__(self, seq_len=60, patch_len=12, num_vars=4, d_model=32):
        super().__init__()
        self.num_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.predict = nn.Linear(self.num_patches * d_model, 1)
        self.patch_len = patch_len
        
    def forward(self, x):
        B, L, C = x.shape
        x_ret = x[:, :, 1] # [Batch, Seq_len]
        # Chia thành patches: [Batch, num_patches, patch_len]
        patches = x_ret.view(B, self.num_patches, self.patch_len) 
        emb = self.patch_proj(patches) # [B, num_patches, d_model]
        out = self.encoder(emb)
        out = out.view(B, -1) # Flatten
        return self.predict(out).squeeze(-1)

# ==========================================
# 3. BACKTESTING & TÀI CHÍNH
# ==========================================
def financial_backtest(returns_true, signals, c=0.001, s=0.0005):
    """
    Backtesting tích hợp phí giao dịch (c) và trượt giá (s) (Mục 5.6)
    """
    r_net_list, cum_returns = [], [1.0]
    w_prev = 0.0
    
    for w_t, r_t in zip(signals, returns_true):
        turnover_l1 = abs(w_t - w_prev)
        turnover_l2 = (w_t - w_prev)**2
        # Tính Lợi nhuận sau phí và slippage
        r_net = (w_t * r_t) - c * turnover_l1 - s * turnover_l2
        r_net_list.append(r_net)
        cum_returns.append(cum_returns[-1] * (1 + r_net))
        w_prev = w_t
        
    r_net_array = np.array(r_net_list)
    ann_return = np.mean(r_net_array) * 252
    volatility = np.std(r_net_array) * np.sqrt(252)
    sharpe = ann_return / (volatility + 1e-8)
    
    cum_array = np.array(cum_returns[1:])
    running_max = np.maximum.accumulate(cum_array)
    max_dd = np.max((running_max - cum_array) / running_max)
    
    return {
        "Ann. Return (%)": ann_return * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Cum_Returns": cum_array
    }

# ==========================================
# 4. CHẠY THỰC NGHIỆM VÀ VẼ ĐỒ THỊ
# ==========================================
def run_experiment_on_asset(ticker, start_date, end_date):
    print(f"\n========== THỰC NGHIỆM TRÊN TÀI SẢN: {ticker} ==========")
    df = download_financial_data(ticker, start_date, end_date)
    X, Y = create_sliding_windows(df, seq_len=60)
    
    # Chia Walk-forward: 75% Train, 25% Test
    split_idx = int(len(X) * 0.75)
    train_loader = DataLoader(TensorDataset(X[:split_idx], Y[:split_idx]), batch_size=32, shuffle=True)
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    models = {
        "DLinear": DLinear(),
        "PatchTST": PatchTST(),
        "iTransformer": iTransformer(),
        "ProposedModel": ProposedModel()
    }
    
    criterion = nn.MSELoss()
    results_metrics = {}
    plot_data = {}
    
    for name, model in models.items():
        print(f"-> Đang huấn luyện {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        # Huấn luyện 5 epochs
        for epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                
        # Kiểm thử ngoài phân phối
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            # Chuyển dự báo thành Vị thế (Weights): w_t = tanh(pred / kappa)
            signals = torch.tanh(preds / 0.05).numpy()
            
            # Backtesting tài chính
            metrics = financial_backtest(Y_test.numpy(), signals)
            plot_data[name] = metrics.pop("Cum_Returns") # Tách dữ liệu vẽ đồ thị
            results_metrics[name] = metrics
            
    # In Bảng Kết Quả
    df_results = pd.DataFrame(results_metrics).T
    print(f"\n--- KẾT QUẢ BACKTESTING ({ticker}) ---")
    print(df_results.round(3))
    
    # Vẽ Đồ thị Tích lũy Lợi nhuận
    plt.figure(figsize=(10, 5))
    for name, cum_ret in plot_data.items():
        plt.plot(cum_ret, label=name, linewidth=1.5)
        
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.title(f"Cumulative Returns Comparison - {ticker} (Out-of-Sample)")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value (Base = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# THỰC THI (Chạy cho cả Cổ phiếu SP500 và Tiền mã hóa)
if __name__ == "__main__":
    # Đánh giá trên dữ liệu chuẩn SP500 (SPY)
    run_experiment_on_asset("SPY", "2019-01-01", "2024-01-01")
    
    # Đánh giá trên dữ liệu chuẩn Crypto (BTC-USD)
    run_experiment_on_asset("BTC-USD", "2019-01-01", "2024-01-01")