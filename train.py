import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from data_generator import generate_dataset
from gnn_model import SimpleGCN

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path('data')
ARTIFACT_DIR = Path('artifacts')
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


def build_graph(df, labels):
    n_nodes = len(labels)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for row in df.itertuples(index=False):
        G.add_edge(int(row.source), int(row.target))

    # Node features are aggregate CTI/network indicators.
    out_conn = df.groupby('source').size().reindex(range(n_nodes), fill_value=0)
    in_conn = df.groupby('target').size().reindex(range(n_nodes), fill_value=0)
    bytes_mean = df.groupby('source')['bytes_sent'].mean().reindex(range(n_nodes), fill_value=0)
    failed_sum = df.groupby('source')['failed'].sum().reindex(range(n_nodes), fill_value=0)
    risky_ports = df[df['destination_port'].isin([21, 23, 445, 3389, 8080])].groupby('source').size().reindex(range(n_nodes), fill_value=0)
    external_ratio = df.groupby('source')['external'].mean().reindex(range(n_nodes), fill_value=0)
    duration_mean = df.groupby('source')['duration'].mean().reindex(range(n_nodes), fill_value=0)
    packet_mean = df.groupby('source')['packets'].mean().reindex(range(n_nodes), fill_value=0)
    degree = pd.Series(dict(G.degree())).reindex(range(n_nodes), fill_value=0)

    features = pd.DataFrame({
        'out_connections': out_conn,
        'in_connections': in_conn,
        'avg_bytes': bytes_mean,
        'failed_connections': failed_sum,
        'risky_port_count': risky_ports,
        'external_ratio': external_ratio,
        'avg_duration': duration_mean,
        'avg_packets': packet_mean,
        'degree': degree,
    })

    X = features.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Symmetric GCN normalization A_hat = D^-1/2 (A+I) D^-1/2.
    A = nx.to_numpy_array(G, nodelist=range(n_nodes), dtype=np.float32)
    A = A + np.eye(n_nodes, dtype=np.float32)
    degree_vec = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vec))
    A_hat = d_inv_sqrt @ A @ d_inv_sqrt

    y = labels.set_index('node').reindex(range(n_nodes))['is_threat'].values.astype(np.int64)
    return features, X, A_hat.astype(np.float32), y, scaler


def stratified_masks(y, train_frac=0.60, val_frac=0.20, seed=42):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        train.extend(idx[:n_train])
        val.extend(idx[n_train:n_train + n_val])
        test.extend(idx[n_train + n_val:])
    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


def train_model(X, A_hat, y, train_idx, val_idx, epochs=250, lr=0.01, weight_decay=1e-4):
    X_t = torch.tensor(X)
    A_t = torch.tensor(A_hat)
    y_t = torch.tensor(y)

    model = SimpleGCN(in_features=X.shape[1], hidden_features=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Handle class imbalance.
    counts = np.bincount(y[train_idx], minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    best_state = None
    best_val_f1 = -1.0
    patience = 35
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t, A_t)
        loss = F.cross_entropy(logits[train_idx], y_t[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = logits[val_idx].argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y[val_idx], val_pred, zero_division=0)

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epoch, best_val_f1


def save_artifacts(model, features, X, A_hat, y, train_idx, val_idx, test_idx, scaler, raw_features):
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_features': X.shape[1],
        'hidden_features': 32,
        'num_classes': 2,
    }, ARTIFACT_DIR / 'model.pt')

    np.savez_compressed(
        ARTIFACT_DIR / 'graph_data.npz',
        X=X,
        A_hat=A_hat,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        raw_features=raw_features.values.astype(np.float32),
    )

    import joblib
    joblib.dump(scaler, ARTIFACT_DIR / 'scaler.joblib')


def main():
    event_path = DATA_DIR / 'network_events.csv'
    label_path = DATA_DIR / 'node_labels.csv'
    if not event_path.exists() or not label_path.exists():
        df, labels = generate_dataset()
    else:
        df = pd.read_csv(event_path)
        labels = pd.read_csv(label_path)

    raw_features, X, A_hat, y, scaler = build_graph(df, labels)
    train_idx, val_idx, test_idx = stratified_masks(y, seed=SEED)
    model, epochs_used, best_val_f1 = train_model(X, A_hat, y, train_idx, val_idx)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X), torch.tensor(A_hat))
        prob = torch.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(dim=1).numpy()

    y_test, p_test = y[test_idx], pred[test_idx]
    metrics = {
        'accuracy': float(accuracy_score(y_test, p_test)),
        'precision': float(precision_score(y_test, p_test, zero_division=0)),
        'recall': float(recall_score(y_test, p_test, zero_division=0)),
        'f1': float(f1_score(y_test, p_test, zero_division=0)),
        'epochs_used': int(epochs_used),
        'best_validation_f1': float(best_val_f1),
        'num_nodes': int(len(y)),
        'num_events': int(len(df)),
        'train_nodes': int(len(train_idx)),
        'validation_nodes': int(len(val_idx)),
        'test_nodes': int(len(test_idx)),
        'note': 'Synthetic educational dataset; metrics are not real-world benchmark results.'
    }
    (ARTIFACT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    out = raw_features.copy()
    out.insert(0, 'node', np.arange(len(out)))
    out['actual_label'] = y
    out['threat_probability'] = prob
    out['predicted_label'] = pred
    out['split'] = 'unassigned'
    out.loc[train_idx, 'split'] = 'train'
    out.loc[val_idx, 'split'] = 'validation'
    out.loc[test_idx, 'split'] = 'test'
    out['risk_level'] = pd.cut(
        out['threat_probability'],
        bins=[-0.001, 0.30, 0.60, 0.80, 1.01],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    out.to_csv(ARTIFACT_DIR / 'node_predictions.csv', index=False)

    cm = confusion_matrix(y_test, p_test).tolist()
    (ARTIFACT_DIR / 'classification_report.txt').write_text(
        classification_report(y_test, p_test, digits=4, zero_division=0) +
        f'\nConfusion matrix: {cm}\n', encoding='utf-8'
    )

    save_artifacts(model, raw_features, X, A_hat, y, train_idx, val_idx, test_idx, scaler, raw_features)

    print('\n=== Cyber Threat Intelligence using GNN ===')
    print(f'Nodes: {len(y)} | Events: {len(df)}')
    print(f'Test Accuracy : {metrics["accuracy"]:.4f}')
    print(f'Test Precision: {metrics["precision"]:.4f}')
    print(f'Test Recall   : {metrics["recall"]:.4f}')
    print(f'Test F1       : {metrics["f1"]:.4f}')
    print('Artifacts saved to ./artifacts/')


if __name__ == '__main__':
    main()
