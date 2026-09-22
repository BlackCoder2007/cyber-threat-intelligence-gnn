import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(n_hosts=120, seed=42, output_path='data/network_events.csv'):
    """Generate a deterministic synthetic network-event dataset for an educational demo."""
    rng = np.random.default_rng(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # A hidden set of hosts is used only to generate realistic-looking labels/features.
    suspicious_hosts = set(rng.choice(n_hosts, size=max(10, n_hosts // 8), replace=False).tolist())
    rows = []
    port_pool = np.array([21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080])

    for source in range(n_hosts):
        n_events = int(rng.integers(10, 30))
        for _ in range(n_events):
            target = int(rng.integers(0, n_hosts))
            if target == source:
                target = (target + 1) % n_hosts

            suspicious = source in suspicious_hosts
            # Suspicious nodes have noisier/failure-heavy traffic and more risky ports.
            failed = int(rng.poisson(2.5 if suspicious else 0.5))
            bytes_sent = int(rng.lognormal(8.2 if suspicious else 6.8, 0.75))
            if suspicious:
                destination_port = int(rng.choice([23, 445, 3389, 8080, 21]))
            else:
                destination_port = int(rng.choice(port_pool, p=np.array([
                    .03, .08, .03, .03, .08, .16, .03, .03, .33, .04, .08, .08
                ])))
            external = int(target >= int(n_hosts * 0.85))
            protocol = str(rng.choice(['TCP', 'UDP']))
            duration = float(max(0.1, rng.lognormal(2.0 if suspicious else 1.4, 0.5)))
            packets = int(max(1, rng.lognormal(4.0 if suspicious else 3.5, 0.45)))

            rows.append({
                'source': source,
                'target': target,
                'bytes_sent': bytes_sent,
                'failed': failed,
                'destination_port': destination_port,
                'external': external,
                'duration': duration,
                'packets': packets,
                'protocol': protocol,
            })

    df = pd.DataFrame(rows)
    # Keep the generator reproducible and save the hidden ground-truth label separately.
    labels = pd.DataFrame({'node': sorted(suspicious_hosts), 'is_threat': 1})
    all_labels = pd.DataFrame({'node': np.arange(n_hosts), 'is_threat': 0})
    all_labels.loc[all_labels['node'].isin(suspicious_hosts), 'is_threat'] = 1
    df.to_csv(output_path, index=False)
    all_labels.to_csv(Path(output_path).with_name('node_labels.csv'), index=False)
    return df, all_labels


if __name__ == '__main__':
    df, labels = generate_dataset()
    print(f'Generated {len(df)} network events for {len(labels)} hosts.')
    print(f'Threat hosts: {int(labels.is_threat.sum())}')
