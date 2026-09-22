from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt

from gnn_model import SimpleGCN

st.set_page_config(page_title='CTI GNN Threat Detector', page_icon='🛡️', layout='wide')

ARTIFACTS = Path('artifacts')
DATA_DIR = Path('data')

st.title('🛡️ Cyber Threat Intelligence using Graph Neural Networks')
st.caption('Educational GCN prototype for node-level threat detection')

if not (ARTIFACTS / 'model.pt').exists():
    st.error('Model artifacts are missing. Run `python train.py` first.')
    st.stop()

metrics = json.loads((ARTIFACTS / 'metrics.json').read_text(encoding='utf-8'))
preds = pd.read_csv(ARTIFACTS / 'node_predictions.csv')
events = pd.read_csv(DATA_DIR / 'network_events.csv')

top_left, top_mid, top_right = st.columns(3)
top_left.metric('Hosts / nodes', metrics['num_nodes'])
top_mid.metric('Network events', metrics['num_events'])
top_right.metric('Test F1', f"{metrics['f1']:.3f}")

st.warning('The default data is synthetic and is intended for demonstrating the pipeline. Do not treat these metrics as real-world cybersecurity performance.')

st.subheader('1. Threat overview')
col1, col2 = st.columns(2)
with col1:
    chart_df = preds['risk_level'].value_counts().reindex(['Low', 'Medium', 'High', 'Critical']).fillna(0)
    st.bar_chart(chart_df)
with col2:
    st.dataframe(
        preds.sort_values('threat_probability', ascending=False).head(15)[
            ['node', 'threat_probability', 'risk_level', 'predicted_label', 'actual_label']
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader('2. Inspect a host')
node_id = st.number_input('Host / node ID', min_value=int(preds.node.min()), max_value=int(preds.node.max()), value=0, step=1)
node = preds[preds.node == node_id].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric('Threat probability', f"{node.threat_probability:.1%}")
c2.metric('Risk level', str(node.risk_level))
c3.metric('Failed connections', int(node.failed_connections))
c4.metric('Risky-port count', int(node.risky_port_count))

st.write('### Network features')
feature_cols = ['out_connections','in_connections','avg_bytes','failed_connections','risky_port_count','external_ratio','avg_duration','avg_packets','degree']
st.dataframe(pd.DataFrame({'Feature': feature_cols, 'Value': [node[c] for c in feature_cols]}), use_container_width=True, hide_index=True)

st.subheader('3. Host communication graph')
G = nx.Graph()
G.add_nodes_from(range(metrics['num_nodes']))
for row in events.itertuples(index=False):
    G.add_edge(int(row.source), int(row.target))

neighbors = list(G.neighbors(int(node_id)))[:25]
sub = G.subgraph([int(node_id)] + neighbors).copy()
pos = nx.spring_layout(sub, seed=42)
fig, ax = plt.subplots(figsize=(10, 6))
nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.35)
node_colors = ['red' if n == int(node_id) else 'skyblue' for n in sub.nodes()]
nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=600, node_color=node_colors)
nx.draw_networkx_labels(sub, pos, ax=ax, font_size=8)
ax.set_axis_off()
st.pyplot(fig)
plt.close(fig)

st.subheader('4. Model details')
with st.expander('Evaluation details'):
    st.json(metrics)
    st.code((ARTIFACTS / 'classification_report.txt').read_text(encoding='utf-8'))
