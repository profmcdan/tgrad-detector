import os
import torch
import numpy as np
import pandas as pd

class TemporalGraphDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        raise NotImplementedError("Subclass must implement load_data")

    def augment_batch_smote(self, batch, smote_ratio=1.0):
        """
        GraphSMOTE-inspired data augmentation.
        Interpolates features of minority class (fraud/cycle) to generate synthetic edges.
        """
        is_cycle = batch['is_cycle']
        fraud_indices = torch.nonzero(is_cycle).squeeze(-1)
        
        # If no fraud in batch or smote ratio is 0, return original
        num_fraud = fraud_indices.numel()
        if num_fraud < 2 or smote_ratio <= 0:
            return batch
            
        num_synthetic = int(num_fraud * smote_ratio)
        
        # Randomly sample pairs of fraud nodes
        idx1 = fraud_indices[torch.randint(0, num_fraud, (num_synthetic,))]
        idx2 = fraud_indices[torch.randint(0, num_fraud, (num_synthetic,))]
        
        # Interpolation weights
        alpha = torch.rand(num_synthetic, 1, device=batch['msg'].device)
        
        # Generate synthetic edges
        synth_msg = alpha * batch['msg'][idx1] + (1 - alpha) * batch['msg'][idx2]
        synth_t = alpha.squeeze(-1) * batch['t'][idx1] + (1 - alpha.squeeze(-1)) * batch['t'][idx2]
        
        # Assign synthetic edges between the pair's src and dst randomly or mix
        synth_src = batch['src'][idx1]
        synth_dst = batch['dst'][idx2]
        synth_is_cycle = torch.ones(num_synthetic, dtype=torch.bool, device=batch['is_cycle'].device)
        
        # Concatenate
        return {
            'src': torch.cat([batch['src'], synth_src]),
            'dst': torch.cat([batch['dst'], synth_dst]),
            't': torch.cat([batch['t'], synth_t]),
            'msg': torch.cat([batch['msg'], synth_msg]),
            'is_cycle': torch.cat([batch['is_cycle'], synth_is_cycle])
        }

class SyntheticGraphStream(TemporalGraphDataset):
    """Fallback synthetic engine for pure topology tests."""
    def __init__(self, num_nodes=1000, num_edges=10000, edge_dim=64, cycle_ratio=0.05):
        super().__init__("synthetic")
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_dim = edge_dim
        self.cycle_ratio = cycle_ratio
        
        self.src = None
        self.dst = None
        self.t = None
        self.msg = None
        self.is_cycle = None

    def load_data(self):
        print(f"Generating synthetic graph stream: {self.num_edges} edges, {self.num_nodes} nodes.")
        self.t = torch.sort(torch.rand(self.num_edges) * 1000.0)[0]
        self.msg = torch.randn(self.num_edges, self.edge_dim)
        
        self.src = torch.randint(0, self.num_nodes, (self.num_edges,))
        self.dst = torch.randint(0, self.num_nodes, (self.num_edges,))
        self.is_cycle = torch.zeros(self.num_edges, dtype=torch.bool)
        
        num_cycles = int(self.num_edges * self.cycle_ratio / 3)
        for i in range(num_cycles):
            idx = torch.randint(0, self.num_edges - 3, (1,)).item()
            nodes = torch.randperm(self.num_nodes)[:3]
            self.src[idx] = nodes[0]
            self.dst[idx] = nodes[1]
            self.src[idx+1] = nodes[1]
            self.dst[idx+1] = nodes[2]
            self.src[idx+2] = nodes[2]
            self.dst[idx+2] = nodes[0]
            self.is_cycle[idx:idx+3] = True
            
        print(f"Injected {num_cycles} cycles.")

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, self.num_edges)
        return {
            'src': self.src[start_idx:end_idx],
            'dst': self.dst[start_idx:end_idx],
            't': self.t[start_idx:end_idx],
            'msg': self.msg[start_idx:end_idx],
            'is_cycle': self.is_cycle[start_idx:end_idx]
        }


class EllipticStandardDataset(TemporalGraphDataset):
    """
    Loads the real Elliptic Bitcoin dataset using the canonical Weber et al. (2019)
    evaluation protocol used throughout the fraud-detection literature:

      * Each transaction is a node with 165 features and a discrete time step (1-49).
      * Edges are directed Bitcoin flows between transactions of the same time step.
      * Labels: illicit = 1, licit = 0, unknown = -1 (excluded from train/test).
      * Temporal split: time steps 1-34 -> train, 35-49 -> test (no leakage).

    The class exposes BOTH a node-level view (for baseline classifiers and node-level
    evaluation) and a chronologically ordered edge stream (for the continuous-time
    temporal encoder and the diffusion engine).
    """

    def __init__(self, data_path: str = "data/elliptic_bitcoin_dataset", train_t_max: int = 34):
        super().__init__(data_path)
        self.edge_dim = 165
        self.train_t_max = train_t_max

    def load_data(self):
        classes_path = os.path.join(self.data_path, "elliptic_txs_classes.csv")
        edges_path = os.path.join(self.data_path, "elliptic_txs_edgelist.csv")
        features_path = os.path.join(self.data_path, "elliptic_txs_features.csv")

        if not (os.path.exists(classes_path) and os.path.exists(edges_path) and os.path.exists(features_path)):
            raise FileNotFoundError(
                f"Real Elliptic dataset not found at {self.data_path}. "
                "This standard-protocol loader requires the genuine CSV files."
            )

        # ---- 1. Node features and time steps (col0 = txId, col1 = time step) ----
        feats = pd.read_csv(features_path, header=None)
        node_ids = feats[0].astype(np.int64).values
        node_map = {tx: i for i, tx in enumerate(node_ids)}
        self.num_nodes = len(node_ids)

        self.node_time = feats[1].astype(np.int64).values                 # (N,)
        self.node_feat = feats.iloc[:, 2:].astype(np.float32).values      # (N, 165)

        # ---- 2. Labels: illicit=1, licit=0, unknown=-1 ----
        df_classes = pd.read_csv(classes_path)
        cls_map = {'1': 1, '2': 0, 'unknown': -1, 1: 1, 2: 0}
        df_classes['y'] = df_classes['class'].map(cls_map).fillna(-1).astype(int)
        label_lookup = dict(zip(df_classes['txId'].astype(np.int64), df_classes['y']))
        self.node_label = np.array([label_lookup.get(tx, -1) for tx in node_ids], dtype=np.int64)

        # ---- 3. Edges -> contiguous indices ----
        df_edges = pd.read_csv(edges_path)
        df_edges['src'] = df_edges['txId1'].map(node_map)
        df_edges['dst'] = df_edges['txId2'].map(node_map)
        df_edges = df_edges.dropna(subset=['src', 'dst']).astype({'src': int, 'dst': int})
        src_arr = df_edges['src'].values
        dst_arr = df_edges['dst'].values
        self.edge_index = torch.tensor(np.vstack([src_arr, dst_arr]), dtype=torch.long)
        self.num_edges = src_arr.shape[0]

        # ---- 4. Chronological edge stream (edge inherits its source-node attributes) ----
        edge_t = self.node_time[src_arr].astype(np.float32)
        edge_msg = self.node_feat[src_arr]
        edge_label = self.node_label[src_arr]

        order = np.argsort(edge_t, kind='stable')
        self.src = torch.tensor(src_arr[order], dtype=torch.long)
        self.dst = torch.tensor(dst_arr[order], dtype=torch.long)
        self.t = torch.tensor(edge_t[order], dtype=torch.float32)
        self.msg = torch.tensor(edge_msg[order], dtype=torch.float32)
        self.edge_label = torch.tensor(edge_label[order], dtype=torch.long)
        # is_cycle kept for backward compatibility: True where the source node is illicit
        self.is_cycle = (self.edge_label == 1)

        # ---- 5. Node-level tensors and temporal masks ----
        self.node_feat_t = torch.tensor(self.node_feat, dtype=torch.float32)
        self.node_label_t = torch.tensor(self.node_label, dtype=torch.long)
        self.node_time_t = torch.tensor(self.node_time, dtype=torch.long)

        labeled = self.node_label_t >= 0
        self.train_mask = labeled & (self.node_time_t <= self.train_t_max)
        self.test_mask = labeled & (self.node_time_t > self.train_t_max)

        n_ill = int((self.node_label_t == 1).sum())
        n_lic = int((self.node_label_t == 0).sum())
        print(f"Elliptic loaded: {self.num_nodes} nodes, {self.num_edges} edges.")
        print(f"  Labeled: {n_ill} illicit, {n_lic} licit, "
              f"{self.num_nodes - n_ill - n_lic} unknown.")
        print(f"  Temporal split (t<= {self.train_t_max}): "
              f"train nodes={int(self.train_mask.sum())}, test nodes={int(self.test_mask.sum())}.")

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, self.num_edges)
        return {
            'src': self.src[start_idx:end_idx],
            'dst': self.dst[start_idx:end_idx],
            't': self.t[start_idx:end_idx],
            'msg': self.msg[start_idx:end_idx],
            'is_cycle': self.is_cycle[start_idx:end_idx],
            'edge_label': self.edge_label[start_idx:end_idx],
        }


class SyntheticCyclicAML(TemporalGraphDataset):
    """
    AMLSim-style synthetic anti-money-laundering benchmark with PLANTED cyclic
    laundering typologies (the proposal names AMLSim as the cycle ground-truth set).

    Crucially, illicit nodes are drawn from the SAME feature distribution as benign
    nodes: fraud is encoded purely in the directed-cycle TOPOLOGY and its bounded
    temporal ordering (Eqs. 17-19), not in node attributes. Attribute-only models
    are therefore near-random, isolating the value of temporal-relational and
    cycle-aware structural reasoning.
    """

    def __init__(self, num_nodes=6000, feat_dim=165, num_benign_edges=18000,
                 num_cycles=600, cycle_len_range=(3, 6), time_span=100.0,
                 cycle_window=1.0, seed=42, train_t_max=None):
        super().__init__("synthetic_cyclic_aml")
        self.num_nodes = num_nodes
        self.edge_dim = feat_dim
        self.num_benign_edges = num_benign_edges
        self.num_cycles = num_cycles
        self.cycle_len_range = cycle_len_range
        self.time_span = time_span
        self.cycle_window = cycle_window
        self.seed = seed
        self.train_t_max = train_t_max if train_t_max is not None else 0.6 * time_span

    def load_data(self):
        rng = np.random.default_rng(self.seed)
        N, F = self.num_nodes, self.edge_dim

        # Node attributes: identical distribution for illicit and benign nodes.
        self.node_feat = rng.standard_normal((N, F)).astype(np.float32)
        node_label = np.zeros(N, dtype=np.int64)

        src, dst, t = [], [], []
        # Benign background traffic (random directed interactions).
        for _ in range(self.num_benign_edges):
            a, b = rng.integers(0, N, 2)
            if a == b:
                continue
            src.append(a); dst.append(b); t.append(rng.random() * self.time_span)

        # Planted directed laundering cycles with bounded temporal ordering.
        lo, hi = self.cycle_len_range
        for _ in range(self.num_cycles):
            L = int(rng.integers(lo, hi + 1))
            nodes = rng.choice(N, L, replace=False)
            t0 = rng.random() * self.time_span
            for i in range(L):
                src.append(int(nodes[i])); dst.append(int(nodes[(i + 1) % L]))
                t.append(t0 + i * (self.cycle_window / L))
                node_label[nodes[i]] = 1

        src = np.array(src); dst = np.array(dst); t = np.array(t, dtype=np.float32)
        order = np.argsort(t, kind="stable")
        self.src = torch.tensor(src[order], dtype=torch.long)
        self.dst = torch.tensor(dst[order], dtype=torch.long)
        self.t = torch.tensor(t[order], dtype=torch.float32)
        self.msg = torch.tensor(self.node_feat[src[order]], dtype=torch.float32)
        self.num_edges = len(src)
        self.edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        edge_label = node_label[src[order]]
        self.edge_label = torch.tensor(edge_label, dtype=torch.long)
        self.is_cycle = (self.edge_label == 1)

        # Node-level time = mean interaction time (cycles stay within one split).
        node_time = np.zeros(N, dtype=np.float32)
        cnt = np.zeros(N, dtype=np.float32)
        np.add.at(node_time, src, t); np.add.at(node_time, dst, t)
        np.add.at(cnt, src, 1.0); np.add.at(cnt, dst, 1.0)
        node_time = node_time / np.clip(cnt, 1, None)

        self.node_feat_t = torch.tensor(self.node_feat, dtype=torch.float32)
        self.node_label_t = torch.tensor(node_label, dtype=torch.long)
        self.node_time_t = torch.tensor(node_time, dtype=torch.float32)
        appears = torch.zeros(N, dtype=torch.bool)
        appears[self.edge_index[0]] = True; appears[self.edge_index[1]] = True
        self.train_mask = appears & (self.node_time_t <= self.train_t_max)
        self.test_mask = appears & (self.node_time_t > self.train_t_max)

        print(f"Synthetic AML: {N} nodes, {self.num_edges} edges, "
              f"{int(node_label.sum())} illicit (cycle) nodes.")
        print(f"  Split: train nodes={int(self.train_mask.sum())}, "
              f"test nodes={int(self.test_mask.sum())}.")

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, self.num_edges)
        return {
            'src': self.src[start_idx:end_idx],
            'dst': self.dst[start_idx:end_idx],
            't': self.t[start_idx:end_idx],
            'msg': self.msg[start_idx:end_idx],
            'is_cycle': self.is_cycle[start_idx:end_idx],
            'edge_label': self.edge_label[start_idx:end_idx],
        }


class EllipticDataset(TemporalGraphDataset):
    """
    Parses Elliptic Bitcoin Data. Transactions act as Nodes, Bitcoin flows as Edges.
    We convert this directly into continuous streaming edge bounds.
    """
    def __init__(self, data_path: str = "data/elliptic_bitcoin_dataset"):
        super().__init__(data_path)
        self.edge_dim = 165 # 166 columns - 1 for time_step
        
    def load_data(self):
        classes_path = os.path.join(self.data_path, "elliptic_txs_classes.csv")
        edges_path = os.path.join(self.data_path, "elliptic_txs_edgelist.csv")
        features_path = os.path.join(self.data_path, "elliptic_txs_features.csv")
        
        if not (os.path.exists(classes_path) and os.path.exists(edges_path) and os.path.exists(features_path)):
            print(f"WARNING: Full Elliptic Dataset missing at {self.data_path}!")
            print("To ensure execution, gracefully failing over to Synthetic Clone matching Elliptic dimensionality...")
            
            # Simulated Elliptic Bounds
            self.num_nodes = 203769 
            self.num_edges = 234355
            
            self.t = torch.sort(torch.rand(self.num_edges) * 49.0)[0] # 49 timesteps
            self.msg = torch.randn(self.num_edges, self.edge_dim)
            self.src = torch.randint(0, self.num_nodes, (self.num_edges,))
            self.dst = torch.randint(0, self.num_nodes, (self.num_edges,))
            self.is_cycle = torch.zeros(self.num_edges, dtype=torch.bool)
            
            # Synthesize ~5,000 illicit behaviors
            num_illicit = int(self.num_edges * 0.05)
            illicit_indices = torch.randperm(self.num_edges)[:num_illicit]
            self.is_cycle[illicit_indices] = True
            
            print(f"Loaded Simulated Elliptic Engine: {self.num_nodes} tx nodes, {self.num_edges} flows.")
            return

        print(f"Loading Pure Elliptic Data via Pandas...")
        
        # 1. Parse Classes (Nodes -> Fraud Target)
        df_classes = pd.read_csv(classes_path)
        df_classes['class'] = df_classes['class'].map({'1': 1, '2': 0, 'unknown': 0}).fillna(0)
        
        # 2. Map Node IDs to flat [0, N-1] arrays for contiguous memory
        unique_nodes = df_classes['txId'].unique()
        self.num_nodes = len(unique_nodes)
        node_map = {tx: idx for idx, tx in enumerate(unique_nodes)}
        
        # 3. Load Edges
        df_edges = pd.read_csv(edges_path)
        df_edges['src'] = df_edges['txId1'].map(node_map)
        df_edges['dst'] = df_edges['txId2'].map(node_map)
        df_edges = df_edges.dropna(subset=['src', 'dst']).astype(int)
        self.num_edges = len(df_edges)
        
        # 4. Load Features & Timesteps (Nodes hold the features initially)
        # First column is txId, Second is timestep (1-49)
        df_features = pd.read_csv(features_path, header=None)
        df_features[0] = df_features[0].map(node_map)
        df_features = df_features.dropna(subset=[0]).sort_values(by=0).set_index(0)
        
        # 5. Connect Edges logically. 
        # For a continuous streaming graph, the "Edge Tensor" passes the features of the Originating Node.
        # This simulates Bitcoin leaving Wallet Tx1 -> Wallet Tx2. 
        src_array = df_edges['src'].values
        mapped_features = df_features.loc[src_array].values
        
        # Extract and sort by timestamps sequentially
        time_steps = mapped_features[:, 0]
        edge_matrix = mapped_features[:, 1:]
        illicit_mapping = df_classes.set_index('txId')['class'].values
        target_labels = illicit_mapping[src_array]
        
        # Memory to Tensors
        self.src = torch.tensor(src_array, dtype=torch.long)
        self.dst = torch.tensor(df_edges['dst'].values, dtype=torch.long)
        self.t = torch.tensor(time_steps, dtype=torch.float32)
        self.msg = torch.tensor(edge_matrix, dtype=torch.float32)
        self.is_cycle = torch.tensor(target_labels, dtype=torch.bool)
        
        # Finally, Sort continuously
        sorted_indices = torch.argsort(self.t)
        self.src = self.src[sorted_indices]
        self.dst = self.dst[sorted_indices]
        self.t = self.t[sorted_indices]
        self.msg = self.msg[sorted_indices]
        self.is_cycle = self.is_cycle[sorted_indices]
        
        print(f"Extraction Successful: {self.num_nodes} nodes, {self.num_edges} streaming bounds.")
        print(f"Fraud distribution detected: {self.is_cycle.sum().item()} instances.")
        
    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, self.num_edges)
        return {
            'src': self.src[start_idx:end_idx],
            'dst': self.dst[start_idx:end_idx],
            't': self.t[start_idx:end_idx],
            'msg': self.msg[start_idx:end_idx],
            'is_cycle': self.is_cycle[start_idx:end_idx]
        }
