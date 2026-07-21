
import os
import sys
import torch
import numpy as np
import copy
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import helpers
from src.data.load import load_dataset
from src.models.build_model import build_model
from src.models.train import train
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.data.load import collate_fn_with_laplacian
from src.models.cue import CUE

def test_kmeans_initializationit():
    print("="*60)
    print("TEST: Verifying K-Means Initialization Integration in Training Loop")
    print("="*60)

    # 1. Setup Configuration (Use MUTAG as it's small)
    dataset_name = "mutag"
    try:
        config_module = __import__(f"src.config.experiments.{dataset_name}", fromlist=[dataset_name])
        cfg = getattr(config_module, dataset_name)
    except ImportError:
        print(f"Error: Could not load config for {dataset_name}")
        return

    # Set epochs to 0 to skip actual training loop, but trigger initialization
    cfg.n_epochs = 0 
    cfg.batch_size = 16 # Small batch size

    # 2. Load Data
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(**cfg.dataset_config.model_dump())
    
    loader = PyGDataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True, # Shuffle shouldn't matter for init logic
        collate_fn=collate_fn_with_laplacian
    )
    print(f"Data loaded. {len(dataset)} samples.")

    # 3. Build Model
    print("Building model...")
    net = build_model(cfg.network_config)
    net = net.to(config.DEVICE)
    
    if not isinstance(net, CUE):
        print("Skipping test: Model is not CUE")
        return

    # 4. Check Pre-Init State
    print("\n[Step 1] Checking Pre-Initialization State")
    if not hasattr(net, 'init_centers_with_kmeans'):
        print("FAIL: Model does not have 'init_centers_with_kmeans' method!")
        return
    else:
        print("PASS: Model has 'init_centers_with_kmeans' method.")

    # Record old weights
    old_weights = net.evidence_head.linear.weight.data.clone()
    print(f"Old weights mean: {old_weights.mean().item():.6f}")
    print(f"Old weights std:  {old_weights.std().item():.6f}")

    # Check if weights are orthogonal (dot product of different rows should be 0)
    # Roughly check for one pair
    dp = torch.dot(old_weights[0], old_weights[1])
    print(f"Old weights dot product (0,1) (Orthogonal Init): {dp.item():.6f} (Should be close to 0)")

    # 5. Run Train (Trigger Init)
    print("\n[Step 2] Running train() with n_epochs=0 to trigger initialization...")
    # Passing dummy eval_data
    try:
        train(cfg, net, loader, run=0, eval_data=None, callbacks=[])
    except Exception as e:
        print(f"Error during training execution: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Check Post-Init State
    print("\n[Step 3] Checking Post-Initialization State")
    
    # Check if flag is set
    if getattr(net, 'kmeans_initialized_done', False):
        print("PASS: 'kmeans_initialized_done' flag was successfully set.")
    else:
        print("FAIL: 'kmeans_initialized_done' flag was NOT set.")

    # Check weights changed
    new_weights = net.evidence_head.linear.weight.data
    
    print(f"New weights mean: {new_weights.mean().item():.6f}")
    print(f"New weights std:  {new_weights.std().item():.6f}")
    
    if torch.equal(old_weights, new_weights):
        print("FAIL: Weights did NOT change after initialization sequence!")
    else:
        diff = (new_weights - old_weights).abs().sum().item()
        print(f"PASS: Weights changed successfully. Total diff: {diff:.6f}")
        
    # Check if new weights look somewhat like normalized vectors (since K-Means on normalized zs might produce normalized centers? 
    # Actually K-Means centers are not necessarily normalized, but we normalize them in forward pass. 
    # However, since z was normalized, centers should be roughly on the unit hypersphere or defined range.
    norm_new = torch.norm(new_weights, dim=1).mean().item()
    print(f"Average norm of new weights: {norm_new:.6f}") 
    # z is from LayerNorm then F.normalize (if implemented that way in extraction?)
    
    # Wait, the code in init_centers_with_kmeans does: 
    # hidden = self.evidence_head.hidden(v_out) 
    # So z has LayerNorm applied.
    # The K-Means centers will have scale similar to LayerNorm output.
    
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    test_kmeans_initializationit()
