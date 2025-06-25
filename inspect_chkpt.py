import torch
import json
from collections import OrderedDict

def inspect_checkpoint(checkpoint_path):
    """
    Comprehensive checkpoint inspection function
    """
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return
    
    # 1. Show top-level keys
    print(f"\n1. TOP-LEVEL KEYS:")
    print("-" * 30)
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # 2. Inspect model state dict
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        # Assume the whole checkpoint is the model
        model_state = checkpoint
    
    print(f"\n2. MODEL ARCHITECTURE:")
    print("-" * 30)
    
    # Group layers by component
    layer_groups = {}
    for key in model_state.keys():
        component = key.split('.')[0]
        if component not in layer_groups:
            layer_groups[component] = []
        layer_groups[component].append(key)
    
    for component, layers in layer_groups.items():
        print(f"\n  {component.upper()}: ({len(layers)} parameters)")
        for layer in sorted(layers)[:5]:  # Show first 5 layers
            tensor_shape = model_state[layer].shape
            print(f"    {layer}: {tensor_shape}")
        if len(layers) > 5:
            print(f"    ... and {len(layers) - 5} more")
    
    # 3. Key architecture details
    print(f"\n3. ARCHITECTURE DETAILS:")
    print("-" * 30)
    
    # Embedding dimension
    if 'pos_embed' in model_state:
        embed_dim = model_state['pos_embed'].shape[-1]
        num_patches = model_state['pos_embed'].shape[1] - 1  # minus cls token
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Number of patches: {num_patches}")
        print(f"  Image size (estimated): {int(num_patches**0.5) * 16}px (assuming patch_size=16)")
    
    # Count transformer blocks
    block_count = 0
    for key in model_state.keys():
        if key.startswith('blocks.') and key.endswith('.norm1.weight'):
            block_count += 1
    if block_count > 0:
        print(f"  Number of transformer blocks: {block_count}")
    
    # Head information
    if 'head.weight' in model_state:
        num_classes = model_state['head.weight'].shape[0]
        print(f"  Number of classes: {num_classes}")
    
    # Positional embedding type
    if 'pos_embed_spatial' in model_state and 'pos_embed_temporal' in model_state:
        spatial_patches = model_state['pos_embed_spatial'].shape[1]
        temporal_patches = model_state['pos_embed_temporal'].shape[1]
        print(f"  Separable positional embeddings:")
        print(f"    Spatial patches: {spatial_patches}")
        print(f"    Temporal patches: {temporal_patches}")
        print(f"  -> This is likely a SpectralGPT/Video ViT model")
    
    # Patch embedding info
    if 'patch_embed.proj.weight' in model_state:
        patch_weight = model_state['patch_embed.proj.weight']
        if len(patch_weight.shape) == 5:  # 3D conv
            out_ch, in_ch, t_patch, h_patch, w_patch = patch_weight.shape
            print(f"  3D Patch embedding: {in_ch} -> {out_ch}")
            print(f"    Patch size: temporal={t_patch}, spatial={h_patch}x{w_patch}")
            print(f"  -> This is a 3D/Temporal model")
        elif len(patch_weight.shape) == 4:  # 2D conv
            out_ch, in_ch, h_patch, w_patch = patch_weight.shape
            print(f"  2D Patch embedding: {in_ch} -> {out_ch}")
            print(f"    Patch size: {h_patch}x{w_patch}")
    
    # 4. Total parameters
    print(f"\n4. PARAMETER COUNT:")
    print("-" * 30)
    total_params = sum(p.numel() for p in model_state.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
    
    # 5. Check for specific components
    print(f"\n5. SPECIAL COMPONENTS:")
    print("-" * 30)
    
    components_found = []
    if any('decoder' in key for key in model_state.keys()):
        components_found.append("Decoder (MAE/Autoencoder)")
    if any('cls_seg' in key for key in model_state.keys()):
        components_found.append("Segmentation head")
    if any('location_embed' in key for key in model_state.keys()):
        components_found.append("Location embeddings")
    if any('temporal_mlp' in key for key in model_state.keys()):
        components_found.append("Temporal MLP")
    if any('mask_token' in key for key in model_state.keys()):
        components_found.append("Mask token (MAE)")
    if any('channel_embed' in key for key in model_state.keys()):
        components_found.append("Channel embeddings")
    
    if components_found:
        for comp in components_found:
            print(f"  ✓ {comp}")
    else:
        print("  Standard ViT components only")
    
    # 6. Save detailed layer info to file
    layer_info = {}
    for key, tensor in model_state.items():
        layer_info[key] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        }
    
    output_file = checkpoint_path.replace('.pth', '_architecture.json')
    with open(output_file, 'w') as f:
        json.dump(layer_info, f, indent=2)
    
    print(f"\n6. DETAILED INFO SAVED:")
    print("-" * 30)
    print(f"  Layer details saved to: {output_file}")
    
    return checkpoint

# Example usage functions
def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """Compare two checkpoints to see architectural differences"""
    print("COMPARING CHECKPOINTS")
    print("=" * 60)
    
    ckpt1 = torch.load(checkpoint1_path, map_location='cpu')
    ckpt2 = torch.load(checkpoint2_path, map_location='cpu')
    
    model1 = ckpt1.get('model', ckpt1)
    model2 = ckpt2.get('model', ckpt2)
    
    keys1 = set(model1.keys())
    keys2 = set(model2.keys())
    
    print(f"Checkpoint 1: {len(keys1)} parameters")
    print(f"Checkpoint 2: {len(keys2)} parameters")
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"\nCommon parameters: {len(common_keys)}")
    print(f"Only in checkpoint 1: {len(only_in_1)}")
    print(f"Only in checkpoint 2: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\nUnique to checkpoint 1:")
        for key in sorted(only_in_1)[:10]:
            print(f"  {key}: {model1[key].shape}")
    
    if only_in_2:
        print(f"\nUnique to checkpoint 2:")
        for key in sorted(only_in_2)[:10]:
            print(f"  {key}: {model2[key].shape}")

def extract_config_from_checkpoint(checkpoint_path):
    """Try to extract model configuration from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if config is stored
    if 'args' in checkpoint:
        print("Found stored arguments:")
        args = checkpoint['args']
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        return args
    
    # Try to infer config from weights
    model_state = checkpoint.get('model', checkpoint)
    config = {}
    
    if 'pos_embed' in model_state:
        embed_dim = model_state['pos_embed'].shape[-1]
        config['embed_dim'] = embed_dim
    
    if 'patch_embed.proj.weight' in model_state:
        patch_weight = model_state['patch_embed.proj.weight']
        if len(patch_weight.shape) == 5:
            _, in_chans, t_patch, patch_size, _ = patch_weight.shape
            config['in_chans'] = in_chans
            config['t_patch_size'] = t_patch
            config['patch_size'] = patch_size
    
    print("Inferred configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

# Quick inspection function
def quick_inspect(checkpoint_path):
    """Quick overview of checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model', checkpoint)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {len(model_state)}")
    print(f"Total size: {sum(p.numel() for p in model_state.values()):,}")
    
    # Key indicators
    if 'pos_embed_spatial' in model_state:
        print("Type: Temporal/3D ViT")
    elif 'decoder_embed' in model_state:
        print("Type: MAE (Masked Autoencoder)")
    elif 'cls_seg' in model_state:
        print("Type: Segmentation model")
    else:
        print("Type: Standard ViT")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/root/projects/data/SpectralGPT+.pth"
    
    # Full inspection
    inspect_checkpoint(checkpoint_path)
    
    # Quick inspection
    # quick_inspect(checkpoint_path)
    
    # Extract configuration
    # extract_config_from_checkpoint(checkpoint_path)