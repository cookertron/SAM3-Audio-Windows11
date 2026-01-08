import sys
import os

# Add current directory to path so sam_audio can be imported without installation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download

# Delayed import of sam_audio to allow setup to run even if dependencies are missing
# from sam_audio import SAMAudio, SAMAudioProcessor

# Constants
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "sam-audio-large")
MODEL_REPO = "facebook/sam-audio-large"
CHECKPOINT_FILENAME = "checkpoint.pt"

def setup_model_dir():
    """Ensures the checkpoint directory exists and has the necessary config."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    # Check for config.json
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(config_path):
        print("Downloading config.json from Hugging Face...")
        try:
            hf_hub_download(repo_id=MODEL_REPO, filename="config.json", local_dir=CHECKPOINT_DIR)
            print("config.json downloaded successfully.")
        except Exception as e:
            print(f"Warning: Could not download config.json (auth required?).")
            print(f"Please ensure you are logged in (huggingface-cli login) OR")
            print(f"manually download 'config.json' from {MODEL_REPO} and place it in:")
            print(f"  {CHECKPOINT_DIR}")
            # We don't exit here, we let the user know what's missing later or let them fix it.
            # But checking for checkpoint now.
    
    # Check for checkpoint.pt
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint_path):
        print(f"\n[IMPORTANT] Model checkpoint missing!")
        print(f"Please move your downloaded '{CHECKPOINT_FILENAME}' to:")
        print(f"  {CHECKPOINT_DIR}")
        print("Then run this script again.")
        sys.exit(1)

    return CHECKPOINT_DIR

def extract_vocals(audio_path, output_dir=".", prompts=["vocals"]):
    """Extracts audio based on the given prompts from the audio file."""
    try:
        from sam_audio import SAMAudio, SAMAudioProcessor
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"Error importing sam_audio: {e}")
        print("Please ensure you have installed the package with `pip install .`")
        sys.exit(1)
    
    # print(f"DEBUG: sam_audio loaded from: {SAMAudio.__module__}")
    # import sam_audio
    # print(f"DEBUG: sam_audio file: {sam_audio.__file__}")
    
    print(f"Loading model from {CHECKPOINT_DIR}...")
    try:
        model = SAMAudio.from_pretrained(CHECKPOINT_DIR)
        processor = SAMAudioProcessor.from_pretrained(CHECKPOINT_DIR)
        model = model.eval()
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            model = model.cuda()
            print("Model loaded on CUDA.")
            # Verify a parameter is on CUDA
            param_device = next(model.parameters()).device
            print(f"Model parameter device: {param_device}")
        else:
            print("Model loaded on CPU.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Processing audio: {audio_path}")
    print(f"Prompts: {prompts}")
    
    descriptions = prompts

    # Prepare batch
    # Note: SAMAudioProcessor handles loading and resampling
    try:
        batch = processor(
            audios=[audio_path],
            descriptions=descriptions,
        )
        if torch.cuda.is_available():
           batch = batch.to("cuda")
    except Exception as e:
        print(f"Error processing audio input: {e}")
        sys.exit(1)

    print("Running separation...")
    with torch.inference_mode():
        # Using default parameters for balance of speed/quality
        # predict_spans=True can help if vocals are sparse, but False is safer for general tracks
        result = model.separate(batch, predict_spans=False, reranking_candidates=1)

    # Save output
    sample_rate = processor.audio_sampling_rate
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Iterate through results (handling multiple prompts if we supported batching multiple prompts, 
    # but separate returns a BatchResult. target is [B, T]. B=1 here unless we batch prompts differently?
    # SAMAudio.separate returns result where result.target is [BatchSize, Time] corresponding to descriptions.
    # But wait, processor documentation says descriptions list length matching audios?
    # Actually, SAMAudioProcessor takes `descriptions` matching `audios` length.
    # If we want 1 audio with N prompts, we usually need to repeat the audio.
    # Let's check if the current implementation supports 1 audio : 1 prompt. 
    # Yes, typically 1 input audio matches 1 description.
    # If we pass prompts=["vocals"], len=1. 
    # If users wants multiple prompts at once, we should technically handle that. 
    # BUT for simplicity, let's assume prompts is a LIST of length 1 for now, or we re-implement to separate loop.
    # Actually, to support multiple prompts properly with the current processor structure (1-to-1 audio-desc),
    # we might just take the first prompt for now or loop.
    # Let's assume prompts[0] is the main one for this single-file function.
    
    # Wait, if we want to extract different things, we could iterate.
    # But for now, let's stick to the FIRST prompt in the list to match current behavior 1:1.
    
    # Re-reading processor.py:
    # assert self.audios.size(0) == len(self.descriptions)
    # So if we have 1 audio file and want multiple prompts, we'd need to duplicate the audio file string in the list passed to processor.
    
    # Let's stick to single prompt extraction for this function signature update to keep it simple and robust.
    # Or, we handle the multi-prompt logic here.
    # Let's loop here if len(prompts) > 1? No, better to stick to 1 prompt per call for clarity,
    # OR replicate the audio list.
    
    # Let's assume prompts is meant to be a single string for valid "extraction" context (one target).
    # IF the user passes multiple, we'll just use the first one and warn? 
    # Or we construct the batch correctly. 
    
    # Let's just output specifically for the first prompt.
    description_sanitized = prompts[0].replace(" ", "_")
    target_path = os.path.join(output_dir, f"{base_name}_{description_sanitized}.wav")
    residual_path = os.path.join(output_dir, f"{base_name}_residual.wav")

    print(f"Saving outputs to {output_dir}")
    
    # Use soundfile directly to avoid torchaudio backend issues
    target_wav = result.target[0].cpu().numpy().T
    residual_wav = result.residual[0].cpu().numpy().T
    sf.write(target_path, target_wav, sample_rate)
    sf.write(residual_path, residual_wav, sample_rate)
    
    print("Done!")
    print(f"  Target: {target_path}")
    print(f"  Residual: {residual_path}")

if __name__ == "__main__":
    setup_model_dir()
    
    if len(sys.argv) < 2:
        print("Usage: python extract_vocals.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
        
    extract_vocals(audio_file)
