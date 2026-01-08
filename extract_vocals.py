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

def process_in_chunks(model, processor, audio_path, prompts, chunk_seconds=30):
    """
    Processes audio in chunks to avoid OOM on large files.
    """
    import soundfile as sf
    import numpy as np
    
    # Load full audio info
    info = sf.info(audio_path)
    sr = info.samplerate
    total_frames = info.frames
    duration = total_frames / sr
    
    print(f"DEBUG: Audio duration: {duration:.2f}s, Sample Rate: {sr}")
    
    # Calculate chunk size in frames
    # SAM Audio works at 48kHz usually, processor handles resampling
    # We'll read from the file directly in chunks to be safe on RAM too, 
    # but simplest is read all to RAM (numpy) then slice.
    
    # Using soundfile to read as numpy array
    full_audio, file_sr = sf.read(audio_path)
    if len(full_audio.shape) > 1:
        # Mix to mono for simple processing/chunking logic (model handles mono/stereo usually but let's stick to what worked)
        # Processor `batch_audio` handles it. 
        # But here we need to slice raw audio.
        # Let's keep distinct channels if present.
        pass
        # Actually processor converts to tensor.
    
    chunks = []
    chunk_size_samples = int(chunk_seconds * file_sr)
    
    # Iterate
    full_target_wavs = []
    full_residual_wavs = []
    
    print(f"Processing in {chunk_seconds}s chunks...")
    
    for start in range(0, len(full_audio), chunk_size_samples):
        end = min(start + chunk_size_samples, len(full_audio))
        chunk_data = full_audio[start:end]
        
        # Save chunk to temp file or pass tensor directly?
        # processor accepts list of str OR tensors.
        # Let's use tensors to avoid disk I/O
        
        # Convert to torch tensor
        # soundfile returns (frames, channels) or (frames,)
        chunk_tensor = torch.from_numpy(chunk_data).float()
        
        # SAMAudioProcessor expects (channels, time) for tensor input or just file path
        if chunk_tensor.ndim == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0) # (1, time)
        else:
            chunk_tensor = chunk_tensor.t() # (time, channels) -> (channels, time)
            
        # Process this chunk
        try:
            batch = processor(
                audios=[chunk_tensor],
                descriptions=prompts,
            )
            if torch.cuda.is_available():
                batch = batch.to("cuda")
                
            with torch.inference_mode():
                # Running separate on chunk
                result = model.separate(batch, predict_spans=False, reranking_candidates=1)
                
            # Collect results (numpy)
            # result.target is list of wavs. [0] is our result.
            # Shape: [1, length] ?
            # Let's convert to numpy T to match sf.write expectation (time, channels)
            
            t_wav = result.target[0].cpu().numpy().T
            r_wav = result.residual[0].cpu().numpy().T
            
            full_target_wavs.append(t_wav)
            full_residual_wavs.append(r_wav)
            
            print(f"  Processed chunk {start/file_sr:.1f}s - {end/file_sr:.1f}s")
            
            # Clear cache to free VRAM
            del batch, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing chunk starting at {start}: {e}")
            import traceback
            traceback.print_exc()
            # If a chunk fails, we might want to fill silence or abort.
            return None, None

    # Concatenate
    final_target = np.concatenate(full_target_wavs, axis=0)
    final_residual = np.concatenate(full_residual_wavs, axis=0)
    
    return final_target, final_residual


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
            # print(f"Device count: {torch.cuda.device_count()}")
            # print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            model = model.cuda()
            print("Model loaded on CUDA.")
        else:
            print("Model loaded on CPU.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Processing audio: {audio_path}")
    print(f"Prompts: {prompts}")

    # Process
    try:
        # Use chunking if file is long? Or always use it to be safe?
        # Let's always use chunking for robustness, or checking length.
        # But for logic simplicity, let's just call the chunker. 
        # It handles the full flow natively.
        
        target_wav, residual_wav = process_in_chunks(model, processor, audio_path, prompts, chunk_seconds=30)
        
        if target_wav is None:
            print("Processing failed.")
            sys.exit(1)

        # Save output
        sample_rate = processor.audio_sampling_rate
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        description_sanitized = prompts[0].replace(" ", "_")
        target_path = os.path.join(output_dir, f"{base_name}_{description_sanitized}.wav")
        residual_path = os.path.join(output_dir, f"{base_name}_residual.wav")

        print(f"Saving outputs to {output_dir}")
        sf.write(target_path, target_wav, sample_rate)
        sf.write(residual_path, residual_wav, sample_rate)
        
        print("Done!")
        print(f"  Target: {target_path}")
        print(f"  Residual: {residual_path}")
        
    except Exception as e:
        print(f"Error processing audio input: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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

