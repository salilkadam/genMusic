from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import scipy.io.wavfile
import torch
from transformers import pipeline
import random
import traceback
import time
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class MusicRequest(BaseModel):
    prompt: str
    duration: int  # Duration for each track

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@app.post("/generate-music/")
async def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    request_start_time = time.time()
    logger.info(f"🎵 Starting music generation request: '{request.prompt}' (duration: {request.duration}s)")
    
    if request.duration <= 0:
        logger.error(f"❌ Invalid duration: {request.duration}")
        raise HTTPException(status_code=400, detail="Duration must be greater than zero")

    synthesiser = None

    try:
        # Set device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

        # Optionally limit GPU memory usage
        if device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.8, device=0)
                logger.info("⚙️ Limited GPU memory usage to 80%")
            except Exception as mem_error:
                logger.warning(f"⚠️ Failed to limit GPU memory: {mem_error}")

        # Load MusicGen Large model
        model_load_start = time.time()
        logger.info("📦 Loading MusicGen Large model...")
        logger.info("⏳ This may take several minutes on first run (model download/loading)")
        
        synthesiser = pipeline(
            "text-to-audio", 
            model="facebook/musicgen-large", 
            device=0 if device.type == 'cuda' else -1
        )
        
        model_load_time = time.time() - model_load_start
        logger.info(f"✅ Model loaded successfully in {model_load_time:.2f} seconds")

        # Generate two audio tracks using a random seed
        logger.info("🎲 Setting up random seeds for generation")
        random_seed = random.randint(0, 2**32 - 1)
        logger.info(f"🔢 Using random seed: {random_seed}")
        
        torch.manual_seed(random_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(random_seed)

        # Generate first track
        logger.info("🎵 Generating first audio track...")
        track1_start = time.time()
        
        music1 = synthesiser(
            request.prompt, 
            forward_params={
                "do_sample": True, 
                "max_length": request.duration * 50
            }
        )
        
        track1_time = time.time() - track1_start
        logger.info(f"✅ First track generated in {track1_time:.2f} seconds")
        logger.info(f"📊 First track - Sample rate: {music1['sampling_rate']}, Audio shape: {music1['audio'].shape}")
        
        # Generate second track
        logger.info("🎵 Generating second audio track...")
        track2_start = time.time()
        
        random_seed += 1
        torch.manual_seed(random_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(random_seed)
        
        music2 = synthesiser(
            request.prompt, 
            forward_params={
                "do_sample": True, 
                "max_length": request.duration * 50
            }
        )
        
        track2_time = time.time() - track2_start
        logger.info(f"✅ Second track generated in {track2_time:.2f} seconds")
        logger.info(f"📊 Second track - Sample rate: {music2['sampling_rate']}, Audio shape: {music2['audio'].shape}")

        # Save audio files
        logger.info("💾 Saving audio files...")
        save_start = time.time()
        
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        output1 = os.path.join(output_dir, "song1.wav")
        output2 = os.path.join(output_dir, "song2.wav")
        
        logger.info(f"💾 Saving first track to: {output1}")
        scipy.io.wavfile.write(output1, rate=music1["sampling_rate"], data=music1["audio"])
        
        logger.info(f"💾 Saving second track to: {output2}")
        scipy.io.wavfile.write(output2, rate=music2["sampling_rate"], data=music2["audio"])
        
        save_time = time.time() - save_start
        logger.info(f"✅ Audio files saved in {save_time:.2f} seconds")
        
        # Calculate total time
        total_time = time.time() - request_start_time
        logger.info(f"🎉 Music generation completed successfully!")
        logger.info(f"⏱️ Total time breakdown:")
        logger.info(f"   - Model loading: {model_load_time:.2f}s")
        logger.info(f"   - Track 1 generation: {track1_time:.2f}s")
        logger.info(f"   - Track 2 generation: {track2_time:.2f}s")
        logger.info(f"   - File saving: {save_time:.2f}s")
        logger.info(f"   - Total request time: {total_time:.2f}s")
        
        # Check file sizes
        if os.path.exists(output1):
            size1 = os.path.getsize(output1)
            logger.info(f"📈 File 1 size: {size1 / (1024*1024):.2f} MB")
        
        if os.path.exists(output2):
            size2 = os.path.getsize(output2)
            logger.info(f"📈 File 2 size: {size2 / (1024*1024):.2f} MB")

        return {"song1": output1, "song2": output2}

    except Exception as e:
        error_time = time.time() - request_start_time
        logger.error(f"❌ Error occurred after {error_time:.2f} seconds")
        logger.error(f"❌ Error details: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        
        # Log the full traceback
        logger.error("❌ Full traceback:")
        traceback.print_exc()
        
        raise HTTPException(status_code=500, detail=f"Error generating music: {e}")

    finally:
        cleanup_start = time.time()
        logger.info("🧹 Starting cleanup...")
        
        if synthesiser:
            logger.info("🧹 Deleting synthesiser object...")
            del synthesiser
        
        if torch.cuda.is_available():
            logger.info("🧹 Clearing CUDA cache...")
            torch.cuda.empty_cache()
        
        cleanup_time = time.time() - cleanup_start
        logger.info(f"✅ Cleanup completed in {cleanup_time:.2f} seconds")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI Music Generation Server Starting Up")
    logger.info(f"⏰ Server started at: {datetime.now()}")
    logger.info(f"🖥️ Available devices: {torch.cuda.device_count()} CUDA devices" if torch.cuda.is_available() else "🖥️ Using CPU only")
    logger.info("📝 Ready to receive music generation requests!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
