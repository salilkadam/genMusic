from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import scipy.io.wavfile
import numpy as np
import torch
from transformers import pipeline
import random
import traceback
import time
import logging
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Generation API", description="Generate music using Facebook's MusicGen models")

# Constants
MAX_SINGLE_TRACK_DURATION = 30  # Maximum duration for a single track generation

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class MusicRequest(BaseModel):
    prompt: str
    duration: int = 10  # Duration in seconds
    model_size: ModelSize = ModelSize.SMALL
    
class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    prompt: str
    duration: int
    model_size: str
    progress: str = "Initializing..."
    error_message: Optional[str] = None
    file_paths: Optional[Dict[str, str]] = None
    generation_time: Optional[float] = None
    model_used: Optional[str] = None
    file_sizes_mb: Optional[List[float]] = None
    segments_generated: Optional[int] = None

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    file_paths: Optional[Dict[str, str]] = None
    file_sizes_mb: Optional[List[float]] = None
    segments_generated: Optional[int] = None

# Global task storage (in production, use Redis or database)
tasks: Dict[str, TaskInfo] = {}
model_cache: Dict[str, Any] = {}

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=2)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_model_name(model_size: str) -> str:
    """Get the full model name for the given size."""
    model_map = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium", 
        "large": "facebook/musicgen-large"
    }
    return model_map.get(model_size, "facebook/musicgen-small")

def calculate_segments(duration: int) -> List[int]:
    """Calculate how to split a duration into segments of max 30 seconds each."""
    segments = []
    remaining = duration
    
    while remaining > 0:
        segment_duration = min(remaining, MAX_SINGLE_TRACK_DURATION)
        segments.append(segment_duration)
        remaining -= segment_duration
    
    return segments

def combine_audio_segments(segments: List[Dict[str, Any]], target_duration: int) -> Dict[str, Any]:
    """Combine multiple audio segments into a single track."""
    if not segments:
        raise ValueError("No segments to combine")
    
    # Get the sampling rate from the first segment
    sampling_rate = segments[0]["sampling_rate"]
    
    # Log segment information for debugging
    logger.info(f"🔗 Combining {len(segments)} audio segments...")
    for i, segment in enumerate(segments):
        audio_shape = segment["audio"].shape
        audio_duration = len(segment["audio"]) / sampling_rate
        logger.info(f"   Segment {i+1}: shape={audio_shape}, duration={audio_duration:.2f}s")
    
    # Handle different audio formats (mono, stereo, or 3D)
    audio_arrays = []
    for segment in segments:
        audio = segment["audio"]
        
        # Handle different audio dimensions
        if audio.ndim == 1:
            # Mono audio - reshape to (samples, 1)
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2:
            # Stereo audio - already in correct format (samples, channels)
            pass
        elif audio.ndim == 3:
            # 3D audio - typically (batch, channels, samples) - reshape to (samples, channels)
            if audio.shape[0] == 1:
                # Remove batch dimension and transpose
                audio = audio.squeeze(0).T  # (channels, samples) -> (samples, channels)
            else:
                raise ValueError(f"Unexpected 3D audio shape: {audio.shape}")
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        audio_arrays.append(audio)
    
    # Combine all audio data along the time axis (axis=0)
    try:
        combined_audio = np.concatenate(audio_arrays, axis=0)
        logger.info(f"✅ Successfully combined audio segments. Final shape: {combined_audio.shape}")
    except ValueError as e:
        logger.error(f"❌ Error concatenating audio segments: {e}")
        logger.error(f"   Audio shapes: {[arr.shape for arr in audio_arrays]}")
        raise
    
    # Ensure the combined audio matches the target duration
    target_samples = int(target_duration * sampling_rate)
    
    if combined_audio.shape[0] > target_samples:
        # Trim to target duration
        combined_audio = combined_audio[:target_samples, :]
        logger.info(f"📏 Trimmed audio from {combined_audio.shape[0] / sampling_rate:.2f}s to {target_duration}s")
    elif combined_audio.shape[0] < target_samples:
        # Pad with silence to target duration
        padding_samples = target_samples - combined_audio.shape[0]
        padding_shape = (padding_samples, combined_audio.shape[1])
        padding = np.zeros(padding_shape, dtype=combined_audio.dtype)
        combined_audio = np.concatenate([combined_audio, padding], axis=0)
        logger.info(f"🔇 Padded audio from {combined_audio.shape[0] / sampling_rate:.2f}s to {target_duration}s")
    
    return {
        "audio": combined_audio,
        "sampling_rate": sampling_rate
    }

def load_model_sync(model_name: str, device: str):
    """Synchronously load the model - runs in thread pool."""
    try:
        logger.info(f"📦 Loading {model_name}...")
        
        # Clear CUDA cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with safetensors support
        synthesiser = pipeline(
            "text-to-audio",
            model=model_name,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use half precision for GPU
        )
        
        # Move model to device with error handling
        if device == "cuda" and torch.cuda.is_available():
            try:
                synthesiser.model = synthesiser.model.to(device="cuda", dtype=torch.float16)
                logger.info("✅ Model moved to CUDA with half precision")
            except Exception as move_error:
                logger.warning(f"⚠️ Failed to move model to CUDA: {move_error}")
                # Fallback to CPU
                synthesiser.model = synthesiser.model.to(device="cpu")
                logger.info("✅ Model moved to CPU as fallback")
        
        return synthesiser
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        # Try alternative loading method
        try:
            logger.info("🔄 Attempting alternative loading method...")
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            # Load model and processor separately with safetensors
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            processor = AutoProcessor.from_pretrained(model_name)
            
            # Create pipeline manually
            synthesiser = pipeline(
                "text-to-audio",
                model=model,
                tokenizer=processor,
                device=0 if device == "cuda" else -1
            )
            
            logger.info("✅ Model loaded with alternative method")
            return synthesiser
        except Exception as alt_error:
            logger.error(f"❌ Alternative loading also failed: {alt_error}")
            raise e

def generate_audio_sync(synthesiser, prompt: str, duration: int, seed: int):
    """Synchronously generate audio - runs in thread pool."""
    try:
        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()
        
        # Use more conservative parameters to avoid CUDA errors and overflow
        max_length = min(duration * 50, 1500)  # Cap max_length to prevent overflow
        
        music = synthesiser(
            prompt,
            forward_params={
                "do_sample": True,
                "max_length": max_length,
                "use_cache": False,  # Disable KV cache to reduce memory usage
                "num_beams": 1,      # Use greedy decoding instead of beam search
                "temperature": 1.0,  # Default temperature
                "top_k": 50,         # Limit top-k sampling
                "top_p": 0.9         # Use nucleus sampling
            }
        )
        return music
    except RuntimeError as e:
        if "device-side assert" in str(e):
            logger.error(f"CUDA device-side assert error: {e}")
            # Try with CPU fallback
            logger.info("Attempting CPU fallback...")
            synthesiser.device = torch.device("cpu")
            torch.cuda.empty_cache()
            return synthesiser(
                prompt,
                forward_params={
                    "do_sample": True,
                    "max_length": max_length,
                    "use_cache": False
                }
            )
        else:
            raise e

async def process_music_generation(task_id: str, request: MusicRequest):
    """Background task to process music generation."""
    task = tasks[task_id]
    
    try:
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        task.progress = "Starting music generation..."
        
        request_start_time = time.time()
        logger.info(f"🎵 Processing task {task_id}: '{request.prompt}' (duration: {request.duration}s, model: {request.model_size})")
        
        # Validate duration
        if request.duration <= 0 or request.duration > 120:
            raise ValueError("Duration must be between 1 and 120 seconds")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {device}")
        
        # Get model name
        model_name = get_model_name(request.model_size)
        logger.info(f"🔧 Selected model: {model_name}")
        
        # Load model (with caching)
        task.progress = "Loading model (this may take a few minutes)..."
        
        if model_name not in model_cache:
            logger.info(f"⏳ Loading model (this may take a few minutes)...")
            
            loop = asyncio.get_event_loop()
            try:
                synthesiser = await asyncio.wait_for(
                    loop.run_in_executor(executor, load_model_sync, model_name, device),
                    timeout=600  # 10 minute timeout for initial model loading
                )
                model_cache[model_name] = synthesiser
                logger.info(f"✅ Model loaded and cached successfully")
            except asyncio.TimeoutError:
                logger.error(f"❌ Model loading timed out after 10 minutes")
                raise HTTPException(status_code=504, detail="Model loading timed out")
        else:
            synthesiser = model_cache[model_name]
            logger.info(f"✅ Using cached model")
        
        task.progress = "Model ready, generating audio..."
        
        # Calculate segments if duration > 30 seconds
        if request.duration > MAX_SINGLE_TRACK_DURATION:
            segments = calculate_segments(request.duration)
            logger.info(f"📏 Splitting {request.duration}s into {len(segments)} segments: {segments}")
            task.segments_generated = len(segments)
        else:
            segments = [request.duration]
            task.segments_generated = 1
        
        # Generate audio tracks
        logger.info(f"🎲 Setting up random seeds for generation")
        base_seed = random.randint(0, 2**32 - 1)
        logger.info(f"🔢 Using base seed: {base_seed}")
        
        loop = asyncio.get_event_loop()
        
        # Generate two complete tracks (each potentially multi-segment)
        track1_segments = []
        track2_segments = []
        
        # Track 1 - Generate all segments
        task.progress = f"Generating first track ({len(segments)} segments)..."
        logger.info(f"🎵 Generating first track with {len(segments)} segments")
        
        for i, segment_duration in enumerate(segments):
            segment_progress = f"Generating first track segment {i+1}/{len(segments)} ({segment_duration}s)..."
            task.progress = segment_progress
            logger.info(f"🎵 {segment_progress}")
            
            segment_start = time.time()
            try:
                segment_audio = await asyncio.wait_for(
                    loop.run_in_executor(executor, generate_audio_sync, synthesiser, request.prompt, segment_duration, base_seed + i),
                    timeout=600  # 10 minute timeout for generation
                )
                track1_segments.append(segment_audio)
                segment_time = time.time() - segment_start
                logger.info(f"✅ First track segment {i+1} completed in {segment_time:.2f} seconds")
            except asyncio.TimeoutError:
                logger.error(f"❌ First track segment {i+1} generation timed out")
                raise HTTPException(status_code=504, detail="Audio generation timed out")
        
        # Track 2 - Generate all segments
        task.progress = f"Generating second track ({len(segments)} segments)..."
        logger.info(f"🎵 Generating second track with {len(segments)} segments")
        
        for i, segment_duration in enumerate(segments):
            segment_progress = f"Generating second track segment {i+1}/{len(segments)} ({segment_duration}s)..."
            task.progress = segment_progress
            logger.info(f"🎵 {segment_progress}")
            
            segment_start = time.time()
            try:
                segment_audio = await asyncio.wait_for(
                    loop.run_in_executor(executor, generate_audio_sync, synthesiser, request.prompt, segment_duration, base_seed + 1000 + i),
                    timeout=600  # 10 minute timeout for generation
                )
                track2_segments.append(segment_audio)
                segment_time = time.time() - segment_start
                logger.info(f"✅ Second track segment {i+1} completed in {segment_time:.2f} seconds")
            except asyncio.TimeoutError:
                logger.error(f"❌ Second track segment {i+1} generation timed out")
                raise HTTPException(status_code=504, detail="Audio generation timed out")
        
        # Combine segments into full tracks
        task.progress = "Combining audio segments..."
        logger.info("🔗 Combining audio segments into full tracks...")
        
        try:
            music1 = combine_audio_segments(track1_segments, request.duration)
            music2 = combine_audio_segments(track2_segments, request.duration)
            logger.info("✅ Audio segments combined successfully")
        except Exception as combine_error:
            logger.error(f"❌ Error combining audio segments: {combine_error}")
            raise HTTPException(status_code=500, detail="Error combining audio segments")
        
        # Save audio files
        task.progress = "Saving audio files..."
        logger.info("💾 Saving audio files...")
        
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output1 = os.path.join(output_dir, f"song1_{task_id}_{timestamp}.wav")
        output2 = os.path.join(output_dir, f"song2_{task_id}_{timestamp}.wav")
        
        scipy.io.wavfile.write(output1, rate=music1["sampling_rate"], data=music1["audio"])
        scipy.io.wavfile.write(output2, rate=music2["sampling_rate"], data=music2["audio"])
        
        # Calculate file sizes
        file_sizes = []
        if os.path.exists(output1):
            size1 = os.path.getsize(output1) / (1024 * 1024)  # MB
            file_sizes.append(size1)
        if os.path.exists(output2):
            size2 = os.path.getsize(output2) / (1024 * 1024)  # MB
            file_sizes.append(size2)
        
        # Update task with results
        total_time = time.time() - request_start_time
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.progress = "Music generation completed successfully!"
        task.file_paths = {
            "song1": output1,
            "song2": output2
        }
        task.generation_time = total_time
        task.model_used = model_name
        task.file_sizes_mb = file_sizes
        
        logger.info(f"🎉 Task {task_id} completed successfully!")
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
        logger.info(f"📈 File sizes: {file_sizes}")
        logger.info(f"🔢 Segments generated: {task.segments_generated}")
        
    except Exception as e:
        error_time = time.time() - request_start_time if 'request_start_time' in locals() else 0
        error_msg = f"Error generating music: {str(e)}"
        
        logger.error(f"❌ Task {task_id} failed after {error_time:.2f} seconds")
        logger.error(f"❌ Error details: {error_msg}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        traceback.print_exc()
        
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.progress = f"Failed: {error_msg}"
        task.error_message = error_msg
        
    finally:
        # Cleanup with better error handling
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ CUDA cache cleared successfully")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ CUDA cleanup failed: {cleanup_error}")
            # Don't let cleanup errors affect the main task

@app.post("/generate-music/", response_model=TaskResponse)
async def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    """Create a new music generation task and return task ID immediately."""
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task info
    task = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now(),
        prompt=request.prompt,
        duration=request.duration,
        model_size=request.model_size,
        progress="Task created, queued for processing..."
    )
    
    # Store task
    tasks[task_id] = task
    
    # Start background processing
    background_tasks.add_task(process_music_generation, task_id, request)
    
    logger.info(f"🆕 Created task {task_id} for prompt: '{request.prompt}'")
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Task created successfully. Use /task/{task_id}/status to check progress."
    )

@app.get("/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a specific task."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        generation_time=task.generation_time,
        error_message=task.error_message,
        file_paths=task.file_paths,
        file_sizes_mb=task.file_sizes_mb,
        segments_generated=task.segments_generated
    )

@app.get("/task/{task_id}/download/{file_type}")
async def download_audio_file(task_id: str, file_type: str):
    """Download a specific audio file from a completed task."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    if not task.file_paths:
        raise HTTPException(status_code=404, detail="No files available for this task")
    
    if file_type not in task.file_paths:
        available_files = list(task.file_paths.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"File type '{file_type}' not found. Available files: {available_files}"
        )
    
    file_path = task.file_paths[file_type]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    filename = f"{file_type}_{task_id}.wav"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/task/{task_id}/files")
async def list_task_files(task_id: str):
    """List all available files for a completed task."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    if not task.file_paths:
        return {"files": [], "message": "No files available for this task"}
    
    files_info = []
    for file_type, file_path in task.file_paths.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            files_info.append({
                "file_type": file_type,
                "size_mb": round(size_mb, 2),
                "download_url": f"/task/{task_id}/download/{file_type}"
            })
    
    return {
        "task_id": task_id,
        "files": files_info,
        "total_files": len(files_info)
    }

@app.get("/tasks")
async def list_tasks(limit: int = 10, status: Optional[TaskStatus] = None):
    """List all tasks with optional filtering."""
    
    task_list = list(tasks.values())
    
    # Filter by status if provided
    if status:
        task_list = [t for t in task_list if t.status == status]
    
    # Sort by created_at (newest first)
    task_list.sort(key=lambda x: x.created_at, reverse=True)
    
    # Limit results
    task_list = task_list[:limit]
    
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "prompt": task.prompt,
                "created_at": task.created_at,
                "progress": task.progress
            }
            for task in task_list
        ],
        "total": len(task_list)
    }

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    # Delete files if they exist
    deleted_files = []
    if task.file_paths:
        for file_type, file_path in task.file_paths.items():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(file_type)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
    
    # Remove from tasks
    del tasks[task_id]
    
    logger.info(f"🗑️ Deleted task {task_id} and {len(deleted_files)} files")
    
    return {
        "message": f"Task {task_id} deleted successfully",
        "deleted_files": deleted_files
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cached_models": list(model_cache.keys()),
        "active_tasks": len([t for t in tasks.values() if t.status == TaskStatus.PROCESSING]),
        "total_tasks": len(tasks)
    }

@app.get("/models")
async def list_models():
    """List available models and their status."""
    models = {
        "small": {
            "name": "facebook/musicgen-small",
            "size": "~2.2GB",
            "description": "Fastest model, good for testing and quick generation",
            "cached": "facebook/musicgen-small" in model_cache,
            "recommended_for": "CPU processing"
        },
        "medium": {
            "name": "facebook/musicgen-medium", 
            "size": "~4GB",
            "description": "Balanced performance and quality",
            "cached": "facebook/musicgen-medium" in model_cache,
            "recommended_for": "GPU processing"
        },
        "large": {
            "name": "facebook/musicgen-large",
            "size": "~8.7GB", 
            "description": "Best quality, slowest generation",
            "cached": "facebook/musicgen-large" in model_cache,
            "recommended_for": "High-end GPU processing"
        }
    }
    
    return {
        "models": models,
        "default": "small",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("🚀 FastAPI Music Generation Server Starting Up")
    logger.info(f"⏰ Server started at: {datetime.now()}")
    logger.info(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"🖥️ CUDA devices: {torch.cuda.device_count()}")
    logger.info("📝 Ready to receive music generation requests!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 