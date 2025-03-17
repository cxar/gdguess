#!/usr/bin/env python3
"""
Interactive microphone interface for real-time Grateful Dead show dating.
"""

import time
import numpy as np
import torch
import datetime
import queue
import threading
import logging

from inference.utils.model_loader import load_model
from inference.base_inference import extract_audio_features
from inference.utils.tta import test_time_augmentation
from inference.interactive.mic_capture import MicrophoneCapture
from inference.interactive.visualization import setup_visualization, update_visualization


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mic_interface")


class LiveInferenceProcessor:
    """Process audio from microphone for live inference."""
    
    def __init__(
        self,
        model,
        sample_rate=24000,
        buffer_duration=15,
        device="cuda",
        use_tta=False,
        tta_transforms=3,
        tta_intensity="light",
    ):
        """
        Initialize the live inference processor.
        
        Args:
            model: PyTorch model for inference
            sample_rate: Audio sample rate
            buffer_duration: Audio buffer duration in seconds
            device: PyTorch device
            use_tta: Whether to use test-time augmentation
            tta_transforms: Number of TTA transforms to use
            tta_intensity: Intensity of TTA transforms
        """
        self.model = model
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = sample_rate * buffer_duration
        self.device = device
        self.use_tta = use_tta
        self.tta_transforms = tta_transforms
        self.tta_intensity = tta_intensity
        
        # Audio processing state
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_filled = False
        self.prediction_queue = queue.Queue()
        
        # Initialize visualization
        self.fig, self.ax_waveform, self.ax_prediction = setup_visualization()
        
        # Base date for model
        self.base_date = datetime.date(1968, 1, 1)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.running = True
        self.processing_thread.start()
    
    def add_audio(self, audio_chunk):
        """
        Add audio chunk to the buffer.
        
        Args:
            audio_chunk: Numpy array of audio samples
        """
        # Roll buffer and add new chunk
        chunk_length = min(len(audio_chunk), self.buffer_size)
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_length)
        self.audio_buffer[-chunk_length:] = audio_chunk[-chunk_length:]
        
        # Mark buffer as filled if we have enough audio
        if not self.buffer_filled:
            # Check if buffer has non-zero values in the first half
            if np.any(self.audio_buffer[:self.buffer_size//2] != 0):
                self.buffer_filled = True
                logger.info("Audio buffer filled, starting inference")
    
    def _process_loop(self):
        """Background thread for audio processing and inference."""
        last_process_time = 0
        throttle_interval = 1.0  # Process at most once per second
        
        # For smoother visualization, we'll keep a rolling average of predictions
        rolling_predictions = []
        max_rolling_size = 3
        
        while self.running:
            if not self.buffer_filled:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            if current_time - last_process_time < throttle_interval:
                time.sleep(0.05)
                continue
            
            # Clone current buffer for processing
            audio_data = self.audio_buffer.copy()
            
            # Normalize audio before processing
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            try:
                # Process audio with or without TTA
                if self.use_tta:
                    results = test_time_augmentation(
                        model=self.model,
                        audio=audio_data,
                        sr=self.sample_rate,
                        num_transforms=self.tta_transforms,
                        intensity=self.tta_intensity,
                        device=self.device,
                        extract_features_fn=extract_audio_features
                    )
                else:
                    # Extract features
                    features = extract_audio_features(audio_data, self.sample_rate)
                    
                    # Move to device and add batch dimension
                    features = {k: v.to(self.device).unsqueeze(0) if v.dim() == 1 else v.unsqueeze(0).to(self.device) 
                              for k, v in features.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        results = self.model(features)
                
                # Interpret prediction
                if isinstance(results, dict):
                    # Handle dictionary output
                    if "days_offset" in results:
                        days = results["days_offset"].item()
                        date = self.base_date + datetime.timedelta(days=days)
                        
                        # Add to the prediction queue for visualization
                        era = None
                        era_confidence = None
                        
                        if "era_logits" in results:
                            # Get predicted era
                            era_probs = torch.nn.functional.softmax(results["era_logits"], dim=-1)
                            era_idx = era_probs.argmax().item()
                            era_confidence = era_probs[0, era_idx].item()
                            
                            # Map index to era
                            era_map = {0: "early", 1: "seventies", 2: "eighties", 3: "nineties"}
                            era = era_map[era_idx]
                        
                        prediction = {
                            "date": date,
                            "era": era,
                            "era_confidence": era_confidence
                        }
                        
                        # Add to rolling predictions for smoothing
                        rolling_predictions.append(prediction)
                        if len(rolling_predictions) > max_rolling_size:
                            rolling_predictions.pop(0)
                        
                        # Average the dates
                        if len(rolling_predictions) > 1:
                            # For dates, we average the day offsets
                            avg_days = sum((d["date"] - self.base_date).days for d in rolling_predictions) // len(rolling_predictions)
                            avg_date = self.base_date + datetime.timedelta(days=avg_days)
                            
                            # For era, we take the most common
                            eras = [d["era"] for d in rolling_predictions if d["era"] is not None]
                            if eras:
                                from collections import Counter
                                most_common_era = Counter(eras).most_common(1)[0][0]
                                
                                # Get average confidence for this era
                                era_confs = [d["era_confidence"] for d in rolling_predictions 
                                           if d["era"] == most_common_era and d["era_confidence"] is not None]
                                avg_confidence = sum(era_confs) / len(era_confs) if era_confs else None
                            else:
                                most_common_era = None
                                avg_confidence = None
                            
                            smoothed_prediction = {
                                "date": avg_date,
                                "era": most_common_era,
                                "era_confidence": avg_confidence
                            }
                            
                            self.prediction_queue.put(smoothed_prediction)
                        else:
                            # Just use the current prediction if we don't have enough for averaging
                            self.prediction_queue.put(prediction)
                else:
                    # Handle tensor output (simpler model)
                    days = results.item()
                    date = self.base_date + datetime.timedelta(days=days)
                    self.prediction_queue.put({"date": date})
            
            except Exception as e:
                logger.error(f"Error in inference: {e}")
            
            last_process_time = time.time()
    
    def update_display(self):
        """Update the visualization with latest audio and prediction."""
        # Update the waveform display
        update_visualization(
            self.fig,
            self.ax_waveform,
            self.ax_prediction,
            self.audio_buffer,
            self.prediction_queue
        )
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)


def run_interactive_inference(
    checkpoint_path,
    device="cuda",
    sample_rate=24000,
    use_tta=False,
    tta_transforms=3,
    tta_intensity="light"
):
    """
    Run interactive inference with microphone input.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: PyTorch device string
        sample_rate: Audio sample rate
        use_tta: Whether to use test-time augmentation
        tta_transforms: Number of TTA transforms to use
        tta_intensity: Intensity of TTA transforms
    """
    try:
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        model = load_model(checkpoint_path, device)
        model.eval()
        
        # Create processor for inference
        print("Initializing inference processor...")
        processor = LiveInferenceProcessor(
            model=model,
            sample_rate=sample_rate,
            device=device,
            use_tta=use_tta,
            tta_transforms=tta_transforms,
            tta_intensity=tta_intensity
        )
        
        # Create mic capture
        print("Initializing microphone capture...")
        mic = MicrophoneCapture(
            sample_rate=sample_rate,
            chunk_size=1024,
            channels=1
        )
        
        # Print instructions
        print("\n====== GRATEFUL DEAD SHOW DATING ======")
        print("Play or sing some Grateful Dead music for dating")
        if use_tta:
            print(f"Using test-time augmentation with {tta_transforms} transforms at {tta_intensity} intensity")
        print("Press Ctrl+C to exit")
        print("========================================\n")
        
        # Start microphone capture
        mic.start(callback=processor.add_audio)
        
        # Main loop for visualization
        try:
            while True:
                processor.update_display()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Clean up
            mic.stop()
            processor.stop()
    
    except Exception as e:
        logger.error(f"Error in interactive inference: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Grateful Dead Show Dating")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Audio sample rate")
    parser.add_argument("--use-tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--tta-transforms", type=int, default=3, help="Number of TTA transforms")
    parser.add_argument("--tta-intensity", type=str, default="light", choices=["light", "medium", "heavy"], 
                       help="TTA intensity")
    
    args = parser.parse_args()
    
    run_interactive_inference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        sample_rate=args.sample_rate,
        use_tta=args.use_tta,
        tta_transforms=args.tta_transforms,
        tta_intensity=args.tta_intensity
    ) 