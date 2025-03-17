#!/usr/bin/env python3
"""
Microphone capture module for real-time inference with the Grateful Dead show dating model.
"""

import numpy as np
import pyaudio


class MicrophoneCapture:
    """Class to handle microphone audio capture with a buffer."""

    def __init__(
        self, sample_rate=24000, chunk_size=1024, channels=1, buffer_seconds=15
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32
        self.buffer_seconds = buffer_seconds
        self.buffer_size = self.sample_rate * self.buffer_seconds

        # Initialize buffer with zeros
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # PyAudio instance
        self.p = pyaudio.PyAudio()

        # Stream variable will be set when we start streaming
        self.stream = None

        # Threading control
        self.is_recording = False
        self.thread = None

    def start_recording(self):
        """Start capturing audio from the microphone."""
        if self.is_recording:
            print("Already recording!")
            return

        self.is_recording = True

        # Open stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )

        print("Started recording from microphone")

        # Start stream
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Roll the buffer and add new data
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data

        return (in_data, pyaudio.paContinue)

    def get_audio(self):
        """Get the current audio buffer."""
        return self.audio_buffer.copy()

    def stop_recording(self):
        """Stop recording and close the audio stream."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        print("Stopped recording")

    def __del__(self):
        """Clean up resources."""
        self.stop_recording()
        if self.p is not None:
            self.p.terminate() 