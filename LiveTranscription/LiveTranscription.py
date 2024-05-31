import argparse
import os
import csv
import numpy as np
import speech_recognition as sr
import whisper
import torch
from pynput import keyboard
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

def main():
    # Command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    
    args = parser.parse_args()

    # Initialize phrase_time
    phrase_time = None

    # Create a safe thread pass through from the threaded recording callback
    data_queue = Queue()

    # Record speech using SpeechRecognizer
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = args.energy_threshold

    # Continuously record by lowering energy threshold 
    recognizer.dynamic_energy_threshold = False

    # Load in microphone
    source = sr.Microphone(sample_rate=16000)

    # Load model 
    model_name = args.model + ".en"
    audio_model = whisper.load_model(model_name)

    transcription = ['']

    with source:
        recognizer.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread-safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread using SR helper
    recognizer.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    transcribe = False

    # Listening for start and stop keys
    def on_press(key):
        nonlocal transcribe
        try:
            if key.char == 's':
                transcribe = True
                print("Transcription started. Press 'e' to stop.")
            elif key.char == 'e':
                transcribe = False
                print("Transcription stopped. Press 's' to start again.")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Open a CSV file to write the transcriptions
    with open('transcriptions.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Cue the user that we're ready to go.
        print("Model loaded.\n")
        
        while True:
            if transcribe:
                try:
                    now = datetime.utcnow()
                    # Pull raw recorded audio from the queue.
                    if not data_queue.empty():
                        phrase_complete = False
                        # If enough time has passed between recordings, consider the phrase complete.
                        # Clear the current working audio buffer to start over with the new data.
                        if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                            phrase_complete = True
                        # This is the last time we received new audio data from the queue.
                        phrase_time = now
                        
                        # Combine audio data from queue
                        audio_data = b''.join(data_queue.queue)
                        data_queue.queue.clear()
                        
                        # Convert in-ram buffer to something the model can use directly without needing a temp file.
                        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        # Read the transcription.
                        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                        text = result['text'].strip()

                        # Write the transcription to the CSV file with a timestamp
                        writer.writerow({'timestamp': now.isoformat(), 'transcription': text})

                        # If we detected a pause between recordings, add a new item to our transcription.
                        # Otherwise edit the existing one.
                        if phrase_complete:
                            transcription.append(text)
                        else:
                            transcription[-1] = text

                        # Clear the console to reprint the updated transcription.
                        os.system('cls' if os.name=='nt' else 'clear')
                        for line in transcription:
                            print(line)
                        # Flush stdout.
                        print('', end='', flush=True)
                    else:
                        # Infinite loops are bad for processors, must sleep.
                        sleep(0.25)
                except KeyboardInterrupt:
                    break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
