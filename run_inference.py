import numpy as np
import tflite_runtime.interpreter as tflite
import scipy.io.wavfile as wav
import sys

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "model.tflite"
TARGET_SAMPLE_RATE = 16000  # change if your training pipeline used a different rate
EXPECTED_SHAPE = (32, 32, 1)  # change to match YOUR model input
# --------------------------

def load_wav_mono(path, target_sr):
    sr, audio = wav.read(path)

    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

    # Convert stereo -> mono
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample
        n = int(len(audio) * target_sr / sr)
        audio = resample(audio, n)

    return audio


def preprocess(audio):
    """
    MATCH exactly whatever preprocessing your training used.
    For now: simple amplitude normalization + reshape.
    Replace this with your exact frontend later.
    """
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Example: convert raw audio to the expected input format.
    # If your model expected spectrograms or MFCCs, compute those instead.
    # For now, we fake a simple reshape just to test the pipeline.

    # Resize audio into EXPECTED_SHAPE
    # WARNING: This is only for testing the pipeline!
    from skimage.transform import resize
    x = resize(audio, EXPECTED_SHAPE, mode="reflect").astype(np.float32)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    return x


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_inference.py input.wav")
        return

    wav_path = sys.argv[1]

    # Load model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load audio
    audio = load_wav_mono(wav_path, TARGET_SAMPLE_RATE)

    # Preprocess
    x = preprocess(audio)

    # Ensure dtype matches model
    x = x.astype(input_details[0]["dtype"])

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    print("Inference output:", output)


if __name__ == "__main__":
    main()
