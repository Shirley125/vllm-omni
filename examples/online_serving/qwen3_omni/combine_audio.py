#!/usr/bin/env python3
"""
Script to combine multiple WAV audio files into a single output file.
Supports both pydub (recommended) and wave (standard library) methods.
"""

import argparse
import os
import sys


def combine_wav_files_pydub(input_files, output_file):
    """
    Combine multiple WAV files using pydub (requires: pip install pydub)
    This method handles different sample rates and channels automatically.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Error: pydub is not installed. Install it with: pip install pydub")
        print("Alternatively, use the --use-wave flag to use the standard library.")
        sys.exit(1)

    # Load the first audio file
    combined = AudioSegment.from_wav(input_files[0])

    # Append all other audio files
    for audio_file in input_files[1:]:
        print(f"Adding: {audio_file}")
        audio = AudioSegment.from_wav(audio_file)
        combined += audio

    # Export the combined audio
    combined.export(output_file, format="wav")
    print(f"\nSuccessfully combined {len(input_files)} files into: {output_file}")
    print(f"Output duration: {len(combined) / 1000.0:.2f} seconds")


def combine_wav_files_wave(input_files, output_file):
    """
    Combine multiple WAV files using the standard library wave module.
    Note: This assumes all files have the same sample rate, channels, and sample width.
    """
    import wave

    # Open the first file to get parameters
    with wave.open(input_files[0], "rb") as first_wav:
        params = first_wav.getparams()
        frames = first_wav.readframes(first_wav.getnframes())

    # Open output file for writing
    with wave.open(output_file, "wb") as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

        # Append frames from remaining files
        for audio_file in input_files[1:]:
            print(f"Adding: {audio_file}")
            with wave.open(audio_file, "rb") as in_wav:
                # Verify parameters match
                if in_wav.getparams() != params:
                    print(f"Warning: {audio_file} has different parameters. Skipping.")
                    continue
                frames = in_wav.readframes(in_wav.getnframes())
                out_wav.writeframes(frames)

    print(f"\nSuccessfully combined {len(input_files)} files into: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple WAV audio files into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all audio_*.wav files in current directory
  python combine_audio.py -o combined.wav audio_*.wav

  # Combine specific files
  python combine_audio.py -o output.wav audio_0.wav audio_1.wav audio_2.wav

  # Use wave module instead of pydub
  python combine_audio.py -o output.wav --use-wave audio_*.wav
        """,
    )

    parser.add_argument(
        "input_files", nargs="+", help="Input WAV files to combine (can use wildcards like audio_*.wav)"
    )

    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")

    parser.add_argument(
        "--use-wave",
        action="store_true",
        help="Use standard library wave module instead of pydub (requires matching audio parameters)",
    )

    parser.add_argument("--sort", action="store_true", help="Sort input files alphabetically before combining")

    args = parser.parse_args()

    # Expand wildcards and get absolute paths
    input_files = []
    for pattern in args.input_files:
        if "*" in pattern or "?" in pattern:
            import glob

            input_files.extend(sorted(glob.glob(pattern)))
        else:
            input_files.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    input_files = [f for f in input_files if not (f in seen or seen.add(f))]

    # Sort if requested
    if args.sort:
        input_files = sorted(input_files)

    # Validate files exist
    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)
        if not f.lower().endswith(".wav"):
            print(f"Warning: {f} does not have .wav extension")

    if len(input_files) == 0:
        print("Error: No input files found")
        sys.exit(1)

    print(f"Combining {len(input_files)} audio files...")
    print(f"Input files: {', '.join(input_files)}")

    # Combine using selected method
    if args.use_wave:
        combine_wav_files_wave(input_files, args.output)
    else:
        combine_wav_files_pydub(input_files, args.output)


if __name__ == "__main__":
    main()
    # python combine_audio.py -o output.wav --use-wave audio_*.wav
