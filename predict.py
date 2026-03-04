"""Cog predictor for song cover generation using RVC."""

from __future__ import annotations

import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Make the ultimate_rvc package importable inside the Cog container.
# Cog copies the project to /src, so the package root is /src/src.
sys.path.insert(0, "/src/src")

from cog import BasePredictor, Input
from cog import Path as CogPath

from ultimate_rvc.common import RVC_MODELS_DIR, VOICE_MODELS_DIR
from ultimate_rvc.core.generate.song_cover import run_pipeline
from ultimate_rvc.typing_extra import AudioExt, F0Method

# HuggingFace 资源，与 prerequisites_download 一致
PREREQUISITES_BASE = (
    "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"
)
PREDICTOR_FILES = ("rmvpe.pt", "fcpe.pt")


def _ensure_predictors() -> None:
    """Download RVC predictor models (rmvpe.pt, fcpe.pt) if missing."""
    predictors_dir = RVC_MODELS_DIR / "predictors"
    predictors_dir.mkdir(parents=True, exist_ok=True)
    for name in PREDICTOR_FILES:
        path = predictors_dir / name
        if path.is_file():
            continue
        url = f"{PREREQUISITES_BASE}/predictors/{name}"
        urlretrieve(url, path)


class Predictor(BasePredictor):
    """Cog predictor that generates an AI song cover via RVC voice conversion."""

    def setup(self) -> None:
        """Pre-initialise binaries, required directories, and predictor models."""
        import static_ffmpeg
        import static_sox

        static_ffmpeg.add_paths()
        static_sox.add_paths()
        VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_predictors()

    def predict(
        self,
        song_input: CogPath = Input(
            description=(
                "Audio file to convert. Upload a file directly or provide "
                "a URL (e.g. https://example.com/song.mp3). "
                "Supported formats: mp3, wav, flac, ogg, m4a, aac."
            ),
        ),
        rvc_model: CogPath = Input(
            description=(
                "Zip archive containing your RVC voice model. "
                "Must include a .pth file and optionally a .index file."
            ),
        ),
        pitch: int = Input(
            description=(
                "Semitone pitch shift for the converted vocals. "
                "Positive values raise pitch, negative values lower it."
            ),
            default=0,
            ge=-24,
            le=24,
        ),
        f0_method: str = Input(
            description="Pitch extraction method used during vocal conversion.",
            default="rmvpe",
            choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"],
        ),
        index_rate: float = Input(
            description=(
                "Influence of the .index file on vocal conversion (0–1). "
                "Higher values make output closer to the voice model."
            ),
            default=0.3,
            ge=0.0,
            le=1.0,
        ),
        rms_mix_rate: float = Input(
            description=(
                "Blending rate of the volume envelope of the converted "
                "vocals with the original (0–1)."
            ),
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        protect_rate: float = Input(
            description=(
                "Protection strength for consonants and breathing sounds "
                "(0–0.5). Lower values protect more."
            ),
            default=0.33,
            ge=0.0,
            le=0.5,
        ),
        split_vocals: bool = Input(
            description="Split audio into segments before vocal conversion.",
            default=False,
        ),
        autotune_vocals: bool = Input(
            description="Apply autotune to the converted vocals.",
            default=False,
        ),
        clean_vocals: bool = Input(
            description="Apply noise reduction to the converted vocals.",
            default=False,
        ),
        room_size: float = Input(
            description="Room size of the reverb effect on converted vocals (0–1).",
            default=0.15,
            ge=0.0,
            le=1.0,
        ),
        wet_level: float = Input(
            description="Wet level of the reverb effect on converted vocals (0–1).",
            default=0.2,
            ge=0.0,
            le=1.0,
        ),
        dry_level: float = Input(
            description="Dry level of the reverb effect on converted vocals (0–1).",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        main_gain: int = Input(
            description="Volume gain (dB) for the converted vocal track.",
            default=0,
            ge=-20,
            le=20,
        ),
        inst_gain: int = Input(
            description="Volume gain (dB) for the instrumental track.",
            default=0,
            ge=-20,
            le=20,
        ),
        backup_gain: int = Input(
            description="Volume gain (dB) for the backup vocals track.",
            default=0,
            ge=-20,
            le=20,
        ),
        output_format: str = Input(
            description="Format of the output audio file.",
            default="mp3",
            choices=["mp3", "wav", "flac"],
        ),
    ) -> CogPath:
        """Generate a song cover by converting the input vocals to a target voice."""
        model_name = f"model_{uuid.uuid4().hex[:12]}"
        model_dir = VOICE_MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract only .pth and .index files from the zip into a flat dir
            with zipfile.ZipFile(str(rvc_model), "r") as zf:
                for member in zf.namelist():
                    stem = Path(member).name
                    if stem and stem.endswith((".pth", ".index")):
                        (model_dir / stem).write_bytes(zf.read(member))

            output_files = run_pipeline(
                source=str(song_input),
                model_name=model_name,
                n_semitones=pitch,
                f0_method=F0Method(f0_method),
                index_rate=index_rate,
                rms_mix_rate=rms_mix_rate,
                protect_rate=protect_rate,
                split_vocals=split_vocals,
                autotune_vocals=autotune_vocals,
                clean_vocals=clean_vocals,
                room_size=room_size,
                wet_level=wet_level,
                dry_level=dry_level,
                main_gain=main_gain,
                inst_gain=inst_gain,
                backup_gain=backup_gain,
                output_format=AudioExt(output_format),
                progress_bar=None,
            )
            return CogPath(output_files[0])
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)
