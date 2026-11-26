"""Names and facts that several parts of AdaMus have to agree on.

Everything an operator may tune is in config.yaml. A value that only one
module needs, and no test or script, is defined in that module. A value that
follows from another one is computed from it and never typed in a second time.
"""

from pathlib import Path

PROGRAM_NAME = "AdaMus"  # as the API and the logs name it
LINE_END = chr(10)  # between the lines of one message: of an error, of a log

# --- Blocks of a session ------------------------------------------------------
REST, FOCUS, FREE = "REST", "FOCUS", "FREE"
BLOCK_MARKERS = {REST: 1, FOCUS: 2, FREE: 3}  # written into the EEG stream at a block start
FOCUS_THRESHOLD = 0.5  # f at the decision threshold of the detector: "focus on" from it up

# --- The focus detector ----------------------------------------------------------
# Its models were trained on exactly these values (adamus.detection.focus_training); a model
# trained on other values is trained again when the program starts.
DETECTOR_ELECTRODES = ("AF7", "AF8", "TP9", "TP10")  # in the order of the models
ARTIFACT_ELECTRODES = ("AF7", "AF8")  # a step is skipped when the signal here is too large
ARTIFACT_UV = 400.0  # only gross artifacts: blinks and eye movements are part of the signal
DETECTOR_FS = 256  # Hz: the Muse, and the recordings the models were trained on
WINDOW_S = 2.0  # the signal one step is computed from
HOP_S = 0.5  # the distance between two steps
CONTEXT_WINDOWS = 2  # the filters run over this many windows, so their start has died away
BROADBAND_HZ = (2.0, 45.0)
BANDS_HZ = (
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 10.0),
    (10.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
    (30.0, 45.0),
)
NOISE_BAND_HZ = BANDS_HZ[-1]  # an electrode without contact, or muscle, fills this band
NOISE_UV = 20.0  # RMS in the noise band above which the artifact electrodes are not clean
FILTER_ORDER = 4  # of every band-pass filter but the eye band's
EYE_BAND_HZ = (0.5, 2.0)  # blinks and eye movements on the forehead
EYE_FILTER_ORDER = 2
MAINS_HZ = (50.0, 60.0)  # hum removed in Europe and in America, where the models' data come from
NOTCH_QUALITY = 30.0
DELAYS_S = (0.0, 0.025, 0.05)  # the broadband signal and its delayed copies, for one covariance
PRIOR_S = 300.0  # the mean of other people counts as this much of the person's own signal
MEMORY_S = 1200.0  # how far back the person's own mean reaches
PRIOR_STEPS = round(PRIOR_S / HOP_S)
MEMORY_STEPS = round(MEMORY_S / HOP_S)
FOCUS_MODEL_FILE = "focus_model.json"  # in FOCUS_DETECTOR_DIR, and copied into every session

# --- What the program obtains itself, when it starts and finds it missing ---------
MODELS_DIR = Path("models")
FOCUS_DETECTOR_DIR = MODELS_DIR / "focus_detector"  # the model and the data it was trained on
FOCUS_MODEL_PATH = FOCUS_DETECTOR_DIR / FOCUS_MODEL_FILE  # relative to the folder it runs in
SETUP_LOG = "setup.log"  # in the folder of what was installed or trained: its output

# --- The generator ------------------------------------------------------------
# Where scripts/install_generator.sh puts things, relative to generator.folder
# (models/music_generator in config.yaml):
GENERATOR_PD_BINARY = "Pd.app/Contents/Resources/bin/pd"
GENERATOR_EXTERNALS = "externals/mrt2~"
GENERATOR_EXTERNAL_FILE = GENERATOR_EXTERNALS + "/mrt2~.pd_darwin"
GENERATOR_ASSETS = "magenta-rt-v2/resources"
GENERATOR_ASSET_FILES = (  # below GENERATOR_ASSETS: what mrt2~ loads as its assets
    "musiccoca/audio_preprocessor.tflite",
    "musiccoca/mapper.tflite",
    "musiccoca/music_encoder.tflite",
    "musiccoca/pretrained_vector_quantizer.tflite",
    "musiccoca/spm.model",
    "musiccoca/text_encoder.tflite",
    "spectrostream/decoder.safetensors",
    "spectrostream/encoder.safetensors",
    "spectrostream/quantizer.safetensors",
    "spectrostream/spectrostream_encoder.mlxfn",
)
GENERATOR_MODEL_FILE = "magenta-rt-v2/models/{model}/{model}.mlxfn"
AUDIO_RATE = 48000  # the only rate the model produces
MODEL_FRAME_S = 0.040  # the model produces audio in frames of this length

# --- Session folder -----------------------------------------------------------
SESSIONS_DIR = Path("sessions")
FILE_CONFIG = "config.yaml"
FILE_META = "meta.json"
FILE_RAW = "raw.csv"
FILE_FILTERED = "filtered.f32"
FILTERED_DTYPE = "<f4"  # little-endian float32
FILE_TICKS = "ticks.jsonl"
FILE_EVENTS = "events.jsonl"
FILE_GENERATOR = "generator.jsonl"  # every message the music generator was sent
FILE_AUDIO = "audio.wav"
FILE_PD_LOG = "pd.log"
FILE_LOG = "adamus.log"
FILE_METRICS = "metrics.json"
FILE_ACTIVITY = "activity.jsonl"  # what the program did around this session
# The sample of a style of music, recorded with its settings (its music at rest, then in focus
# from SAMPLE_FOCUS_FROM_S on), and its cover: in the folder of the style, named by its id, in
# STYLE_FILES, and as a session keeps them, beside its settings. A sample is a WAV, which a
# player puts exactly where it is asked to (in a FLAC it lands seconds off) and the program reads.
FILE_SAMPLE, FILE_COVER = "sample.wav", "cover.png"
STYLE_FILES = Path(__file__).parent / "server" / "styles"
SAMPLE_FOCUS_FROM_S = 10.0

# --- Datasets: <DATASETS_DIR>/<dataset>/<recording>/ holds one recording ----------
DATASETS_DIR = Path("data")
FILE_DATASET_EEG = "eeg.csv"  # a line of electrode names, then one line per sample of them all
FILE_DATASET_PROTOCOL = "protocol.yaml"  # the blocks the samples were recorded in; who, if known
OWN_RECORDINGS = "own_recordings"  # the dataset of the sessions with the headset
TEXT_ENCODING = (
    "utf-8-sig"  # of a text file that a spreadsheet may have begun with a byte order mark
)

# --- What the metrics and the statistics report ---------------------------------
PERCENTILE = 95  # the "high" value reported next to a median

# --- The processes the program starts, and the computer they talk on ------------
LOCAL_HOST = "127.0.0.1"  # Pd and the server of the API listen here only: this computer
STOP_WAIT_S = 5.0  # the longest Pd or the server may take to end when told to
