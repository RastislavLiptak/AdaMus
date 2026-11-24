#!/bin/sh
# Installs the music generator inside this project: Pure Data, the Magenta
# RealTime 2 model files and the mrt2~ external.
#
#   sh scripts/install_generator.sh <folder> <model>
#
# The program runs it itself when it starts and the generator is not there,
# with generator.folder and generator.model of config.yaml, and with the tools
# of the project's Python (mrt) first on the PATH. Steps whose result already
# exists are skipped, so the script can be run again. The paths below the
# folder are the ones src/adamus/constants.py names.
set -eu

FOLDER="$1"
MODEL="$2"

# What is downloaded is checked against what was downloaded when the versions were chosen:
# a server that now sends something else is refused.
PD_VERSION="0.57-0"
PD_URL="https://msp.ucsd.edu/Software/pd-${PD_VERSION}.macos.zip"
PD_SHA256="1025273983272911841e657b2e51f83c160e3b83bb476342a2e75ff62ce133a4"
MRT_VERSION="v2.0.3"
MRT_COMMIT="694a545e4ba0b88bf1150137b129582166d3e07f"
MRT_URL="https://github.com/magenta/magenta-realtime.git"

case "$FOLDER" in
    /*) ROOT="$FOLDER" ;;
    *) ROOT="$(pwd)/$FOLDER" ;;
esac
BUILD="$ROOT/.installing"  # removed at the end: not the .build of a project in the same folder
mkdir -p "$ROOT"

# What mrt2~ loads as its assets, below magenta-rt-v2/resources; the program checks the same.
ASSET_FILES="musiccoca/audio_preprocessor.tflite musiccoca/mapper.tflite musiccoca/music_encoder.tflite musiccoca/pretrained_vector_quantizer.tflite musiccoca/spm.model musiccoca/text_encoder.tflite spectrostream/decoder.safetensors spectrostream/encoder.safetensors spectrostream/quantizer.safetensors spectrostream/spectrostream_encoder.mlxfn"

assets_missing() {
    for file in $ASSET_FILES; do
        [ -f "$ROOT/magenta-rt-v2/resources/$file" ] || return 0
    done
    return 1
}

if [ ! -x "$ROOT/Pd.app/Contents/Resources/bin/pd" ]; then
    echo "== Pure Data $PD_VERSION"
    mkdir -p "$BUILD"
    curl --fail --location --output "$BUILD/pd.zip" "$PD_URL"
    echo "$PD_SHA256  $BUILD/pd.zip" | shasum -a 256 --check --status - \
        || { echo "error: $PD_URL is not the Pure Data $PD_VERSION that was checked" >&2; exit 1; }
    unzip -q -o "$BUILD/pd.zip" -d "$BUILD"            # the archive holds a disk image
    hdiutil attach -quiet -nobrowse -readonly -mountpoint "$BUILD/pd-image" \
        "$BUILD/Pd-$PD_VERSION.dmg"
    trap 'hdiutil detach -quiet "$BUILD/pd-image"' EXIT  # also when copying fails
    rm -rf "$ROOT/Pd.app"
    cp -R "$BUILD/pd-image/Pd-$PD_VERSION.app" "$ROOT/Pd.app"
    hdiutil detach -quiet "$BUILD/pd-image"
    trap - EXIT
fi

if assets_missing; then
    echo "== Magenta RealTime 2: style and codec models"
    mrt models init --download-path "$ROOT/magenta-rt-v2"
fi

if [ ! -f "$ROOT/magenta-rt-v2/models/$MODEL/$MODEL.mlxfn" ]; then
    echo "== Magenta RealTime 2: model $MODEL"
    mrt models download "$MODEL" --download-path "$ROOT/magenta-rt-v2"
fi

if [ ! -f "$ROOT/externals/mrt2~/mrt2~.pd_darwin" ]; then
    echo "== mrt2~ external for Pure Data (needs Xcode with the Metal toolchain)"
    mkdir -p "$BUILD"
    rm -rf "$BUILD/magenta-realtime"
    git clone --quiet --depth 1 --branch "$MRT_VERSION" --recurse-submodules \
        "$MRT_URL" "$BUILD/magenta-realtime"
    [ "$(git -C "$BUILD/magenta-realtime" rev-parse HEAD)" = "$MRT_COMMIT" ] \
        || { echo "error: $MRT_VERSION of $MRT_URL is not the commit that was checked" >&2; exit 1; }
    # A library the build fetches still asks for a CMake older than CMake 4 accepts.
    cmake -S "$BUILD/magenta-realtime" -B "$BUILD/magenta-realtime/build" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    # One compiler per 4 GB of memory: more of them run a 16 GB Mac out of memory.
    JOBS=$(( $(sysctl -n hw.memsize) / 4294967296 ))
    cmake --build "$BUILD/magenta-realtime/build" --target mrt2_pd --parallel "$(( JOBS > 0 ? JOBS : 1 ))"
    mkdir -p "$ROOT/externals/mrt2~"
    cp "$BUILD/magenta-realtime/build/examples/pd/mrt2~.pd_darwin" \
       "$BUILD/magenta-realtime/build/examples/pd/mlx.metallib" "$ROOT/externals/mrt2~/"
fi

rm -rf "$BUILD"
echo "Generator installed in $ROOT"
