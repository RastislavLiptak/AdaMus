#!/bin/sh
# Build AdaMus.app from src/ui with Xcode and put it into the project folder, where it runs the
# program of that folder. Run from the project folder; run again after a change of the app.
# It is signed so that macOS asks only once whether it may use the folder (sign_app.sh).
set -eu
xcodebuild -project src/ui/AdaMus.xcodeproj -scheme AdaMus -configuration Release \
    -destination 'platform=macOS,arch=arm64' -derivedDataPath .build/xcode -quiet build
rm -rf .build/AdaMus.app
ditto .build/xcode/Build/Products/Release/AdaMus.app .build/AdaMus.app
sh "$(dirname "$0")/sign_app.sh" .build/AdaMus.app
rm -rf AdaMus.app
mv .build/AdaMus.app AdaMus.app
echo "Built AdaMus.app"
