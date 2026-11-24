#!/bin/sh
# Sign the apps given with the certificate of this project folder, made the first time (and again
# once it has expired) and kept in a keychain of its own in .build. Run from the project folder.
#
# macOS remembers what an app may use (such as the Documents folder the project lies in) by its
# signature. An app signed by its content alone, as Xcode signs it here, is a new app after
# every build, and macOS asks again; an app signed with a certificate stays the same app, and
# macOS asks only once. build_app.sh signs AdaMus.app so, and the tests of the app their builds.
#
# The certificate is signed by itself, so macOS never counts it among the valid identities
# (`find-identity -v`); whether it is there and has not expired is asked instead.
set -eu
KEYCHAIN="$PWD/.build/signing.keychain-db"
PASSWORD=adamus  # guards nothing: the certificate signs only this app, on this Mac
IDENTITY="AdaMus local signing"
VALID_DAYS=3650

usable() {
    security find-identity -p codesigning "$KEYCHAIN" 2>/dev/null | grep -q "\"$IDENTITY\"" &&
        security find-certificate -c "$IDENTITY" -p "$KEYCHAIN" | /usr/bin/openssl x509 -checkend 0 -noout
}

if ! usable; then
    made=$(mktemp -d)
    trap 'rm -rf "$made"' EXIT
    /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days "$VALID_DAYS" -subj "/CN=$IDENTITY" \
        -addext "keyUsage=critical,digitalSignature" -addext "extendedKeyUsage=critical,codeSigning" \
        -addext "basicConstraints=critical,CA:false" \
        -keyout "$made/key.pem" -out "$made/certificate.pem"
    /usr/bin/openssl pkcs12 -export -inkey "$made/key.pem" -in "$made/certificate.pem" \
        -name "$IDENTITY" -passout "pass:$PASSWORD" -out "$made/identity.p12"
    mkdir -p .build
    rm -f "$KEYCHAIN"
    security create-keychain -p "$PASSWORD" "$KEYCHAIN"
    security set-keychain-settings "$KEYCHAIN"  # never locks itself
    security import "$made/identity.p12" -k "$KEYCHAIN" -P "$PASSWORD" -T /usr/bin/codesign >/dev/null
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$PASSWORD" "$KEYCHAIN" >/dev/null
fi
security unlock-keychain -p "$PASSWORD" "$KEYCHAIN"

for app in "$@"; do
    codesign --force --sign "$IDENTITY" --keychain "$KEYCHAIN" "$app"
done
