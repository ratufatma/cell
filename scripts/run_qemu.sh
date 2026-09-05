#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMINE_VERSION="v9.6.7"
TOOLS_DIR="${ROOT_DIR}/.tools/limine"
LIMINE_DIR=""
ISO_DIR="${ROOT_DIR}/target/cell-iso"
ISO_PATH="${ROOT_DIR}/target/cell.iso"
KERNEL="${ROOT_DIR}/target/x86_64-unknown-none/debug/cell-baremetal"

command -v cargo >/dev/null || { echo "cargo is required" >&2; exit 1; }
command -v xorriso >/dev/null || { echo "xorriso is required" >&2; exit 1; }
command -v qemu-system-x86_64 >/dev/null || { echo "qemu-system-x86_64 is required" >&2; exit 1; }

if [[ ! -x "${TOOLS_DIR}/limine" ]]; then
    mkdir -p "${TOOLS_DIR}"
    archive="${TOOLS_DIR}/limine.tar.gz"
    rm -rf "${TOOLS_DIR}"/Limine-* "${TOOLS_DIR}"/limine-*
    curl --fail --location --output "${archive}" \
        "https://github.com/limine-bootloader/limine/archive/refs/tags/${LIMINE_VERSION}-binary.tar.gz"
    tar -xzf "${archive}" -C "${TOOLS_DIR}"
    LIMINE_DIR="$(find "${TOOLS_DIR}" -mindepth 1 -maxdepth 1 -type d -iname '*limine*' -print -quit)"
    test -n "${LIMINE_DIR}" || { echo "Limine directory was not extracted" >&2; exit 1; }
    make -C "${LIMINE_DIR}"
    mkdir -p "${TOOLS_DIR}/bin"
    cp "${LIMINE_DIR}/limine" "${TOOLS_DIR}/limine"
fi

if [[ -z "${LIMINE_DIR}" ]]; then
    LIMINE_DIR="$(find "${TOOLS_DIR}" -mindepth 1 -maxdepth 1 -type d -iname '*limine*' -print -quit)"
fi

cargo build --target x86_64-unknown-none -p cell-baremetal
rm -rf "${ISO_DIR}"
mkdir -p "${ISO_DIR}/boot/limine"
cp "${KERNEL}" "${ISO_DIR}/boot/cell-baremetal"
cp "${LIMINE_DIR}/limine-bios-cd.bin" "${ISO_DIR}/boot/limine/"
cp "${LIMINE_DIR}/limine-bios.sys" "${ISO_DIR}/boot/limine/"
cat > "${ISO_DIR}/boot/limine/limine.conf" <<'EOF'
timeout: 0
serial: yes
verbose: yes

/CELL
    protocol: limine
    path: boot():/boot/cell-baremetal
EOF
xorriso -as mkisofs -b boot/limine/limine-bios-cd.bin \
    -no-emul-boot -boot-load-size 4 -boot-info-table \
    -partition_offset 16 --protective-msdos-label "${ISO_DIR}" \
    -o "${ISO_PATH}"
"${TOOLS_DIR}/limine" bios-install "${ISO_PATH}"
exec qemu-system-x86_64 -cpu max -cdrom "${ISO_PATH}" -serial stdio -display none -smp 4 -m 512M
