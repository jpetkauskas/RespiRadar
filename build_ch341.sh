#!/usr/bin/env bash
# Build and install the CH340/CH341 USB-serial driver (and the usbserial core,
# if the kernel lacks it) as out-of-tree modules for the running kernel.
# Usage:  bash build_ch341.sh
set -euo pipefail

KVER=$(uname -r)
KBUILD=/lib/modules/$KVER/build
BASE=$(echo "$KVER" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')   # e.g. 7.0
WORK="$HOME/ch341-build"

echo "==> Kernel $KVER (upstream base v$BASE)"
[ -d "$KBUILD" ] || { echo "!! No kernel headers at $KBUILD — install linux-headers-$KVER first"; exit 1; }

sudo apt-get install -y build-essential curl >/dev/null

# Does the kernel already have the usb-serial core?
CFG="$KBUILD/.config"
NEED_USBSERIAL=1
if [ -f "$CFG" ] && grep -qE '^CONFIG_USB_SERIAL=(y|m)' "$CFG"; then
  NEED_USBSERIAL=0
fi
if [ -f "$CFG" ] && grep -qE '^CONFIG_MODULE_SIG_FORCE=y' "$CFG"; then
  echo "!! Kernel enforces signed modules — an unsigned out-of-tree module will be rejected."
  exit 1
fi
echo "==> usbserial core present in kernel: $([ $NEED_USBSERIAL = 0 ] && echo yes || echo no, building it too)"

mkdir -p "$WORK" && cd "$WORK"
rm -f ./*.c ./*.o ./*.ko Makefile

FILES=(ch341.c)
[ $NEED_USBSERIAL = 1 ] && FILES+=(usb-serial.c generic.c bus.c)

fetch() {  # $1 = ref
  for f in "${FILES[@]}"; do
    curl -fsSL "https://raw.githubusercontent.com/torvalds/linux/$1/drivers/usb/serial/$f" -o "$f" || return 1
  done
}
if fetch "v$BASE"; then
  echo "==> Fetched driver sources from tag v$BASE"
else
  echo "!! Tag v$BASE not found; falling back to master (may not match your kernel exactly)"
  fetch master
fi

{
  echo "obj-m += ch341.o"
  if [ $NEED_USBSERIAL = 1 ]; then
    echo "obj-m += usbserial.o"
    echo "usbserial-y := usb-serial.o generic.o bus.o"
  fi
} > Makefile

echo "==> Building"
make -C "$KBUILD" M="$WORK" modules

echo "==> Installing"
sudo mkdir -p "/lib/modules/$KVER/extra"
sudo cp ./*.ko "/lib/modules/$KVER/extra/"
sudo depmod -a
sudo modprobe ch341
echo ch341 | sudo tee /etc/modules-load.d/ch341.conf >/dev/null

echo "==> Done. Unplug and replug the XM125, then:"
echo "    ls -l /dev/ttyUSB*"
