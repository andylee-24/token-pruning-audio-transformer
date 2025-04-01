#!/bin/bash
sudo mkdir /tmp/ramdisk  # Create a directory for the RAM file system
sudo mount -t tmpfs -o size=4G tmpfs /tmp/ramdisk  # Mount a 4GB RAM file system

# sudo umount /mnt/ramdisk  # Unmount the RAM file system