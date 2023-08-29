#!/bin/bash
rm -rf `find . -type d -name dataset`
rm -rf `find . -type d -name logs`
echo "Cleaned up all dataset and logs"