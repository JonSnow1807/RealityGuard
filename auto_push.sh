#!/bin/bash
# Auto-push script for immediate discovery documentation

push_discovery() {
    git add -A
    git commit -m "Discovery: $1

    Automated push of new findings
    Timestamp: $(date)"
    git push origin main
}

# Usage: ./auto_push.sh "Description of discovery"
if [ $# -eq 0 ]; then
    echo "Usage: ./auto_push.sh 'Description of discovery'"
else
    push_discovery "$1"
fi