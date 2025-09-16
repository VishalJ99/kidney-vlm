#!/bin/bash
# ABOUTME: Convert absolute paths in worklist to relative paths for rsync --files-from
# ABOUTME: Input: worklist with absolute paths, Output: relative paths from source root

# Create relative path list from worklist
# Strip the /vol/biomedic3/histopatho/win_share/ prefix to get relative paths
sed 's|^/vol/biomedic3/histopatho/win_share/||' \
    /data2/vj724/kidney-vlm/pomi_worklists/all_hne_paths.txt \
    > /data2/vj724/kidney-vlm/pomi_worklists/rsync_relative_paths.txt

echo "Created rsync file list with $(wc -l < /data2/vj724/kidney-vlm/pomi_worklists/rsync_relative_paths.txt) files"
echo "First few entries:"
head -3 /data2/vj724/kidney-vlm/pomi_worklists/rsync_relative_paths.txt