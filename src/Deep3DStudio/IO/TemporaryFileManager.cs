using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Deep3DStudio.IO
{
    /// <summary>
    /// Manages temporary files created by the application and ensures they are cleaned up.
    /// </summary>
    public static class TemporaryFileManager
    {
        private static readonly HashSet<string> _trackedFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private static readonly object _lock = new object();

        /// <summary>
        /// Registers a file to be deleted during cleanup.
        /// </summary>
        public static void RegisterFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return;
            lock (_lock)
            {
                _trackedFiles.Add(Path.GetFullPath(path));
            }
        }

        /// <summary>
        /// Unregisters a file (e.g. if it was already deleted).
        /// </summary>
        public static void UnregisterFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return;
            lock (_lock)
            {
                _trackedFiles.Remove(Path.GetFullPath(path));
            }
        }

        /// <summary>
        /// Performs cleanup of all registered and discovered temporary files.
        /// </summary>
        /// <param name="progressCallback">Callback for progress reporting (message, progress 0-1)</param>
        public static void Cleanup(Action<string, float>? progressCallback = null)
        {
            List<string> filesToDelete;
            lock (_lock)
            {
                filesToDelete = _trackedFiles.ToList();
            }

            // Also discover potential leaked temp files from previous runs
            try
            {
                string tempDir = Path.GetTempPath();
                var discovered = Directory.GetFiles(tempDir, "tmp*.tmp")
                    .Concat(Directory.GetFiles(tempDir, "deep3dstudio_*"))
                    .ToList();
                
                foreach (var file in discovered)
                {
                    if (!filesToDelete.Contains(file, StringComparer.OrdinalIgnoreCase))
                    {
                        // Check if it's likely ours (e.g. .py scripts or if we want to be aggressive with .tmp)
                        // For now, let's only take specific ones to be safe, or all .tmp if requested.
                        if (Path.GetFileName(file).StartsWith("deep3dstudio_", StringComparison.OrdinalIgnoreCase))
                            filesToDelete.Add(file);
                        // Standard .NET temp files are harder to verify as "ours" without tracking,
                        // but we'll include tracked ones.
                    }
                }
            }
            catch { /* Ignore enumeration errors */ }

            int total = filesToDelete.Count;
            int deleted = 0;

            if (total == 0)
            {
                progressCallback?.Invoke("Cleanup complete (no files to delete)", 1.0f);
                return;
            }

            foreach (var file in filesToDelete)
            {
                float progress = (float)deleted / total;
                progressCallback?.Invoke($"Deleting {Path.GetFileName(file)}...", progress);

                try
                {
                    if (File.Exists(file))
                    {
                        File.Delete(file);
                    }
                }
                catch { /* Best effort */ }

                deleted++;
            }

            progressCallback?.Invoke("Cleanup complete", 1.0f);
            
            lock (_lock)
            {
                _trackedFiles.Clear();
            }
        }
    }
}
