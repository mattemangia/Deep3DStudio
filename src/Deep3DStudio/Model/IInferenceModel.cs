using System;

namespace Deep3DStudio.Model
{
    /// <summary>
    /// Interface for inference models that support progress reporting.
    /// Used by both pythonnet-based and subprocess-based inference classes.
    /// </summary>
    public interface IInferenceWithProgress
    {
        /// <summary>
        /// Event fired during model loading with (stage, progress 0-1, message)
        /// </summary>
        event Action<string, float, string>? OnLoadProgress;

        /// <summary>
        /// Whether the model is currently loaded
        /// </summary>
        bool IsLoaded { get; }
    }
}
