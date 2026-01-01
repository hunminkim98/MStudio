namespace MStudio.Core.Messaging
{
    /// <summary>
    /// Message types for ViewModel communication via IMessenger (EventAggregator pattern).
    /// 
    /// Clean Architecture:
    /// - Messages are defined in the Core layer (shared across all layers)
    /// - ViewModels subscribe/publish without direct references to each other
    /// - Enables loose coupling and testability
    /// 
    /// Usage:
    /// - Publisher: WeakReferenceMessenger.Default.Send(new MarkerDataChangedMessage());
    /// - Subscriber: WeakReferenceMessenger.Default.Register<MarkerDataChangedMessage>(this, (r, m) => { ... });
    /// </summary>
    
    /// <summary>
    /// Sent when marker data has been modified (e.g., gap filling, smoothing)
    /// and all views displaying marker data should refresh.
    /// </summary>
    public sealed class MarkerDataChangedMessage
    {
        /// <summary>
        /// Index of the marker that was changed. -1 if all markers changed.
        /// </summary>
        public int MarkerIndex { get; init; } = -1;

        /// <summary>
        /// Optional description of what changed.
        /// </summary>
        public string? ChangeDescription { get; init; }
    }

    /// <summary>
    /// Sent when the selected marker changes.
    /// Allows multiple ViewModels to sync their marker selection.
    /// </summary>
    public sealed class MarkerSelectionChangedMessage
    {
        /// <summary>
        /// The new selected marker index.
        /// </summary>
        public int SelectedMarkerIndex { get; init; }

        /// <summary>
        /// The source ViewModel that initiated the selection change.
        /// Used to prevent infinite loops.
        /// </summary>
        public object? Source { get; init; }
    }

    /// <summary>
    /// Sent when the application wants to request a visual refresh
    /// without implying data changes.
    /// </summary>
    public sealed class RefreshVisualsMessage
    {
        /// <summary>
        /// If true, only refresh the 3D viewport.
        /// </summary>
        public bool ViewportOnly { get; init; }

        /// <summary>
        /// If true, only refresh the graph view.
        /// </summary>
        public bool GraphOnly { get; init; }
    }

    /// <summary>
    /// Sent when motion data is loaded or unloaded.
    /// </summary>
    public sealed class MotionDataLoadedMessage
    {
        /// <summary>
        /// True if motion data was loaded, false if unloaded/cleared.
        /// </summary>
        public bool IsLoaded { get; init; }

        /// <summary>
        /// File path of the loaded motion data (if loaded).
        /// </summary>
        public string? FilePath { get; init; }
    }

    /// <summary>
    /// Sent when CMJ analysis is completed to enable visualization in viewport.
    /// </summary>
    public sealed class CMJAnalysisCompletedMessage
    {
        /// <summary>
        /// The CMJ analysis result containing key frames and metrics.
        /// </summary>
        public required Models.Analysis.CMJAnalysisResult Result { get; init; }

        /// <summary>
        /// Whether to show the visualization (false to hide/clear).
        /// </summary>
        public bool ShowVisualization { get; init; } = true;
    }

    /// <summary>
    /// Sent when user requests to navigate to a specific frame from the viewport.
    /// </summary>
    public sealed class NavigateToFrameMessage
    {
        /// <summary>
        /// The target frame to navigate to.
        /// </summary>
        public int Frame { get; init; }

        /// <summary>
        /// Source of the navigation request.
        /// </summary>
        public object? Source { get; init; }
    }
}
