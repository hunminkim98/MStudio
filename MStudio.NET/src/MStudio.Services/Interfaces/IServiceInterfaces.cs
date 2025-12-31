using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Services.Interfaces
{
    public interface ISessionService : INotifyPropertyChanged
    {
        MotionData? CurrentMotion { get; }
        string? CurrentFilePath { get; }
        bool IsLoading { get; }
        Task LoadMotionAsync(string filePath);
        void LoadFile(string filePath);
        void CloseSession();
    }

    public interface ITimelineService : INotifyPropertyChanged
    {
        int CurrentFrame { get; set; }
        int TotalFrames { get; }
        float FrameRate { get; }
        bool IsPlaying { get; set; }
        double PlaybackSpeed { get; set; }
        bool IsLooping { get; set; }
        bool IsReverse { get; set; }
        TimeSpan CurrentTime { get; }
        void Play();
        void Pause();
        void TogglePlay();
        void StepForward();
        void StepBackward();
    }

    /// <summary>
    /// Abstraction for UI dialogs to maintain Clean Architecture.
    /// ViewModel should not directly depend on platform-specific UI elements.
    /// 
    /// Implementation lives in the App layer (MStudio.App.Services.DialogService)
    /// while the interface lives in the Services layer.
    /// 
    /// This allows:
    /// - Unit testing ViewModels without UI dependencies
    /// - Swapping dialog implementations (WPF, Avalonia, Console, etc.)
    /// - Proper dependency inversion
    /// </summary>
    public interface IDialogService
    {
        /// <summary>
        /// Shows an open file dialog and returns the selected file path.
        /// Returns null if the user cancels.
        /// </summary>
        /// <param name="filter">File filter (e.g., "Text Files|*.txt|All Files|*.*")</param>
        /// <param name="title">Optional dialog title</param>
        /// <returns>Selected file path or null if cancelled</returns>
        string? ShowOpenFileDialog(string filter, string? title = null);

        /// <summary>
        /// Shows a save file dialog and returns the selected file path.
        /// Returns null if the user cancels.
        /// </summary>
        /// <param name="filter">File filter</param>
        /// <param name="defaultFileName">Default file name</param>
        /// <param name="title">Optional dialog title</param>
        /// <returns>Selected file path or null if cancelled</returns>
        string? ShowSaveFileDialog(string filter, string? defaultFileName = null, string? title = null);

        /// <summary>
        /// Shows an information message to the user.
        /// </summary>
        void ShowInfo(string message, string? title = null);

        /// <summary>
        /// Shows a warning message to the user.
        /// </summary>
        void ShowWarning(string message, string? title = null);

        /// <summary>
        /// Shows an error message to the user.
        /// </summary>
        void ShowError(string message, string? title = null);

        /// <summary>
        /// Shows a confirmation dialog with Yes/No options.
        /// </summary>
        /// <returns>True if user confirms, false otherwise</returns>
        bool ShowConfirmation(string message, string? title = null);

        /// <summary>
        /// Shows the analysis selection dialog.
        /// </summary>
        /// <returns>The selected AnalysisType or null if cancelled</returns>
        MStudio.Core.Models.Analysis.AnalysisType? ShowAnalysisSelectionDialog();
    }

    /// <summary>
    /// Service for managing multiple motion capture trials.
    /// Supports loading, selecting, and managing multiple trials simultaneously.
    /// 
    /// Clean Architecture Notes:
    /// - This service manages the collection of trials
    /// - Selection state affects what gets visualized
    /// - Timeline service uses this to determine playback range
    /// </summary>
    public interface ITrialService : INotifyPropertyChanged
    {
        /// <summary>
        /// All loaded trials.
        /// </summary>
        IReadOnlyList<Trial> Trials { get; }
        
        /// <summary>
        /// Currently selected trials for visualization.
        /// </summary>
        IReadOnlyList<Trial> SelectedTrials { get; }
        
        /// <summary>
        /// IDs of currently selected trials.
        /// </summary>
        IReadOnlySet<string> SelectedTrialIds { get; }
        
        /// <summary>
        /// Maximum frame count among selected trials.
        /// Used by TimelineService to determine playback range.
        /// </summary>
        int MaxSelectedFrameCount { get; }
        
        /// <summary>
        /// Maximum frame rate among selected trials.
        /// Used by TimelineService for playback timing.
        /// </summary>
        float MaxSelectedFrameRate { get; }
        
        /// <summary>
        /// Indicates if any trials are currently loaded.
        /// </summary>
        bool HasTrials { get; }
        
        /// <summary>
        /// Indicates if any trials are currently selected.
        /// </summary>
        bool HasSelectedTrials { get; }
        
        /// <summary>
        /// Adds a new trial from the specified file path.
        /// </summary>
        /// <param name="filePath">Path to the motion file</param>
        /// <returns>The newly created Trial</returns>
        Task<Trial> AddTrialAsync(string filePath);
        
        /// <summary>
        /// Removes a trial by its ID.
        /// </summary>
        /// <param name="trialId">The trial ID to remove</param>
        void RemoveTrial(string trialId);
        
        /// <summary>
        /// Clears all trials and releases resources.
        /// </summary>
        void ClearTrials();
        
        /// <summary>
        /// Sets the selection state of a trial.
        /// </summary>
        /// <param name="trialId">The trial ID</param>
        /// <param name="isSelected">Whether the trial should be selected</param>
        void SetTrialSelected(string trialId, bool isSelected);
        
        /// <summary>
        /// Toggles the selection state of a trial.
        /// </summary>
        /// <param name="trialId">The trial ID to toggle</param>
        void ToggleTrialSelection(string trialId);
        
        /// <summary>
        /// Checks if a specific trial is selected.
        /// </summary>
        /// <param name="trialId">The trial ID to check</param>
        /// <returns>True if selected, false otherwise</returns>
        bool IsTrialSelected(string trialId);
        
        /// <summary>
        /// Selects all loaded trials.
        /// </summary>
        void SelectAllTrials();
        
        /// <summary>
        /// Deselects all trials.
        /// </summary>
        void DeselectAllTrials();
        
        /// <summary>
        /// Raised when the trials collection changes (add/remove).
        /// </summary>
        event NotifyCollectionChangedEventHandler? TrialsCollectionChanged;
        
        /// <summary>
        /// Raised when the selection state changes.
        /// </summary>
        event EventHandler? SelectionChanged;
    }
    
    /// <summary>
    /// Service for exporting motion data to various file formats (TRC, C3D).
    /// </summary>
    public interface IExportService
    {
        /// <summary>
        /// Saves the trial data to its original file path (overwrite).
        /// </summary>
        /// <param name="trial">The trial to save</param>
        /// <returns>True if saved successfully, false otherwise</returns>
        Task<bool> SaveAsync(Trial trial);
        
        /// <summary>
        /// Opens a save dialog and saves the trial data to the selected path.
        /// </summary>
        /// <param name="trial">The trial to save</param>
        /// <returns>True if saved successfully, false otherwise</returns>
        Task<bool> SaveAsAsync(Trial trial);
    }
}
