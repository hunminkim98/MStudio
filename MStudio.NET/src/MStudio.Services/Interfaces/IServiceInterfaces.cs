using System;
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
    }
}
