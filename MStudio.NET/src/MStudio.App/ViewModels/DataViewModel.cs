using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// ViewModel for the Data panel.
    /// Displays current file information.
    /// </summary>
    public partial class DataViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;

        /// <summary>
        /// Currently loaded file path
        /// </summary>
        [ObservableProperty]
        private string? _currentFilePath;

        /// <summary>
        /// Number of markers in current motion
        /// </summary>
        [ObservableProperty]
        private int _markerCount;

        /// <summary>
        /// Total frames in current motion
        /// </summary>
        [ObservableProperty]
        private int _frameCount;

        /// <summary>
        /// Frame rate (Hz)
        /// </summary>
        [ObservableProperty]
        private double _frameRate;

        /// <summary>
        /// Duration formatted as M:SS.ms
        /// </summary>
        [ObservableProperty]
        private string _durationText = "--:--";

        /// <summary>
        /// Data validity percentage
        /// </summary>
        [ObservableProperty]
        private double _dataValidity;

        /// <summary>
        /// Whether a file is currently loaded
        /// </summary>
        public bool HasLoadedFile => !string.IsNullOrEmpty(CurrentFilePath);

        /// <summary>
        /// Whether no file is loaded (for empty state)
        /// </summary>
        public bool HasNoFile => string.IsNullOrEmpty(CurrentFilePath);

        /// <summary>
        /// Current file name (without path)
        /// </summary>
        public string? CurrentFileName => Path.GetFileName(CurrentFilePath);

        public DataViewModel(ISessionService sessionService)
        {
            _sessionService = sessionService;

            // Subscribe to session changes
            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    UpdateFileInfo();
                }
            };

            // Initial update in case motion is already loaded
            UpdateFileInfo();
        }

        /// <summary>
        /// Updates file info from current session
        /// </summary>
        private void UpdateFileInfo()
        {
            var motion = _sessionService.CurrentMotion;
            
            if (motion == null)
            {
                CurrentFilePath = null;
                MarkerCount = 0;
                FrameCount = 0;
                FrameRate = 0;
                DurationText = "--:--";
                DataValidity = 0;
            }
            else
            {
                CurrentFilePath = _sessionService.CurrentFilePath;
                MarkerCount = motion.Metadata.MarkerNames.Count;
                FrameCount = motion.Metadata.TotalFrames;
                FrameRate = motion.Metadata.FrameRate;

                // Calculate duration
                if (FrameRate > 0)
                {
                    var seconds = FrameCount / FrameRate;
                    var ts = TimeSpan.FromSeconds(seconds);
                    DurationText = $"{(int)ts.TotalMinutes}:{ts.Seconds:D2}.{ts.Milliseconds / 10:D2}";
                }

                // Calculate data validity (percentage of non-NaN values)
                DataValidity = CalculateDataValidity(motion);
            }

            OnPropertyChanged(nameof(HasLoadedFile));
            OnPropertyChanged(nameof(HasNoFile));
            OnPropertyChanged(nameof(CurrentFileName));
        }

        /// <summary>
        /// Calculates the percentage of valid (non-NaN) data points
        /// </summary>
        private double CalculateDataValidity(Core.Models.MotionData motion)
        {
            var markers = motion.Markers;
            if (markers.FrameCount == 0 || markers.MarkerCount == 0)
                return 0;

            int totalPoints = 0;
            int validPoints = 0;

            for (int m = 0; m < markers.MarkerCount; m++)
            {
                for (int f = 0; f < markers.FrameCount; f++)
                {
                    totalPoints++;
                    var pos = markers.GetPosition(m, f);
                    if (!float.IsNaN(pos.X) && !float.IsNaN(pos.Y) && !float.IsNaN(pos.Z))
                    {
                        validPoints++;
                    }
                }
            }

            return totalPoints > 0 ? (validPoints * 100.0 / totalPoints) : 0;
        }
    }
}
