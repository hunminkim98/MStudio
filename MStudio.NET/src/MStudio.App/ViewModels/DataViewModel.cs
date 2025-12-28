using System;
using System.IO;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using MStudio.Core.Models;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// ViewModel for the Data panel.
    /// Displays information about the currently selected trial(s).
    /// </summary>
    public partial class DataViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITrialService? _trialService;

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
        /// Number of selected trials (for multi-trial info)
        /// </summary>
        [ObservableProperty]
        private int _selectedTrialCount;

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

        // Primary constructor with TrialService support
        public DataViewModel(ISessionService sessionService, ITrialService trialService)
            : this(sessionService)
        {
            _trialService = trialService;
            
            // Subscribe to trial selection changes
            _trialService.SelectionChanged += (s, e) => UpdateFileInfo();
            _trialService.TrialsCollectionChanged += (s, e) => UpdateFileInfo();
        }

        // Legacy constructor for backward compatibility
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
        /// Updates file info from selected trials or current session
        /// </summary>
        private void UpdateFileInfo()
        {
            // Use TrialService if available
            if (_trialService != null)
            {
                UpdateFromTrialService();
                return;
            }
            
            // Fallback to SessionService
            var motion = _sessionService.CurrentMotion;
            
            if (motion == null)
            {
                ClearFileInfo();
            }
            else
            {
                CurrentFilePath = _sessionService.CurrentFilePath;
                MarkerCount = motion.Metadata.MarkerNames.Count;
                FrameCount = motion.Metadata.TotalFrames;
                FrameRate = motion.Metadata.FrameRate;
                SelectedTrialCount = 1;

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
        /// Updates info from TrialService (first selected trial)
        /// </summary>
        private void UpdateFromTrialService()
        {
            if (_trialService == null || !_trialService.HasSelectedTrials)
            {
                ClearFileInfo();
                OnPropertyChanged(nameof(HasLoadedFile));
                OnPropertyChanged(nameof(HasNoFile));
                OnPropertyChanged(nameof(CurrentFileName));
                return;
            }

            var selectedTrials = _trialService.SelectedTrials;
            SelectedTrialCount = selectedTrials.Count;

            // Use first selected trial for basic info
            var firstTrial = selectedTrials[0];
            var motion = firstTrial.MotionData;

            CurrentFilePath = firstTrial.FilePath;
            MarkerCount = motion.Metadata.MarkerNames.Count;
            FrameCount = _trialService.MaxSelectedFrameCount; // Use max for multi-trial
            FrameRate = _trialService.MaxSelectedFrameRate;

            // Calculate duration based on max frame count
            if (FrameRate > 0)
            {
                var seconds = FrameCount / FrameRate;
                var ts = TimeSpan.FromSeconds(seconds);
                DurationText = $"{(int)ts.TotalMinutes}:{ts.Seconds:D2}.{ts.Milliseconds / 10:D2}";
            }

            // Calculate data validity for first trial
            DataValidity = CalculateDataValidity(motion);

            OnPropertyChanged(nameof(HasLoadedFile));
            OnPropertyChanged(nameof(HasNoFile));
            OnPropertyChanged(nameof(CurrentFileName));
        }

        /// <summary>
        /// Clears all file info (no file/trial loaded or selected)
        /// </summary>
        private void ClearFileInfo()
        {
            CurrentFilePath = null;
            MarkerCount = 0;
            FrameCount = 0;
            FrameRate = 0;
            DurationText = "--:--";
            DataValidity = 0;
            SelectedTrialCount = 0;
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
