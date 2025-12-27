using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using MStudio.Core.Messaging;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// Main application ViewModel - orchestrates the application state.
    /// 
    /// Clean Architecture Notes:
    /// - Uses IMessenger (WeakReferenceMessenger) for ViewModel-to-ViewModel communication
    /// - Does not directly call methods on child ViewModels for data refresh
    /// - Publishes MarkerDataChangedMessage after data modifications
    /// - Other ViewModels subscribe and react independently
    /// </summary>
    public partial class MainViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;
        private readonly IDialogService _dialogService;

        // Track selected marker locally (synced via messaging)
        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

        [ObservableProperty]
        private string _statusText = "Ready";

        [ObservableProperty]
        private string _title = "MStudio - Motive Grade Workspace";

        // ViewportViewModel for 3D rendering (exposed for DataContext binding in XAML)
        public MStudioViewportViewModel ViewportViewModel { get; }
        
        // GraphViewModel for time-series data (exposed for DataContext binding in XAML)
        public GraphViewModel GraphViewModel { get; }

        public MainViewModel(
            ISessionService sessionService, 
            ITimelineService timelineService,
            IDialogService dialogService,
            MStudioViewportViewModel viewportViewModel, 
            GraphViewModel graphViewModel)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;
            _dialogService = dialogService;
            ViewportViewModel = viewportViewModel;
            GraphViewModel = graphViewModel;

            // Re-broadcast property changes from services if needed
            _timelineService.PropertyChanged += (s, e) => 
            {
                if (e.PropertyName == nameof(ITimelineService.CurrentFrame))
                    OnPropertyChanged(nameof(CurrentFrame));
                if (e.PropertyName == nameof(ITimelineService.TotalFrames))
                    OnPropertyChanged(nameof(TotalFrames));
                if (e.PropertyName == nameof(ITimelineService.IsPlaying))
                    OnPropertyChanged(nameof(IsPlaying));
                if (e.PropertyName == nameof(ITimelineService.CurrentTime))
                    OnPropertyChanged(nameof(CurrentTimeText));
            };

            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    OnPropertyChanged(nameof(CurrentMotionName));
                }
            };

            // Subscribe to marker selection changes from child ViewModels
            WeakReferenceMessenger.Default.Register<MarkerSelectionChangedMessage>(this, (r, m) =>
            {
                // Update our local tracking (for FillGaps/SmoothData to use)
                // Use SetProperty to avoid triggering PropertyChanged unnecessarily
                if (m.Source != this && m.SelectedMarkerIndex != SelectedMarkerIndex)
                {
                    SetProperty(ref _selectedMarkerIndex, m.SelectedMarkerIndex, nameof(SelectedMarkerIndex));
                }
            });
        }

        public bool IsPlaying => _timelineService.IsPlaying;

        public int CurrentFrame
        {
            get => _timelineService.CurrentFrame;
            set => _timelineService.CurrentFrame = value;
        }

        public int TotalFrames => _timelineService.TotalFrames;

        public double PlaybackSpeed
        {
            get => _timelineService.PlaybackSpeed;
            set
            {
                _timelineService.PlaybackSpeed = Math.Clamp(value, 0.0, 10.0);
                OnPropertyChanged();
            }
        }

        public bool IsLooping
        {
            get => _timelineService.IsLooping;
            set
            {
                _timelineService.IsLooping = value;
                OnPropertyChanged();
            }
        }

        public bool IsReverse
        {
            get => _timelineService.IsReverse;
            set
            {
                _timelineService.IsReverse = value;
                OnPropertyChanged();
            }
        }

        public string CurrentTimeText
        {
            get
            {
                var t = _timelineService.CurrentTime;
                return $"{t.Minutes:D1}:{t.Seconds:D2}:{t.Milliseconds:D3}";
            }
        }

        public string CurrentMotionName => _sessionService.CurrentMotion != null 
            ? Path.GetFileName(_sessionService.CurrentMotion.Metadata.FilePath) 
            : "No Motion Loaded";

        [RelayCommand]
        private void Play() => _timelineService.Play();

        [RelayCommand]
        private void Pause() => _timelineService.Pause();

        [RelayCommand]
        private void TogglePlay() => _timelineService.TogglePlay();

        [RelayCommand]
        private void ResetSpeed() => PlaybackSpeed = 1.0;

        [RelayCommand]
        private void StepForward() => _timelineService.StepForward();

        [RelayCommand]
        private void StepBackward() => _timelineService.StepBackward();

        /// <summary>
        /// Fills gaps in the selected marker's data using interpolation.
        /// Uses message-based communication to notify ViewModels of data change.
        /// </summary>
        [RelayCommand]
        private void FillGaps()
        {
            var motion = _sessionService.CurrentMotion;
            
            if (motion != null && SelectedMarkerIndex >= 0)
            {
                motion.Markers.FillGaps(SelectedMarkerIndex, 100); // Default 100 frames
                
                // Notify all interested ViewModels via messaging (Clean Architecture)
                WeakReferenceMessenger.Default.Send(new MarkerDataChangedMessage
                {
                    MarkerIndex = SelectedMarkerIndex,
                    ChangeDescription = "Gap filling completed"
                });
                
                StatusText = $"Filled gaps for marker {SelectedMarkerIndex}";
            }
        }

        /// <summary>
        /// Smooths the selected marker's data using a moving average filter.
        /// Uses message-based communication to notify ViewModels of data change.
        /// </summary>
        [RelayCommand]
        private void SmoothData()
        {
            var motion = _sessionService.CurrentMotion;

            if (motion != null && SelectedMarkerIndex >= 0)
            {
                motion.Markers.SmoothData(SelectedMarkerIndex, 5); // Default 5 frame window

                // Notify all interested ViewModels via messaging (Clean Architecture)
                WeakReferenceMessenger.Default.Send(new MarkerDataChangedMessage
                {
                    MarkerIndex = SelectedMarkerIndex,
                    ChangeDescription = "Data smoothing completed"
                });
                
                StatusText = $"Smoothed data for marker {SelectedMarkerIndex}";
            }
        }

        /// <summary>
        /// Opens a file using the abstracted dialog service.
        /// This maintains Clean Architecture by not depending on platform-specific UI elements.
        /// </summary>
        [RelayCommand]
        private async Task OpenFile()
        {
            const string filter = "Motion Files (*.trc; *.c3d; *.json)|*.trc;*.c3d;*.json|All files (*.*)|*.*";
            
            var filePath = _dialogService.ShowOpenFileDialog(filter, "Open Motion File");
            
            if (filePath != null)
            {
                StatusText = $"Loading {Path.GetFileName(filePath)}...";
                try
                {
                    await _sessionService.LoadMotionAsync(filePath);
                    StatusText = $"Loaded {Path.GetFileName(filePath)}";
                }
                catch (Exception ex)
                {
                    _dialogService.ShowError($"Failed to load file: {ex.Message}", "Load Error");
                    StatusText = $"Error: {ex.Message}";
                }
            }
        }
    }
}

