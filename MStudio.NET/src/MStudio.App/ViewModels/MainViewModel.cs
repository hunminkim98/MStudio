using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;

        [ObservableProperty]
        private string _statusText = "Ready";

        [ObservableProperty]
        private string _title = "MStudio - Motive Grade Workspace";

        // ViewportViewModel for 3D rendering
        public MStudioViewportViewModel ViewportViewModel { get; }
        
        // GraphViewModel for time-series data
        public GraphViewModel GraphViewModel { get; }

        public MainViewModel(ISessionService sessionService, ITimelineService timelineService, 
                           MStudioViewportViewModel viewportViewModel, GraphViewModel graphViewModel)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;
            ViewportViewModel = viewportViewModel;
            GraphViewModel = graphViewModel;

            // Sync marker selection between Viewport and Graph
            ViewportViewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(MStudioViewportViewModel.SelectedMarkerIndex))
                {
                    GraphViewModel.SelectedMarkerIndex = ViewportViewModel.SelectedMarkerIndex;
                }
            };
            
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

        [RelayCommand]
        private void FillGaps()
        {
            var motion = _sessionService.CurrentMotion;
            int selectedIdx = ViewportViewModel.SelectedMarkerIndex;
            
            if (motion != null && selectedIdx >= 0)
            {
                motion.Markers.FillGaps(selectedIdx, 100); // Default 100 frames
                
                // Refresh visuals
                ViewportViewModel.UpdateMarkersAndVisuals();
                GraphViewModel.UpdatePoints(); // UpdatePoints is private, I need to make it public or trigger via property change
            }
        }

        [RelayCommand]
        private void SmoothData()
        {
            var motion = _sessionService.CurrentMotion;
            int selectedIdx = ViewportViewModel.SelectedMarkerIndex;

            if (motion != null && selectedIdx >= 0)
            {
                motion.Markers.SmoothData(selectedIdx, 5); // Default 5 frame window

                // Refresh visuals
                ViewportViewModel.UpdateMarkersAndVisuals();
                GraphViewModel.UpdatePoints();
            }
        }

        [RelayCommand]
        private async Task OpenFile()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Motion Files (*.trc; *.c3d; *.json)|*.trc;*.c3d;*.json|All files (*.*)|*.*"
            };

            if (dialog.ShowDialog() == true)
            {
                StatusText = $"Loading {Path.GetFileName(dialog.FileName)}...";
                try
                {
                    await _sessionService.LoadMotionAsync(dialog.FileName);
                    StatusText = $"Loaded {Path.GetFileName(dialog.FileName)}";
                }
                catch (Exception ex)
                {
                    StatusText = $"Error: {ex.Message}";
                }
            }
        }
    }
}
