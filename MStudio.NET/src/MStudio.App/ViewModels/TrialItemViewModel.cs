using System;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using MStudio.Core.Models;
using MStudio.Services.Implementations;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// ViewModel for a single Trial item in the Trials list.
    /// Wraps a Trial model and provides UI-bindable properties.
    /// 
    /// Clean Architecture Notes:
    /// - Wraps domain model (Trial) for presentation layer
    /// - Coordinates with ITrialService for selection state changes
    /// - Converts domain values to UI-friendly formats (colors, durations)
    /// </summary>
    public partial class TrialItemViewModel : ObservableObject
    {
        private readonly Trial _trial;
        private readonly ITrialService _trialService;
        
        [ObservableProperty]
        private bool _isSelected;
        
        public TrialItemViewModel(Trial trial, ITrialService trialService)
        {
            _trial = trial;
            _trialService = trialService;
            _isSelected = trialService.IsTrialSelected(trial.Id);
            
            // Subscribe to selection changes from service
            _trialService.SelectionChanged += OnServiceSelectionChanged;
        }
        
        #region Properties
        
        /// <summary>
        /// Unique identifier of the trial.
        /// </summary>
        public string Id => _trial.Id;
        
        /// <summary>
        /// Display name of the trial (typically filename without extension).
        /// </summary>
        public string Name => _trial.Name;
        
        /// <summary>
        /// Total number of frames in the trial.
        /// </summary>
        public int FrameCount => _trial.FrameCount;
        
        /// <summary>
        /// Number of markers in the trial.
        /// </summary>
        public int MarkerCount => _trial.MarkerCount;
        
        /// <summary>
        /// Frame rate in Hz.
        /// </summary>
        public float FrameRate => _trial.FrameRate;
        
        /// <summary>
        /// Color index for visual distinction.
        /// </summary>
        public int ColorIndex => _trial.ColorIndex;
        
        /// <summary>
        /// Duration text formatted as m:ss.fff
        /// </summary>
        public string DurationText
        {
            get
            {
                if (_trial.FrameRate <= 0) return "0:00.000";
                var duration = TimeSpan.FromSeconds(_trial.FrameCount / _trial.FrameRate);
                return $"{(int)duration.TotalMinutes}:{duration.Seconds:D2}.{duration.Milliseconds:D3}";
            }
        }
        
        /// <summary>
        /// Short summary text: Frames / Markers
        /// </summary>
        public string SummaryText => $"{FrameCount:N0} frames • {MarkerCount} markers";
        
        /// <summary>
        /// Color for visual distinction in multi-trial view.
        /// </summary>
        public System.Windows.Media.Color TrialColor
        {
            get
            {
                var palette = TrialService.TrialColorPalette;
                int idx = ColorIndex % palette.Length;
                var c = palette[idx];
                return System.Windows.Media.Color.FromRgb(
                    (byte)(c.R * 255),
                    (byte)(c.G * 255),
                    (byte)(c.B * 255));
            }
        }
        
        /// <summary>
        /// SolidColorBrush for XAML binding.
        /// </summary>
        public SolidColorBrush TrialBrush => new SolidColorBrush(TrialColor);
        
        /// <summary>
        /// The underlying Trial model.
        /// </summary>
        public Trial Trial => _trial;
        
        #endregion
        
        #region Selection Handling
        
        private bool _isDisposed;
        
        partial void OnIsSelectedChanged(bool value)
        {
            // Don't update service if disposed
            if (_isDisposed) return;
            
            // Update service when UI changes selection
            try
            {
                _trialService.SetTrialSelected(_trial.Id, value);
            }
            catch
            {
                // Ignore errors during disposal
            }
        }
        
        private void OnServiceSelectionChanged(object? sender, EventArgs e)
        {
            // Don't update if disposed
            if (_isDisposed) return;
            
            try
            {
                // Sync selection state from service (e.g., when other code changes selection)
                var newState = _trialService.IsTrialSelected(_trial.Id);
                if (newState != _isSelected)
                {
                    // Use SetProperty to avoid triggering OnIsSelectedChanged again
                    SetProperty(ref _isSelected, newState, nameof(IsSelected));
                }
            }
            catch
            {
                // Trial may have been removed, ignore
            }
        }
        
        #endregion
        
        #region Cleanup
        
        /// <summary>
        /// Call this when removing the item to unsubscribe from events.
        /// </summary>
        public void Dispose()
        {
            _isDisposed = true;
            try
            {
                _trialService.SelectionChanged -= OnServiceSelectionChanged;
            }
            catch
            {
                // Ignore errors during unsubscription
            }
        }
        
        #endregion
    }
}
