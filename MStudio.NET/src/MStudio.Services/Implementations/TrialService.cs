using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MStudio.Core.Models;
using MStudio.Core.Parsers;
using MStudio.Services.Interfaces;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// Service for managing multiple motion capture trials.
    /// Implements Clean Architecture principles by:
    /// - Using parsers from Core layer for file loading
    /// - Providing observable properties for UI binding
    /// - Firing events for collection and selection changes
    /// </summary>
    public class TrialService : ITrialService, IDisposable
    {
        private readonly IEnumerable<IFileParser> _parsers;
        private readonly ObservableCollection<Trial> _trials;
        private readonly HashSet<string> _selectedTrialIds;
        private int _nextColorIndex;

        // Predefined color palette for trial distinction (12 colors)
        public static readonly (float R, float G, float B)[] TrialColorPalette = new[]
        {
            (0.2f, 0.8f, 0.3f),  // Green
            (0.3f, 0.6f, 1.0f),  // Blue
            (1.0f, 0.5f, 0.2f),  // Orange
            (0.9f, 0.2f, 0.4f),  // Red/Pink
            (0.7f, 0.4f, 0.9f),  // Purple
            (1.0f, 0.85f, 0.2f), // Yellow
            (0.2f, 0.9f, 0.8f),  // Cyan
            (0.95f, 0.5f, 0.7f), // Pink
            (0.5f, 0.8f, 0.5f),  // Light Green
            (0.4f, 0.4f, 0.9f),  // Indigo
            (0.9f, 0.6f, 0.4f),  // Coral
            (0.6f, 0.9f, 0.3f),  // Lime
        };

        public TrialService(IEnumerable<IFileParser> parsers)
        {
            _parsers = parsers;
            _trials = new ObservableCollection<Trial>();
            _selectedTrialIds = new HashSet<string>();
            _nextColorIndex = 0;

            _trials.CollectionChanged += OnTrialsCollectionChanged;
        }

        #region ITrialService Properties

        public IReadOnlyList<Trial> Trials => _trials;

        public IReadOnlyList<Trial> SelectedTrials => 
            _trials.Where(t => _selectedTrialIds.Contains(t.Id)).ToList();

        public IReadOnlySet<string> SelectedTrialIds => _selectedTrialIds;

        public int MaxSelectedFrameCount => 
            SelectedTrials.Count > 0 
                ? SelectedTrials.Max(t => t.FrameCount) 
                : 0;

        public float MaxSelectedFrameRate => 
            SelectedTrials.Count > 0 
                ? SelectedTrials.Max(t => t.FrameRate) 
                : 30.0f;

        public bool HasTrials => _trials.Count > 0;

        public bool HasSelectedTrials => _selectedTrialIds.Count > 0;

        #endregion

        #region Trial Management

        public async Task<Trial> AddTrialAsync(string filePath)
        {
            var parser = _parsers.FirstOrDefault(p => p.CanParse(filePath));
            if (parser == null)
            {
                throw new NotSupportedException($"No parser found for file: {filePath}");
            }

            var motionData = await parser.ParseAsync(filePath);

            var trial = new Trial
            {
                Id = Guid.NewGuid().ToString(),
                Name = Path.GetFileNameWithoutExtension(filePath),
                MotionData = motionData,
                ColorIndex = _nextColorIndex
            };

            _nextColorIndex = (_nextColorIndex + 1) % TrialColorPalette.Length;

            _trials.Add(trial);

            // Auto-select newly added trial
            SetTrialSelected(trial.Id, true);

            OnPropertyChanged(nameof(Trials));
            OnPropertyChanged(nameof(HasTrials));

            return trial;
        }

        public void RemoveTrial(string trialId)
        {
            var trial = _trials.FirstOrDefault(t => t.Id == trialId);
            if (trial != null)
            {
                _selectedTrialIds.Remove(trialId);
                _trials.Remove(trial);
                trial.MotionData.Markers.Dispose();

                OnPropertyChanged(nameof(Trials));
                OnPropertyChanged(nameof(HasTrials));
                OnPropertyChanged(nameof(SelectedTrials));
                OnPropertyChanged(nameof(HasSelectedTrials));
                OnPropertyChanged(nameof(MaxSelectedFrameCount));
                OnPropertyChanged(nameof(MaxSelectedFrameRate));
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        public void ClearTrials()
        {
            // Early exit if nothing to clear
            if (_trials.Count == 0) return;
            
            // Clear selection first before disposing
            _selectedTrialIds.Clear();
            
            // Notify selection changed first (so listeners can react before items are removed)
            SelectionChanged?.Invoke(this, EventArgs.Empty);
            
            // Now dispose and clear trials
            foreach (var trial in _trials.ToList()) // Use ToList to avoid modification during iteration
            {
                try
                {
                    trial.MotionData.Markers.Dispose();
                }
                catch
                {
                    // Ignore dispose errors
                }
            }

            _trials.Clear();
            _nextColorIndex = 0;

            OnPropertyChanged(nameof(Trials));
            OnPropertyChanged(nameof(HasTrials));
            OnPropertyChanged(nameof(SelectedTrials));
            OnPropertyChanged(nameof(HasSelectedTrials));
            OnPropertyChanged(nameof(MaxSelectedFrameCount));
            OnPropertyChanged(nameof(MaxSelectedFrameRate));
        }

        #endregion

        #region Selection Management

        public void SetTrialSelected(string trialId, bool isSelected)
        {
            var trial = _trials.FirstOrDefault(t => t.Id == trialId);
            if (trial == null) return;

            bool changed = false;
            if (isSelected && !_selectedTrialIds.Contains(trialId))
            {
                _selectedTrialIds.Add(trialId);
                changed = true;
            }
            else if (!isSelected && _selectedTrialIds.Contains(trialId))
            {
                _selectedTrialIds.Remove(trialId);
                changed = true;
            }

            if (changed)
            {
                OnPropertyChanged(nameof(SelectedTrials));
                OnPropertyChanged(nameof(SelectedTrialIds));
                OnPropertyChanged(nameof(HasSelectedTrials));
                OnPropertyChanged(nameof(MaxSelectedFrameCount));
                OnPropertyChanged(nameof(MaxSelectedFrameRate));
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        public void ToggleTrialSelection(string trialId)
        {
            bool currentlySelected = _selectedTrialIds.Contains(trialId);
            SetTrialSelected(trialId, !currentlySelected);
        }

        public bool IsTrialSelected(string trialId)
        {
            return _selectedTrialIds.Contains(trialId);
        }

        public void SelectAllTrials()
        {
            bool anyChange = false;
            foreach (var trial in _trials)
            {
                if (!_selectedTrialIds.Contains(trial.Id))
                {
                    _selectedTrialIds.Add(trial.Id);
                    anyChange = true;
                }
            }

            if (anyChange)
            {
                OnPropertyChanged(nameof(SelectedTrials));
                OnPropertyChanged(nameof(SelectedTrialIds));
                OnPropertyChanged(nameof(HasSelectedTrials));
                OnPropertyChanged(nameof(MaxSelectedFrameCount));
                OnPropertyChanged(nameof(MaxSelectedFrameRate));
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        public void DeselectAllTrials()
        {
            if (_selectedTrialIds.Count > 0)
            {
                _selectedTrialIds.Clear();
                OnPropertyChanged(nameof(SelectedTrials));
                OnPropertyChanged(nameof(SelectedTrialIds));
                OnPropertyChanged(nameof(HasSelectedTrials));
                OnPropertyChanged(nameof(MaxSelectedFrameCount));
                OnPropertyChanged(nameof(MaxSelectedFrameRate));
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        public void DeleteMarker(string trialId, int markerIndex)
        {
            var trial = _trials.FirstOrDefault(t => t.Id == trialId);
            if (trial == null) return;
            
            if (markerIndex < 0 || markerIndex >= trial.MarkerCount) return;

            // 1. Mutate data container (Shared reference)
            // Note: Since MotionData is a record, 'Markers' is a reference type (MarkerDataContainer)
            // so we can modify it in place.
            trial.MotionData.Markers.RemoveMarker(markerIndex);
            
            // 2. Update Metadata to reflect name removal
            // Since Metadata is a record, we must create a new one.
            var nameList = trial.MotionData.Metadata.MarkerNames.ToList();
            nameList.RemoveAt(markerIndex);
            
            var newMetadata = trial.MotionData.Metadata with
            {
                MarkerNames = nameList.AsReadOnly()
            };
            
            // 3. Create new MotionData with updated Metadata
            var newMotionData = trial.MotionData with { Metadata = newMetadata };
            
            // 4. Create new Trial with updated MotionData
            var newTrial = trial with { MotionData = newMotionData };
            
            // 5. Replace in ObservableCollection to notify listeners
            // This triggers CollectionChanged -> ViewModels will refresh
            int index = _trials.IndexOf(trial);
            if (index >= 0)
            {
                _trials[index] = newTrial;
            }
            
            // 6. If this trial was selected, we should probably raise SelectionChanged 
            // to force full refresh of active visualization
            if (IsTrialSelected(trialId))
            {
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        #endregion

        #region Events

        public event NotifyCollectionChangedEventHandler? TrialsCollectionChanged;
        public event EventHandler? SelectionChanged;
        public event PropertyChangedEventHandler? PropertyChanged;

        private void OnTrialsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            TrialsCollectionChanged?.Invoke(this, e);
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            ClearTrials();
            _trials.CollectionChanged -= OnTrialsCollectionChanged;
        }

        #endregion
    }
}
